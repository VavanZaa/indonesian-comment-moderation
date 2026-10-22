"""CLI driving the three annotators.

    python -m src.annotate.run --config config/local.yaml [--annotators a1,a2,a3]
        [--limit N] [--backend llamacpp|transformers|stub] [--split dev|heldout]

Reads `data/interim/comments_normalized.parquet` (build contract §2, produced by
`src.normalize.text`), filters to `--split` (default `dev`; `--split heldout` exists
only for the Channel E run - AGENTS.md R3), runs the requested annotators, and writes:

    data/interim/votes_a1.parquet   comment_id, annotator_id, signal, raw_json, confidence, platform
    data/interim/votes_a2.parquet   (same shape)
    data/interim/votes_a3.parquet   comment_id, annotator_id, label,  raw_json, confidence, platform
    models.lock.json                model identity per annotator, merged across runs

R9/D35: comments now carry a `platform` column (youtube | instagram | x). The
annotators themselves are platform-agnostic - they read only `text_normalized` - but this
module carries `platform` through to the vote parquets so `src/report.py` can break
results down per platform later. That column is additive beyond the build-contract
table in docs/05-build-contract.md §3, which predates the D35 platform pivot; downstream
code that reads the documented columns by name is unaffected.

`models.lock.json` intentionally holds ONLY model identity (model id, revision,
quantization, backend, and A2's rule-file hash) - never per-run counts like "N comments
annotated". That file is inside the frozen surface (R3): the dev run writes it, the
operator locks it, and the heldout run must reproduce byte-identical content or
`freeze.verify()` correctly reports drift. Run-specific numbers (A3's off-schema count
in particular) are printed to the console instead, every run, per AGENTS.md's
instruction to "count off-schema emissions and report them at the end of a run."
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..schemas import Comment, HarmVote, LabelVote
from .a1_indobertweet import A1IndoBERTweet, harm_counts
from .a2_rules import A2Rules, family_counts
from .a3_llm import A3LocalLLM, off_schema_summary

DEFAULT_CONFIG = "config/local.yaml"
DEFAULT_INPUT = Path("data/interim/comments_normalized.parquet")
VOTES_A1_PATH = Path("data/interim/votes_a1.parquet")
VOTES_A2_PATH = Path("data/interim/votes_a2.parquet")
VOTES_A3_PATH = Path("data/interim/votes_a3.parquet")
LOCK_PATH = Path("models.lock.json")

#: The held-out run writes BESIDE the development votes, never over them (R3).
#:
#: Both runs annotate with the same frozen system, so the natural implementation is one
#: set of output paths. That is wrong here: `--split heldout` would overwrite the
#: development votes, `python -m src.consensus` would then rebuild `consensus.parquet`
#: from Channel E alone, and every dev-side number - the phrase families, the baseline
#: arms, the dev column of the transfer table - would silently become held-out numbers.
#: The freeze would not catch it, because no frozen file changed.
HELDOUT_DIR = Path("data/interim/heldout")

#: A3's resume checkpoint. Not an output and not part of the frozen surface - purely a
#: crash guard for the one stage that takes over an hour. See `_load_a3_cache`.
A3_CACHE_NAME = "a3_generations.jsonl"

ALL_ANNOTATORS: tuple[str, ...] = ("a1", "a2", "a3")
SPLITS: frozenset[str] = frozenset({"dev", "heldout"})

_DEV_VOTE_PATHS: dict[str, Path] = {
    "a1": VOTES_A1_PATH, "a2": VOTES_A2_PATH, "a3": VOTES_A3_PATH,
}


def votes_path(annotator: str, split: str = "dev") -> Path:
    """Where one annotator's votes go for a given split.

    Development keeps the documented paths (build contract §3); the held-out run is
    namespaced under `data/interim/heldout/` so the two never collide.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {sorted(SPLITS)}, got {split!r}")
    dev = _DEV_VOTE_PATHS[annotator]
    return dev if split == "dev" else HELDOUT_DIR / dev.name


def a3_cache_path(split: str = "dev", root: str | Path = ".") -> Path:
    """Where A3's resume checkpoint lives, namespaced by split like the vote parquets."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {sorted(SPLITS)}, got {split!r}")
    rel = (Path("data/interim") if split == "dev" else HELDOUT_DIR) / A3_CACHE_NAME
    return _resolve_path(root, rel)


# --------------------------------------------------------------------------- A3 resume
#
# A full development run is ~16k sequential generations - well over an hour of GPU time
# in a single process that, until this existed, held every result in memory and wrote
# nothing until the last comment finished. A crash at comment 15,900 threw away the
# entire run. A3 now appends each raw generation to a JSONL sidecar and replays it on
# restart.
#
# The danger in any such cache is that it silently answers a question nobody asked
# again: reusing generations produced by a different model, a different prompt, or a
# different corpus would put stale labels into a real run while every file downstream
# looked freshly computed - and the freeze could not catch it, because a checkpoint is
# not a frozen artefact. So the file carries a header pinning A3's full
# `model_identity()` (model hash, prompt hash, backend, token cap) plus a hash of the
# input corpus, and a cache whose header does not match byte for byte is not used. It is
# moved aside rather than deleted, so a stale checkpoint stays available for inspection
# instead of vanishing at the moment someone would want to ask what was in it.


def _corpus_fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def a3_cache_guard(identity: Mapping[str, Any], split: str, corpus_sha256: str) -> dict[str, Any]:
    """Everything that must be identical for a cached generation to still be valid."""
    return {"split": split, "corpus_sha256": corpus_sha256, "identity": dict(identity)}


def _load_a3_cache(path: Path, guard: Mapping[str, Any]) -> dict[str, str]:
    """Replayable generations from a previous run, or `{}` if there are none to trust."""
    if not path.is_file():
        return {}
    try:
        with path.open(encoding="utf-8") as fh:
            header_line = fh.readline()
            header = json.loads(header_line) if header_line.strip() else {}
            if header.get("guard") != dict(guard):
                stale = path.with_suffix(path.suffix + ".stale")
                print(
                    f"[annotate] a3: checkpoint at {path} was produced by a different "
                    "model, prompt or corpus - NOT reusing it. Moving it to "
                    f"{stale.name} and regenerating from scratch.",
                    file=sys.stderr,
                )
                fh.close()
                path.replace(stale)
                return {}
            cached: dict[str, str] = {}
            for line in fh:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    # A crash mid-write can leave one truncated final line. Everything
                    # before it is still good; stop here and regenerate the rest.
                    break
                cid, raw = rec.get("comment_id"), rec.get("raw")
                if isinstance(cid, str) and isinstance(raw, str):
                    cached[cid] = raw
            return cached
    except OSError as exc:
        print(f"[annotate] a3: could not read checkpoint {path} ({exc}); regenerating.",
              file=sys.stderr)
        return {}


def _open_a3_cache(path: Path, guard: Mapping[str, Any], resumed: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    if resumed:
        return path.open("a", encoding="utf-8")
    fh = path.open("w", encoding="utf-8")
    fh.write(json.dumps({"guard": dict(guard)}, sort_keys=True, ensure_ascii=False) + "\n")
    fh.flush()
    return fh


def _progress_printer(every: int = 250, label: str = "a3"):
    """Print throughput and an ETA periodically. A silent hour-long stage is unreviewable."""
    start = time.monotonic()

    def report(done: int, total: int) -> None:
        if done != total and (done % every or done == 0):
            return
        elapsed = time.monotonic() - start
        rate = done / elapsed if elapsed > 0 else 0.0
        remaining = (total - done) / rate if rate > 0 else float("nan")
        print(
            f"[annotate] {label}: {done}/{total} "
            f"({done / total:.1%}) {rate:.1f} comments/s "
            f"elapsed {elapsed / 60:.1f}m eta {remaining / 60:.1f}m",
            flush=True,
        )

    return report


# --------------------------------------------------------------------------- config


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    import yaml  # pyyaml is a hard dependency; kept local so imports stay cheap

    p = Path(path)
    if not p.is_file():
        print(
            f"[annotate] warning: config {p} not found - using contract defaults.",
            file=sys.stderr,
        )
        return {}
    with p.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise SystemExit(f"config {p} did not parse to a mapping")
    return cfg


def _resolve_path(root: str | Path, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else Path(root) / path


# --------------------------------------------------------------------------- comments in


def _row_to_comment(row: Mapping[str, Any]) -> Comment:
    """Build a `Comment` from whatever columns the input parquet actually has.

    `schemas.Comment` (frozen, not owned by this lane) predates the R9/D35
    multi-platform pivot: it has `channel_id`/`video_id`, not the newer
    `source_account`/`post_id`. Neither field is read by any annotator (they read only
    `comment_id` and `text_normalized`/`text_original`), so this falls back through the
    old and new column names rather than requiring either.
    """
    return Comment(
        comment_id=str(row["comment_id"]),
        channel_id=str(row.get("channel_id") or row.get("source_account") or row.get("figure_id") or ""),
        video_id=str(row.get("video_id") or row.get("post_id") or ""),
        author_hash=str(row.get("author_hash") or ""),
        published_at=str(row.get("published_at") or ""),
        text_original=str(row.get("text_original") or ""),
        text_normalized=str(row.get("text_normalized") or ""),
    )


def load_comments(
    input_path: str | Path,
    *,
    split: str = "dev",
    limit: int | None = None,
):
    """Read the normalized parquet, filter to `split`, return `(comments, dataframe)`.

    The dataframe is returned alongside the `Comment` list so callers can pull columns
    (like `platform`) that `Comment` itself does not carry.
    """
    import pandas as pd

    if split not in SPLITS:
        raise ValueError(f"split must be one of {sorted(SPLITS)}, got {split!r}")

    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(
            f"{input_path} not found. Run the normalize stage first:\n"
            f"    python -m src.normalize.text --config config/local.yaml"
        )

    df = pd.read_parquet(input_path)
    if "comment_id" not in df.columns:
        raise ValueError(f"{input_path} has no comment_id column (found: {list(df.columns)})")

    if "split" in df.columns:
        df = df[df["split"] == split].reset_index(drop=True)
    else:
        print(
            f"[annotate] warning: {input_path} has no `split` column - annotating all "
            f"{len(df)} rows as if split={split!r}.",
            file=sys.stderr,
        )

    if limit is not None:
        df = df.head(int(limit)).reset_index(drop=True)

    comments = [_row_to_comment(r) for r in df.to_dict("records")]
    return comments, df


def _platform_map(df) -> dict[str, str]:
    """comment_id -> platform, or {} if the input predates the D35 platform column."""
    if "platform" not in df.columns:
        return {}
    return {str(cid): str(p) for cid, p in zip(df["comment_id"], df["platform"])}


# --------------------------------------------------------------------------- votes out


def harm_votes_to_frame(votes: Sequence[HarmVote], platform_map: Mapping[str, str]):
    import pandas as pd

    rows = [
        {
            "comment_id": v.comment_id,
            "annotator_id": v.annotator_id,
            "signal": v.signal.value,
            "raw_json": json.dumps(v.raw, ensure_ascii=False, sort_keys=True, default=str),
            "confidence": v.confidence,
            "platform": platform_map.get(v.comment_id, ""),
        }
        for v in votes
    ]
    columns = ["comment_id", "annotator_id", "signal", "raw_json", "confidence", "platform"]
    return pd.DataFrame(rows, columns=columns)


def label_votes_to_frame(votes: Sequence[LabelVote], platform_map: Mapping[str, str]):
    import pandas as pd

    rows = [
        {
            "comment_id": v.comment_id,
            "annotator_id": v.annotator_id,
            "label": v.label.value,
            "raw_json": json.dumps(v.raw, ensure_ascii=False, sort_keys=True, default=str),
            "confidence": v.confidence,
            "platform": platform_map.get(v.comment_id, ""),
        }
        for v in votes
    ]
    columns = ["comment_id", "annotator_id", "label", "raw_json", "confidence", "platform"]
    return pd.DataFrame(rows, columns=columns)


def _write_parquet(df, path: Path, root: str | Path) -> Path:
    out = _resolve_path(root, path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, engine="pyarrow", index=False)
    return out


# --------------------------------------------------------------------------- builders


def build_a1(cfg: Mapping[str, Any], backend: str) -> A1IndoBERTweet:
    a1cfg = ((cfg.get("annotators") or {}).get("a1_indobertweet")) or {}
    # A1 only ever has two real backends (it is a classifier, not a generator): the
    # global --backend flag maps straight onto them, `llamacpp`/`transformers` -> the
    # real checkpoint path, `stub` -> the keyword stand-in.
    resolved_backend = "stub" if backend == "stub" else "transformers"
    return A1IndoBERTweet(
        checkpoint=a1cfg.get("checkpoint"),
        backend=resolved_backend,
        base_model=a1cfg.get("base_model", "indolem/indobertweet-base-uncased"),
        revision=a1cfg.get("revision"),
        batch_size=int(a1cfg.get("batch_size", 32)),
        max_length=int(a1cfg.get("max_length", 128)),
        threshold=float(a1cfg.get("threshold", 0.5)),
    )


def build_a2(cfg: Mapping[str, Any]) -> A2Rules:
    a2cfg = ((cfg.get("annotators") or {}).get("a2_rules")) or {}
    lexicon_dir = a2cfg.get("lexicon_dir", "data/raw/lexicons")
    return A2Rules(lexicon_dir=lexicon_dir)


def build_a3(cfg: Mapping[str, Any], backend: str) -> A3LocalLLM:
    a3cfg = ((cfg.get("annotators") or {}).get("a3_llm")) or {}
    kwargs: dict[str, Any] = dict(
        model=a3cfg.get("model"),
        backend=backend,
        max_new_tokens=int(a3cfg.get("max_new_tokens", 24)),
        quantization=a3cfg.get("quantization"),
        revision=a3cfg.get("revision"),
    )
    # n_gpu_layers: -1 (full offload) unless the config overrides it. See
    # a3_llm.py::A3LocalLLM._load_llamacpp for why a machine might need 0 here -
    # not a performance knob by default, a documented hardware workaround.
    if "n_gpu_layers" in a3cfg:
        kwargs["n_gpu_layers"] = int(a3cfg["n_gpu_layers"])
    if "n_threads" in a3cfg:
        kwargs["n_threads"] = int(a3cfg["n_threads"])
    if "n_ctx" in a3cfg:
        kwargs["n_ctx"] = int(a3cfg["n_ctx"])
    return A3LocalLLM(**kwargs)


# --------------------------------------------------------------------------- lockfile


def _load_lock(root: str | Path) -> dict[str, Any]:
    p = _resolve_path(root, LOCK_PATH)
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _write_lock(root: str | Path, identities: Mapping[str, Any]) -> Path:
    p = _resolve_path(root, LOCK_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(identities, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                 encoding="utf-8")
    return p


# --------------------------------------------------------------------------- run


def run(
    config: Mapping[str, Any] | None = None,
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    input_path: str | Path | None = None,
    root: str | Path = ".",
    annotators: Sequence[str] = ALL_ANNOTATORS,
    limit: int | None = None,
    backend: str = "stub",
    split: str = "dev",
    resume: bool = False,
) -> dict[str, Any]:
    """Run the requested annotators and write vote parquets + models.lock.json.

    Returns a summary dict (per-annotator counts, output paths) so callers - including
    tests - do not have to scrape stdout.

    `resume=True` turns on A3's crash checkpoint (see `_load_a3_cache`). It is off by
    default so that a plain call generates everything fresh and touches no sidecar file;
    the long real runs opt in.
    """
    requested = list(dict.fromkeys(annotators))  # de-dupe, keep order
    unknown = [a for a in requested if a not in ALL_ANNOTATORS]
    if unknown:
        raise ValueError(f"unknown annotator(s) {unknown}; choose from {ALL_ANNOTATORS}")
    if not requested:
        raise ValueError("no annotators requested")

    if config is None:
        config = load_config(config_path)

    resolved_input = Path(input_path) if input_path is not None else _resolve_path(root, DEFAULT_INPUT)
    comments, df = load_comments(resolved_input, split=split, limit=limit)
    platform_map = _platform_map(df)
    if "platform" not in df.columns:
        print(
            "[annotate] warning: input has no `platform` column (pre-D35 ingest output) "
            "- writing platform='' to the vote parquets.",
            file=sys.stderr,
        )
    print(f"[annotate] {len(comments)} comments loaded (split={split!r}) from {resolved_input}")

    identities = _load_lock(root)
    summary: dict[str, Any] = {"split": split, "n_comments": len(comments), "paths": {}}

    if "a1" in requested:
        a1 = build_a1(config, backend)
        votes = a1.annotate(comments)
        out = _write_parquet(harm_votes_to_frame(votes, platform_map),
                             votes_path("a1", split), root)
        identities["a1"] = a1.model_identity()
        counts = harm_counts(votes)
        summary["a1"] = {"n_votes": len(votes), "counts": counts}
        summary["paths"]["a1"] = str(out)
        print(f"[annotate] a1: {len(votes)} votes -> {out}  {counts}")

    if "a2" in requested:
        a2 = build_a2(config)
        votes = a2.annotate(comments)
        out = _write_parquet(harm_votes_to_frame(votes, platform_map),
                             votes_path("a2", split), root)
        identities["a2"] = a2.model_identity()
        counts = family_counts(votes)
        summary["a2"] = {"n_votes": len(votes), "counts": counts}
        summary["paths"]["a2"] = str(out)
        print(f"[annotate] a2: {len(votes)} votes -> {out}  {counts}")

    if "a3" in requested:
        a3 = build_a3(config, backend)
        # Computed before annotating, not after: it is what the resume checkpoint is
        # guarded on, so it has to be known before a single generation is replayed.
        a3_identity = a3.model_identity()
        cache_path = a3_cache_path(split, root)
        cached: dict[str, str] = {}
        cache_fh = None
        if resume:
            guard = a3_cache_guard(a3_identity, split, _corpus_fingerprint(resolved_input))
            cached = _load_a3_cache(cache_path, guard)
            wanted = {c.comment_id for c in comments}
            replayable = len(wanted & cached.keys())
            if replayable:
                print(
                    f"[annotate] a3: resuming - {replayable}/{len(comments)} generations "
                    f"replayed from {cache_path}, {len(comments) - replayable} to generate."
                )
            cache_fh = _open_a3_cache(cache_path, guard, resumed=bool(cached))

        def _record(comment_id: str, raw: str) -> None:
            if cache_fh is None:
                return
            cache_fh.write(
                json.dumps({"comment_id": comment_id, "raw": raw}, ensure_ascii=False) + "\n"
            )
            cache_fh.flush()  # a checkpoint that is still buffered when we crash is not one

        try:
            votes = a3.annotate(
                comments,
                cache=cached or None,
                on_generate=_record,
                progress=_progress_printer(),
            )
        finally:
            if cache_fh is not None:
                cache_fh.close()
        out = _write_parquet(label_votes_to_frame(votes, platform_map),
                             votes_path("a3", split), root)
        identities["a3"] = a3_identity
        off = off_schema_summary(a3)
        summary["a3"] = {"n_votes": len(votes), **off}
        summary["paths"]["a3"] = str(out)
        print(f"[annotate] a3: {off['n_votes']}/{off['n_total']} votes written -> {out}")
        # AGENTS.md / build contract: off-schema emissions are counted and reported at
        # the end of a run, never silently coerced.
        if off["n_off_schema"]:
            print(
                f"[annotate] a3: {off['n_off_schema']} OFF-SCHEMA generation(s) - "
                "dropped, NOT coerced to a default label. Samples "
                f"(comment_id, raw): {off['off_schema_examples'][:5]}",
                file=sys.stderr,
            )
        else:
            print("[annotate] a3: 0 off-schema generations")

    lock_out = _write_lock(root, identities)
    summary["paths"]["lock"] = str(lock_out)
    print(f"[annotate] model identity -> {lock_out}")
    return summary


# --------------------------------------------------------------------------- CLI


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m src.annotate.run",
        description="Run A1/A2/A3 over the normalized comment corpus.",
    )
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--annotators", default="a1,a2,a3", help="comma-separated subset of a1,a2,a3")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--backend", choices=("llamacpp", "transformers", "stub"), default="stub")
    ap.add_argument("--split", choices=("dev", "heldout"), default="dev")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="checkpoint A3's generations and replay them if the run is restarted "
             "(only reused when model, prompt and corpus are unchanged)",
    )
    ap.add_argument("--root", default=".", help="repo root the output paths hang off")
    ap.add_argument("--input", default=None, help="override the normalized-comments parquet path")
    args = ap.parse_args(argv)

    requested = tuple(a.strip() for a in args.annotators.split(",") if a.strip())
    try:
        run(
            config_path=args.config,
            input_path=args.input,
            root=args.root,
            annotators=requested,
            limit=args.limit,
            backend=args.backend,
            split=args.split,
            resume=args.resume,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[annotate] ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
