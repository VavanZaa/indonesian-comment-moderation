"""The shared multi-platform ingest schema and row builder.  **SHARED CONTRACT.**

`docs/05-build-contract.md` §1 / `docs/01-decision-log.md` D35 / `AGENTS.md` R9.

Every connector (`youtube.py`, `instagram.py`, `x.py`, `tiktok.py`) and the fixture generator
(`fixture.py`, owned by a different lane) build rows through the functions in this
module and nothing else, so the on-disk schema, the `comment_id` convention, R8's
hashing choke point and R9's split-per-figure guarantee all live in exactly one place.

**R8** lives at `build_row`/`hash_author`: a raw author id is consumed and hashed the
moment a row is built. No function in this module accepts an already-built row and
returns one containing a raw handle - `author_hash` is the only thing that survives the
call.

**R9** lives at `load_figures`/`validate_frame`: the split is asserted once per
`figure_id`, never per source or account, and both the config-parsing entry point and
the assembled-corpus entry point raise `SplitLeakError` if a figure's sources ever
disagree about which split they belong to. Provenance strings reaching a display
artefact are built from `provenance_string`/`provenance_for_posts`, which only ever
see the pseudonymous alias from `build_figure_aliases` - never a real account name.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "COLUMNS",
    "PLATFORMS",
    "SPLITS",
    "COLLECTION_MODES",
    "RAW_PATH",
    "DEV_MANIFEST_PATH",
    "FIGURE_ALIASES_PATH",
    "DEFAULT_CONFIG",
    "SplitLeakError",
    "load_config",
    "author_hash_salt",
    "hash_author",
    "make_comment_id",
    "build_row",
    "provenance_string",
    "provenance_for_posts",
    "load_figures",
    "validate_frame",
    "write_corpus",
    "write_dev_manifest",
    "build_figure_aliases",
    "write_figure_aliases",
    "write_collection",
    "normalize_timestamp",
    "read_import_file",
    "import_files_for_account",
]

# --------------------------------------------------------------------------- paths
RAW_PATH = "data/raw/comments_raw.parquet"
DEV_MANIFEST_PATH = "data/processed/dev_manifest.json"
FIGURE_ALIASES_PATH = "data/processed/figure_aliases.json"
DEFAULT_CONFIG = "config/local.yaml"

# --------------------------------------------------------------------------- schema
#: Exact parquet column order (build contract §1). `comment_id` is globally unique
#: (`"{platform}:{native_id}"`); every other column is per the contract table, plus
#: `provenance`, the pseudonymized display string (R9).
COLUMNS: tuple[str, ...] = (
    "comment_id",
    "platform",
    "figure_id",
    "source_account",
    "post_id",
    "post_url",
    "parent_id",
    "author_hash",
    "published_at",
    "text_original",
    "split",
    "genre",
    "collection_mode",
    "provenance",
    "like_count",
)

#: Instagram and TikTok live collection use unofficial, lazily imported scrapers;
#: import mode remains available when live collection is inaccessible or inappropriate.
PLATFORMS: frozenset[str] = frozenset({"youtube", "instagram", "x", "tiktok"})
SPLITS: frozenset[str] = frozenset({"dev", "heldout"})
#: "fixture" is the fixture generator's own tag (a different lane); connectors in this
#: module produce "api", "scrape", or "import".
COLLECTION_MODES: frozenset[str] = frozenset({"api", "scrape", "import", "fixture"})

_PLATFORM_DISPLAY: dict[str, str] = {
    "youtube": "YouTube", "instagram": "Instagram", "x": "X", "tiktok": "TikTok",
}


class SplitLeakError(ValueError):
    """R9: a `figure_id` was declared under more than one split.

    The held-out unit is the FIGURE, across every platform, never the channel/account.
    Holding out one YouTube channel while training on the same person's Instagram leaks
    the held-out set through the figure, and the freeze cannot catch it because nothing
    drifted - so this is checked mechanically, twice: once while parsing the config
    (`load_figures`) and once again over the assembled corpus (`validate_frame`).
    """


# --------------------------------------------------------------------------- config
def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    import yaml  # pyyaml is a hard dependency; kept local so imports stay cheap

    p = Path(path)
    if not p.is_file():
        raise SystemExit(
            f"config not found: {p}\n"
            "Copy config/poc.yaml to config/local.yaml and fill it in."
        )
    with p.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise SystemExit(f"config {p} did not parse to a mapping")
    return cfg


def author_hash_salt(config: Mapping[str, Any]) -> str:
    """R8: the salt that makes `author_hash` non-reversible by anyone holding the
    parquet. Required - ingest refuses to run without one."""
    privacy = config.get("privacy") or {}
    youtube = config.get("youtube") or {}
    salt = privacy.get("author_hash_salt") or youtube.get("author_hash_salt")
    if not salt or not str(salt).strip():
        raise ValueError(
            "privacy.author_hash_salt is missing from the config.\n"
            "R8 requires author ids to be hashed at ingest with a salt; refusing to "
            "collect without one."
        )
    return str(salt)


# --------------------------------------------------------------------------- R8
def hash_author(author_id: str, salt: str) -> str:
    """`sha256(salt + author_id).hexdigest()[:16]` - build contract §1.

    Call this the instant a raw handle/author id is parsed out of an API response or
    an imported export row. The raw value must not survive past the call: it is not
    stored, not returned, and not logged anywhere in this package. `build_row` is the
    only place a row is meant to call this from.
    """
    if not salt or not str(salt).strip():
        raise ValueError("author_hash_salt is required (R8) - refusing to hash without one")
    return hashlib.sha256((str(salt) + str(author_id or "")).encode("utf-8")).hexdigest()[:16]


def make_comment_id(platform: str, native_id: Any) -> str:
    """`"{platform}:{native_id}"` - native ids collide across apps (build contract §1)."""
    if platform not in PLATFORMS:
        raise ValueError(f"unknown platform {platform!r}, expected one of {sorted(PLATFORMS)}")
    if native_id is None or str(native_id).strip() == "":
        raise ValueError("native_id is required to build a comment_id")
    return f"{platform}:{native_id}"


def build_row(
    *,
    platform: str,
    figure_id: str,
    source_account: str,
    post_id: str,
    post_url: str,
    parent_id: str | None,
    author_id: str | None,
    salt: str,
    published_at: str,
    text_original: str,
    split: str,
    genre: str,
    collection_mode: str,
    provenance: str,
    like_count: int = 0,
) -> dict[str, Any]:
    """Build one output record - every column except `comment_id`.

    R8 is enforced here, not downstream: `author_id` is consumed and hashed via
    `hash_author` and never appears in the returned dict, only `author_hash` does.
    Callers must not retain the raw `author_id` past this call.

    `comment_id` is deliberately not a parameter - different platforms discover their
    native comment id at different points in the parsing (a YouTube reply's id lives
    two levels deeper than its thread), so callers finish the row themselves::

        row = build_row(platform="youtube", figure_id=fig["id"], ..., author_id=raw_id,
                         salt=salt, provenance=prov)
        row["comment_id"] = make_comment_id("youtube", native_comment_id)
    """
    if platform not in PLATFORMS:
        raise ValueError(f"unknown platform {platform!r}, expected one of {sorted(PLATFORMS)}")
    if split not in SPLITS:
        raise ValueError(f"split must be one of {sorted(SPLITS)}, got {split!r}")
    if collection_mode not in COLLECTION_MODES:
        raise ValueError(
            f"collection_mode must be one of {sorted(COLLECTION_MODES)}, got {collection_mode!r}"
        )
    return {
        "platform": str(platform),
        "figure_id": str(figure_id),
        "source_account": str(source_account),
        "post_id": str(post_id),
        "post_url": str(post_url or ""),
        "parent_id": None if parent_id in (None, "") else str(parent_id),
        "author_hash": hash_author(author_id, salt),
        "published_at": str(published_at),
        "text_original": str(text_original),
        "split": str(split),
        "genre": str(genre),
        "collection_mode": str(collection_mode),
        "provenance": str(provenance),
        "like_count": int(like_count),
    }


# --------------------------------------------------------------------------- R9 provenance
def provenance_string(platform: str, figure_alias: str, post_index: int, post_total: int) -> str:
    """R9: the pseudonymized "which post, which app" string shown on screen.

    `figure_alias` must be the display alias from `build_figure_aliases`
    (`"Figure A"`), never a real account name.

    >>> provenance_string("youtube", "Figure A", 4, 15)
    'YouTube · Figure A · post 4 of 15'
    """
    display = _PLATFORM_DISPLAY.get(platform, platform)
    return f"{display} · {figure_alias} · post {post_index} of {post_total}"


def provenance_for_posts(platform: str, figure_alias: str, post_ids: Sequence[str]) -> dict[str, str]:
    """`{post_id: provenance_string(...)}` for a connector's discovered post order.

    `post_total` is the number of posts actually collected for this figure/platform
    source, and `post_index` is 1-based, in the order the connector discovered them.
    """
    total = len(post_ids)
    return {
        str(pid): provenance_string(platform, figure_alias, i + 1, total)
        for i, pid in enumerate(post_ids)
    }


def _alias_suffix(n: int) -> str:
    """0 -> A, 1 -> B, ..., 25 -> Z, 26 -> AA, ... Excel-column style, so the alphabet
    never runs out."""
    n += 1
    letters: list[str] = []
    while n > 0:
        n, rem = divmod(n - 1, 26)
        letters.append(chr(ord("A") + rem))
    return "".join(reversed(letters))


def build_figure_aliases(figures: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """`figure_id -> "Figure A"`, `"Figure B"`, ... in config order.

    This mapping is the only thing that is allowed to reach a display artefact (R9);
    the account/channel names behind each figure stay in the config.
    """
    return {str(fig["id"]): f"Figure {_alias_suffix(i)}" for i, fig in enumerate(figures)}


def write_figure_aliases(aliases: Mapping[str, str], path: str | Path) -> None:
    """Write `data/processed/figure_aliases.json`.

    `figure_id -> alias` only - no account names, no platform/source data. This file
    is itself a display artefact (R9) and must never contain a real account name.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(dict(sorted(aliases.items())), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- R9 figures/config
def load_figures(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Parse the `figures:` block (build contract §1).

    Returns one dict per figure: `{id, genre, split, sources: [{platform, account,
    mode}, ...]}`, in config order, where a source may also carry the optional
    `max_videos` / `max_comments` per-source collection limits. Raises
    `SplitLeakError` (a `ValueError`) if the same `figure_id` is declared under both
    `dev` and `heldout` - R9's one hard rule: split is assigned per figure, never per
    account or source.
    """
    raw = config.get("figures")
    if not raw:
        raise ValueError(
            "config has no `figures:` block (docs/05-build-contract.md §1 / AGENTS.md R9). "
            "The pre-D35 flat `youtube.development_channels` / `heldout_channel` shape is "
            "not enough on its own any more: every figure now needs a `sources:` list "
            "spanning the platforms it was collected from."
        )

    by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for entry in raw:
        if not isinstance(entry, Mapping):
            raise ValueError(f"figures entry must be a mapping, got {entry!r}")
        fid = str(entry.get("id") or "").strip()
        if not fid:
            raise ValueError(f"figures entry missing `id`: {entry!r}")
        split = str(entry.get("split") or "").strip().lower()
        if split not in SPLITS:
            raise ValueError(
                f"figure {fid!r}: split must be one of {sorted(SPLITS)}, got {entry.get('split')!r}"
            )
        genre = str(entry.get("genre") or "")

        sources: list[dict[str, Any]] = []
        for src in entry.get("sources") or []:
            if not isinstance(src, Mapping):
                raise ValueError(f"figure {fid!r}: source entry must be a mapping, got {src!r}")
            platform = str(src.get("platform") or "").strip().lower()
            if platform not in PLATFORMS:
                raise ValueError(
                    f"figure {fid!r}: source platform must be one of {sorted(PLATFORMS)}, "
                    f"got {src.get('platform')!r}"
                )
            mode = str(src.get("mode") or "api").strip().lower()
            if mode not in ("api", "import", "scrape") or (
                mode == "scrape" and platform not in ("instagram", "tiktok")
            ):
                raise ValueError(
                    f"figure {fid!r}/{platform}: invalid source mode (scrape is IG/TikTok only), "
                    f"got {src.get('mode')!r}"
                )
            account = str(src.get("account") or "")
            if not account:
                raise ValueError(f"figure {fid!r}/{platform}: source is missing `account`")
            parsed: dict[str, Any] = {"platform": platform, "account": account, "mode": mode}
            # Optional per-source collection limits (2026-09-09). Kept an explicit
            # allowlist rather than `dict(src)`: this dict is what every connector
            # sees, and passing arbitrary YAML through it is how an unvalidated key
            # ends up quietly steering collection. Absent => the connector's own
            # channel-wide config default applies.
            for limit_key in ("max_videos", "max_comments"):
                if src.get(limit_key) is not None:
                    value = int(src[limit_key])
                    if value <= 0:
                        raise ValueError(
                            f"figure {fid!r}/{platform}: {limit_key} must be a positive "
                            f"integer, got {src[limit_key]!r}"
                        )
                    parsed[limit_key] = value
            sources.append(parsed)

        if fid in by_id:
            if by_id[fid]["split"] != split:
                raise SplitLeakError(
                    f"R9 violation: figure {fid!r} has sources declared under both "
                    f"{by_id[fid]['split']!r} and {split!r} splits. split is assigned per "
                    "figure, never per account: every source of a held-out figure must be "
                    "held out with it, or the freeze cannot detect the leak."
                )
            by_id[fid]["sources"].extend(sources)
            if genre and not by_id[fid]["genre"]:
                by_id[fid]["genre"] = genre
        else:
            by_id[fid] = {"id": fid, "genre": genre, "split": split, "sources": sources}
            order.append(fid)

    return [by_id[fid] for fid in order]


def validate_frame(df) -> None:
    """Raise on any schema violation or R9 split leak in an assembled corpus.

    Defense in depth: `load_figures` catches a split leak at config-parsing time;
    this catches it again over the rows actually produced, in case two figure
    entries were merged incorrectly upstream or a connector mislabelled a row.
    """
    cols = list(df.columns)
    missing = [c for c in COLUMNS if c not in cols]
    extra = [c for c in cols if c not in COLUMNS]
    if missing or extra:
        raise ValueError(
            f"schema violation: missing columns {missing}, unexpected columns {extra}. "
            f"expected exactly {list(COLUMNS)}"
        )

    for c in COLUMNS:
        if c == "parent_id":
            continue  # null for every top-level comment, by design
        if df[c].isna().any():
            raise ValueError(f"schema violation: column {c!r} has null/missing values")

    bad_platform = sorted(set(df["platform"]) - PLATFORMS)
    if bad_platform:
        raise ValueError(f"schema violation: unknown platform(s) {bad_platform}")

    bad_split = sorted(set(df["split"]) - SPLITS)
    if bad_split:
        raise ValueError(f"schema violation: unknown split(s) {bad_split}")

    bad_mode = sorted(set(df["collection_mode"]) - COLLECTION_MODES)
    if bad_mode:
        raise ValueError(f"schema violation: unknown collection_mode(s) {bad_mode}")

    if len(df):
        ids = df["comment_id"].astype(str)
        dupes = sorted(set(ids[ids.duplicated()]))
        if dupes:
            raise ValueError(
                f"schema violation: comment_id must be globally unique, duplicates: {dupes[:5]}"
            )

        prefixes = ids.str.split(":", n=1).str[0]
        mismatched = int((prefixes != df["platform"]).sum())
        if mismatched:
            raise ValueError(
                "schema violation: comment_id must be '{platform}:{native_id}' - "
                f"{mismatched} row(s) do not match their own platform column"
            )

        leak = df.groupby("figure_id")["split"].nunique()
        leaked_figures = sorted(leak[leak > 1].index)
        if leaked_figures:
            raise SplitLeakError(
                f"R9 violation: figure_id(s) {leaked_figures} appear with more than one "
                "split in the assembled corpus. split is assigned per figure, never per "
                "account or platform."
            )


# --------------------------------------------------------------------------- writers
def write_corpus(rows, path: str | Path) -> None:
    """Write `rows` (a DataFrame, or a sequence of row dicts) to `path` as parquet,
    reindexed onto `COLUMNS` exactly and validated (`validate_frame`) first."""
    import pandas as pd

    df = rows if hasattr(rows, "columns") else pd.DataFrame(list(rows))
    df = df.reindex(columns=list(COLUMNS))
    validate_frame(df)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, engine="pyarrow", index=False)


def write_dev_manifest(
    df,
    config: Mapping[str, Any],
    path: str | Path,
    *,
    collected_at: str | None = None,
) -> None:
    """Write `data/processed/dev_manifest.json` - DEV ROWS ONLY (R3/D31).

    Including a held-out comment id here would leak the held-out figure into the
    frozen surface, which is exactly what the freeze is meant to protect.
    """
    dev = df[df["split"] == "dev"] if len(df) else df

    figures_cfg = load_figures(config)
    figures_out: list[dict[str, Any]] = []
    for fig in figures_cfg:
        if fig["split"] != "dev":
            continue
        sources_out = []
        for src in fig["sources"]:
            mode = src["mode"]
            if len(dev):
                mask = ((dev["figure_id"] == fig["id"]) & (dev["platform"] == src["platform"])
                        & (dev["source_account"] == src["account"]))
                observed = sorted(set(dev.loc[mask, "collection_mode"]))
                if observed:
                    mode = observed[0]
            sources_out.append(
                {"platform": src["platform"], "account": src["account"], "collection_mode": mode}
            )
        figures_out.append({"id": fig["id"], "genre": fig["genre"], "sources": sources_out})

    platforms_count: dict[str, int] = {}
    if len(dev):
        for platform, n in dev["platform"].value_counts().items():
            platforms_count[str(platform)] = int(n)

    ids = sorted(str(c) for c in dev["comment_id"]) if len(dev) else []
    digest = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()[:16]

    if len(df):
        heldout_ids = set(str(c) for c in df.loc[df["split"] == "heldout", "comment_id"])
        leaked = heldout_ids.intersection(ids)
        if leaked:  # pragma: no cover - validate_frame already guards this upstream
            raise RuntimeError(
                f"R3 violation: held-out ids reached the dev manifest: {sorted(leaked)[:5]}"
            )

    payload = {
        "figures": figures_out,
        "platforms": platforms_count,
        "n_comments": len(ids),
        "comment_ids_sha256": digest,
        "collected_at": collected_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_collection(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    root: str | Path = ".",
) -> Any:
    """Validate `rows` and write all three shared ingest artefacts - the raw corpus,
    the dev manifest, and the figure alias map - then print the R9 collection-mix
    summary. Used by `collect.py` and by every single-platform connector CLI, so they
    all write byte-for-byte the same shape of output. Returns the assembled DataFrame.
    """
    import pandas as pd

    df = pd.DataFrame(list(rows))
    df = df.reindex(columns=list(COLUMNS))
    validate_frame(df)

    figures = load_figures(config)
    aliases = build_figure_aliases(figures)

    raw_path = Path(root) / RAW_PATH
    manifest_path = Path(root) / DEV_MANIFEST_PATH
    aliases_path = Path(root) / FIGURE_ALIASES_PATH

    write_corpus(df, raw_path)
    write_dev_manifest(df, config, manifest_path)
    write_figure_aliases(aliases, aliases_path)

    print(f"\nwrote {len(df)} comments -> {raw_path.as_posix()}")
    print(f"dev manifest (dev rows only, R3) -> {manifest_path.as_posix()}")
    print(f"figure aliases (R9, pseudonymized) -> {aliases_path.as_posix()}")
    if len(df):
        print("by split:  " + ", ".join(f"{k}={v}" for k, v in sorted(df["split"].value_counts().items())))
        print("collection mix (R9 - the report must disclose this):")
        mix = df.groupby(["platform", "collection_mode"]).size().sort_index()
        for (platform, mode), n in mix.items():
            print(f"  {platform:<10} {mode:<8} {n}")
    return df


# --------------------------------------------------------------------------- timestamps
_EPOCH_RE = re.compile(r"^-?\d+$")
_OFFSET_RE = re.compile(r"([+-]\d{2})(\d{2})$")


def _from_epoch(value: float) -> str:
    n = float(value)
    if abs(n) > 10**12:  # looks like milliseconds, not seconds
        n /= 1000.0
    return datetime.fromtimestamp(n, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_timestamp(value: Any) -> str:
    """Normalize a platform-native timestamp to UTC ISO-8601, `%Y-%m-%dT%H:%M:%SZ`.

    Covers what the connectors actually send: YouTube's `publishedAt`, Instaloader's
    datetime, X API v2's `created_at`, and TikTok/import epoch seconds or milliseconds.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError("published_at is required and cannot be normalized from an empty value")

    if isinstance(value, bool):
        raise ValueError(f"published_at cannot be a bool: {value!r}")
    if isinstance(value, (int, float)):
        return _from_epoch(value)

    s = str(value).strip()
    if _EPOCH_RE.match(s):
        return _from_epoch(int(s))

    iso = s.replace("Z", "+00:00").replace("z", "+00:00")
    iso = _OFFSET_RE.sub(r"\1:\2", iso)
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError as exc:
        raise ValueError(f"could not normalize timestamp {value!r} to UTC ISO-8601") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- import mode
def read_import_file(path: str | Path) -> list[dict[str, Any]]:
    """Read one local export file - JSON (a list of objects, or `{"comments": [...]}` /
    `{"data": [...]}`) or CSV (header row) - into plain dicts. Connectors map
    platform-specific field names onto `build_row`'s arguments from there.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    suffix = p.suffix.lower()
    if suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, Mapping):
            data = data.get("comments") or data.get("data") or []
        if not isinstance(data, list):
            raise ValueError(f"{p}: expected a JSON list of comment objects")
        return [dict(row) for row in data]
    if suffix == ".csv":
        with p.open(encoding="utf-8-sig", newline="") as fh:
            return [dict(row) for row in csv.DictReader(fh)]
    raise ValueError(f"{p}: unsupported import file type {suffix!r} (expected .json or .csv)")


def import_files_for_account(root: str | Path, platform: str, account: str) -> list[Path]:
    """Export files for one source account under `data/raw/imports/{platform}/`.

    Matches `{account}.json` / `{account}.csv` exactly, plus any
    `{account}__*.json` / `{account}__*.csv` for a multi-file export, sorted for
    determinism. Returns `[]` (never raises) when the directory or file is absent -
    the caller logs and skips, per the "missing input is a skip, not a crash" rule.
    """
    d = Path(root) / "data" / "raw" / "imports" / platform
    if not d.is_dir():
        return []
    exact = [d / f"{account}.json", d / f"{account}.csv"]
    found = [p for p in exact if p.is_file()]
    prefixed = sorted(
        p for p in d.iterdir()
        if p.is_file()
        and p.suffix.lower() in (".json", ".csv")
        and p.stem.startswith(f"{account}__")
    )
    return found + prefixed
