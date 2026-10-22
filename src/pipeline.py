"""Pipeline driver — runs the development-channel stages in order.

    ingest -> normalize -> annotate -> consensus -> mine -> families -> baseline

WHY THIS STOPS BEFORE `freeze lock` (AGENTS.md R3, D19)
-------------------------------------------------------
It would be one more line to call `src.freeze.lock()` at the end of this file, and it
would destroy the only guarantee the freeze provides.

The freeze is not a checksum utility. It is a claim about TIMING: that the system which
scored the held-out figure is the system that existed *before anyone looked at that
figure*. That claim is made by a person who knows they have finished with the development
channels and have not glanced at Channel E. A pipeline cannot know either of those things.
Auto-locking would produce a lockfile on every run, including the run you did after
noticing the normalizer mangles a slang form in the held-out data — which is precisely the
failure R3 exists to detect.

So this driver prints the two commands and stops. Do not "improve" it by automating them.

The held-out scoring stages are absent from `STAGES` for the same reason: they run after a
deliberate lock, from `python -m src.report heldout`, never as part of a batch.
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path


# --------------------------------------------------------------------------
# stage table
# --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Stage:
    name: str
    module: str
    #: artefacts this stage writes, checked afterwards so a silent no-op is caught
    produces: tuple[str, ...] = ()
    #: artefacts this stage reads that an EARLIER stage writes
    requires: tuple[str, ...] = ()
    #: extra argv beyond --config
    extra: tuple[str, ...] = field(default=())


#: `ingest` is resolved at runtime: --fixture swaps the offline generator in for the
#: live connectors. Everything downstream is identical either way, which is the point
#: of the shared ingest schema.
INGEST_LIVE = "src.ingest.collect"
INGEST_FIXTURE = "src.ingest.fixture"

STAGES: tuple[Stage, ...] = (
    Stage(
        name="ingest",
        module=INGEST_LIVE,
        produces=("data/raw/comments_raw.parquet", "data/processed/dev_manifest.json"),
    ),
    Stage(
        name="normalize",
        module="src.normalize.text",
        requires=("data/raw/comments_raw.parquet",),
        produces=("data/interim/comments_normalized.parquet",),
    ),
    Stage(
        name="annotate",
        module="src.annotate.run",
        requires=("data/interim/comments_normalized.parquet",),
        produces=(
            "data/interim/votes_a1.parquet",
            "data/interim/votes_a2.parquet",
            "data/interim/votes_a3.parquet",
            "models.lock.json",
        ),
    ),
    Stage(
        name="consensus",
        module="src.consensus",
        requires=(
            "data/interim/votes_a1.parquet",
            "data/interim/votes_a2.parquet",
            "data/interim/votes_a3.parquet",
        ),
        produces=("data/processed/consensus.parquet", "data/processed/ambiguous.parquet",
                  "data/processed/review_queue.parquet"),                  # D39
    ),
    Stage(
        name="mine",
        module="src.phrases.mine",
        requires=("data/processed/consensus.parquet",),
        produces=("data/processed/phrase_candidates.json",),
    ),
    Stage(
        name="families",
        module="src.phrases.families",
        requires=("data/processed/phrase_candidates.json",),
        produces=(
            "data/processed/phrase_families.json",
            "data/processed/recommendations.json",
        ),
    ),
    Stage(
        name="baseline",
        module="src.baseline",
        requires=(
            "data/processed/recommendations.json",
            "data/raw/static_profanity_id.txt",
        ),
        produces=("data/processed/baseline_comparison.json",),
    ),
)

STAGE_NAMES: tuple[str, ...] = tuple(s.name for s in STAGES)

#: artefact -> the stage that writes it, so a missing input names its producer rather
#: than surfacing a bare FileNotFoundError from three modules down.
PRODUCER: dict[str, str] = {
    path: stage.name for stage in STAGES for path in stage.produces
}
PRODUCER.setdefault("data/raw/static_profanity_id.txt", "the lexicons in data/raw (checked in, not generated)")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _describe(root: Path, rel: str) -> str:
    """One-line size summary of an artefact, for the per-stage report."""
    path = root / rel
    if not path.exists():
        return f"{rel} (MISSING)"
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq

            return f"{rel} ({pq.ParquetFile(path).metadata.num_rows:,} rows)"
        except Exception:
            return f"{rel} ({path.stat().st_size:,} bytes)"
    if path.suffix == ".json":
        try:
            import json

            payload = json.loads(path.read_text(encoding="utf-8"))
            for key in ("families", "candidates", "arms"):
                if isinstance(payload, dict) and isinstance(payload.get(key), list):
                    return f"{rel} ({len(payload[key]):,} {key})"
        except Exception:
            pass
    return f"{rel} ({path.stat().st_size:,} bytes)"


def check_requirements(stage: Stage, root: Path) -> list[str]:
    """Missing inputs, each named with the stage that produces it."""
    problems: list[str] = []
    for rel in stage.requires:
        if not (root / rel).exists():
            producer = PRODUCER.get(rel, "an earlier stage")
            problems.append(f"{rel} is missing - produced by stage '{producer}'")
    return problems


def run_stage(stage: Stage, argv: Sequence[str], root: Path) -> int:
    """Invoke a stage's `main()` in-process. Returns its exit code."""
    module = importlib.import_module(stage.module)
    main = getattr(module, "main", None)
    if main is None:
        print(f"  ERROR: {stage.module} has no main()", file=sys.stderr)
        return 1
    try:
        # `src.ingest.fixture.main` returns None; everything else returns an int.
        return int(main(list(argv)) or 0)
    except SystemExit as exc:               # argparse and hard stage errors
        return int(exc.code or 0)


def _slice_stages(from_: str | None, to: str | None) -> list[Stage]:
    start = STAGE_NAMES.index(from_) if from_ else 0
    stop = STAGE_NAMES.index(to) + 1 if to else len(STAGES)
    if start > stop - 1:
        raise SystemExit(f"--from {from_} comes after --to {to}")
    return list(STAGES[start:stop])


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def run(
    config: str = "config/local.yaml",
    *,
    fixture: bool = False,
    from_: str | None = None,
    to: str | None = None,
    root: Path = Path("."),
    backend: str | None = None,
    seed: int | None = None,
    stop_on_error: bool = True,
) -> int:
    selected = _slice_stages(from_, to)
    print(f"pipeline: {' -> '.join(s.name for s in selected)}")
    print(f"config:   {config}")
    print(f"ingest:   {'fixture (offline, seeded)' if fixture else 'live connectors'}\n")

    failed: list[str] = []
    for stage in selected:
        module = INGEST_FIXTURE if (stage.name == "ingest" and fixture) else stage.module
        stage = Stage(
            name=stage.name, module=module,
            produces=stage.produces, requires=stage.requires, extra=stage.extra,
        )

        problems = check_requirements(stage, root)
        if problems:
            print(f"[{stage.name}] SKIPPED - inputs not ready:")
            for p in problems:
                print(f"    {p}")
            failed.append(stage.name)
            if stop_on_error:
                break
            continue

        argv = ["--config", config, *stage.extra]
        if stage.name == "annotate" and backend:
            argv += ["--backend", backend]
        if stage.name == "ingest" and fixture and seed is not None:
            argv += ["--seed", str(seed)]

        print(f"[{stage.name}] {stage.module} {' '.join(argv)}")
        started = time.perf_counter()
        code = run_stage(stage, argv, root)
        elapsed = time.perf_counter() - started

        if code != 0:
            print(f"[{stage.name}] FAILED (exit {code}) after {elapsed:.1f}s\n")
            failed.append(stage.name)
            if stop_on_error:
                break
            continue

        print(f"[{stage.name}] ok in {elapsed:.1f}s")
        for rel in stage.produces:
            print(f"    -> {_describe(root, rel)}")
        print()

    if failed:
        print(f"pipeline stopped: {', '.join(failed)}")
        return 1

    _print_freeze_instructions(config)
    return 0


def _print_freeze_instructions(config: str) -> None:
    """The commands a PERSON runs. See the module docstring for why not us."""
    print("=" * 72)
    print("Development-channel stages complete.")
    print()
    print("The freeze is a deliberate act with a timing requirement: it must happen")
    print("after the development figures and BEFORE anyone looks at the held-out one.")
    print("This driver cannot know that you have, so it will not lock for you.")
    print()
    print("  python -m src.freeze lock")
    print(f"  python -m src.annotate.run --config {config} --split heldout")
    print(f"  python -m src.consensus   --config {config} --split heldout")
    print(f"  python -m src.report heldout --config {config}")
    print()
    print("The held-out figure is annotated by the FROZEN system, so its annotate and")
    print("consensus runs come AFTER the lock, not inside the pipeline above. They")
    print("write to their own files and never overwrite the development votes or")
    print("consensus - overwriting them would turn every dev-side number into a")
    print("held-out number with no frozen file changing, so nothing would detect it.")
    print()
    print("`freeze lock` refuses to run while any frozen artefact is missing;")
    print("`report heldout` verifies the lock, and refuses if the held-out split")
    print("carries no consensus labels rather than reporting zeros as a result.")
    print("=" * 72)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m src.pipeline",
        description="Run the development-channel pipeline. Stops before the freeze.",
        epilog="Stages: " + " ".join(STAGE_NAMES),
    )
    ap.add_argument("--config", default="config/local.yaml")
    ap.add_argument(
        "--fixture", action="store_true",
        help="use the seeded offline corpus instead of the live platform connectors",
    )
    ap.add_argument("--from", dest="from_", choices=STAGE_NAMES, default=None)
    ap.add_argument("--to", choices=STAGE_NAMES, default=None)
    ap.add_argument("--root", default=".", help="repository root")
    ap.add_argument(
        "--backend", default=None,
        help="annotator backend passed through to the annotate stage "
             "(llamacpp | transformers | stub)",
    )
    ap.add_argument("--seed", type=int, default=None, help="fixture seed")
    ap.add_argument(
        "--keep-going", action="store_true",
        help="continue after a failing stage instead of stopping",
    )
    args = ap.parse_args(argv)

    return run(
        args.config,
        fixture=args.fixture,
        from_=args.from_,
        to=args.to,
        root=Path(args.root),
        backend=args.backend,
        seed=args.seed,
        stop_on_error=not args.keep_going,
    )


if __name__ == "__main__":
    raise SystemExit(main())
