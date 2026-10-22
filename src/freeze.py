"""FREEZE (AGENTS.md R3), made mechanical instead of a promise.

The held-out channel must meet a system nobody touched after seeing it. That is easy
to violate by accident: you glance at Channel E, notice the normalizer mangles a slang
form, fix it, and the channel is no longer unseen. This module hashes the frozen
surface so the violation is detected rather than remembered.

    python -m src.freeze lock     # after dev channels, before touching E
    python -m src.freeze verify   # run this immediately before scoring E

`freeze.lock.json` is generated, not shipped. The repository template must not contain
one: a lockfile full of MISSING entries looks like a real experimental freeze and is not.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

#: Everything whose behaviour must not change between the lock and the held-out run.
#:
#: Code alone is not enough. `phrase_families.json` can change while every .py file
#: stays byte-identical, and a swapped alay dictionary silently changes normalization.
#: The conceptual freeze is:
#:     code + config + external resources + model identity + learned artifacts + dev-data identity
FROZEN_SURFACE: tuple[str, ...] = (
    # code and config
    "config/local.yaml",
    "prompts",                          # A3's prompt - the easiest thing to tweak silently
    "src/normalize",
    "src/annotate",
    "src/consensus.py",
    "src/phrases",
    "src/schemas.py",
    # external resources that determine behaviour without being code
    "data/raw/static_profanity_id.txt", # the baseline arm
    "data/raw/lexicons",                # kamus-alay, abusive lexicon, rule families
    # model identity - written by the annotate step: {model_id, revision, quantization}
    "models.lock.json",
    # learned artifacts - these change while the code does not
    "data/processed/phrase_candidates.json",
    "data/processed/phrase_families.json",
    "data/processed/recommendations.json",
    # dev-data identity - which comments the system learned from
    "data/processed/dev_manifest.json",  # channel ids + comment id hashes for A-D
)

LOCK_PATH = Path("freeze.lock.json")


#: Never hashed. Everything else in a frozen directory is, regardless of extension -
#: a lexicon can be .csv, .tsv, .toml or .parquet, and an extension allowlist would
#: silently skip it while the directory looked frozen.
IGNORED_NAMES: frozenset[str] = frozenset({".DS_Store", "Thumbs.db"})
IGNORED_DIRS: frozenset[str] = frozenset({"__pycache__", ".git", ".ipynb_checkpoints"})

MISSING = "MISSING"


def _hash_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def _should_hash(p: Path) -> bool:
    if not p.is_file() or p.name in IGNORED_NAMES:
        return False
    return not any(part in IGNORED_DIRS for part in p.parts)


def compute_manifest(root: Path = Path(".")) -> dict[str, str]:
    """Hash every regular file in the frozen surface. No extension allowlist."""
    out: dict[str, str] = {}
    for entry in FROZEN_SURFACE:
        target = root / entry
        if target.is_file():
            out[entry] = _hash_file(target)
        elif target.is_dir():
            found = False
            for f in sorted(target.rglob("*")):
                if _should_hash(f):
                    # POSIX separators always: a lockfile written on Windows must
                    # verify on Linux, and `FROZEN_SURFACE` entries are posix-style.
                    out[f.relative_to(root).as_posix()] = _hash_file(f)
                    found = True
            if not found:
                out[entry] = MISSING          # an empty frozen directory is not frozen
        else:
            out[entry] = MISSING
    return out


def lock(root: Path = Path(".")) -> None:
    """Write the lockfile. Refuses to lock an incomplete surface.

    FROZEN_SURFACE declares what MUST exist at lock time. A lockfile containing
    MISSING entries looks like a real experimental freeze and is not: you could lock
    before `phrase_families.json` or `models.lock.json` existed, generate them after
    glancing at Channel E, and `verify` would report the surface intact.

    If an entry genuinely does not apply to your run, remove it from FROZEN_SURFACE
    deliberately rather than locking around it.
    """
    manifest = compute_manifest(root)
    missing = sorted(k for k, v in manifest.items() if v == MISSING)
    if missing:
        raise RuntimeError(
            "refusing to lock - these frozen artefacts do not exist yet:\n  "
            + "\n  ".join(missing)
            + "\n\nRun the development-channel pipeline first. If an entry does not "
              "apply, remove it from FROZEN_SURFACE deliberately."
        )
    (root / LOCK_PATH).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"frozen {len(manifest)} files -> {LOCK_PATH}")
    print("Do not modify anything above before the held-out run.")


def verify(root: Path = Path(".")) -> list[str]:
    """Return the list of drifted paths. Empty means the freeze held."""
    lock_file = root / LOCK_PATH
    if not lock_file.exists():
        raise SystemExit(f"{LOCK_PATH} not found - run `python -m src.freeze lock` first")
    locked = json.loads(lock_file.read_text())
    current = compute_manifest(root)
    drifted = sorted(
        set(locked) ^ set(current)
        | {k for k in set(locked) & set(current) if locked[k] != current[k]}
    )
    return drifted


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if cmd == "lock":
        try:
            lock()
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        return
    drifted = verify()
    if drifted:
        print("FREEZE VIOLATED - these changed after the lock:")
        for d in drifted:
            print(f"  {d}")
        print("\nChannel E is no longer unseen. Either revert, or re-lock and say so on camera.")
        raise SystemExit(1)
    print("freeze intact - safe to score the held-out channel")


if __name__ == "__main__":
    main()
