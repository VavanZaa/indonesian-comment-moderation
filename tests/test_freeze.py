"""AGENTS.md R3 - the freeze must be detectable, not remembered."""
import json

from src.freeze import FROZEN_SURFACE, compute_manifest, verify


def test_frozen_surface_covers_everything_that_changes_behaviour():
    for entry in ("config/local.yaml", "prompts", "src/normalize",
                  "src/annotate", "src/consensus.py", "src/phrases"):
        assert entry in FROZEN_SURFACE


def test_manifest_detects_a_change(tmp_path):
    (tmp_path / "src").mkdir()
    target = tmp_path / "src" / "consensus.py"
    target.write_text("x = 1\n")
    before = compute_manifest(tmp_path)
    target.write_text("x = 2\n")
    after = compute_manifest(tmp_path)
    assert before["src/consensus.py"] != after["src/consensus.py"]


def test_verify_reports_drift(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "consensus.py").write_text("x = 1\n")
    (tmp_path / "freeze.lock.json").write_text(
        json.dumps(compute_manifest(tmp_path), indent=2, sort_keys=True))
    assert verify(tmp_path) == []
    (tmp_path / "src" / "consensus.py").write_text("x = 2\n")
    assert "src/consensus.py" in verify(tmp_path)


# --- regressions for the hardened freeze (D34) -------------------------------

import pytest

from src.freeze import MISSING, lock


def test_lock_refuses_missing_required_artifacts(tmp_path):
    """A lockfile full of MISSING looks like a real experimental freeze and is not.

    Without this, you could lock before `phrase_families.json` existed, generate it
    after glancing at Channel E, and `verify` would still report the surface intact.
    """
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "consensus.py").write_text("x = 1\n")
    with pytest.raises(RuntimeError) as exc:
        lock(tmp_path)
    assert "refusing to lock" in str(exc.value)
    assert "data/processed/phrase_families.json" in str(exc.value)
    assert not (tmp_path / "freeze.lock.json").exists(), "no lockfile may be written"


def test_manifest_marks_absent_entries_missing(tmp_path):
    manifest = compute_manifest(tmp_path)
    assert manifest["models.lock.json"] == MISSING
    assert manifest["data/raw/lexicons"] == MISSING, "an empty frozen dir is not frozen"


def test_non_text_resources_are_hashed(tmp_path):
    """An extension allowlist would silently skip a .csv or .parquet lexicon while the
    directory still looked frozen."""
    lex = tmp_path / "data" / "raw" / "lexicons"
    lex.mkdir(parents=True)
    (lex / "alay.csv").write_text("slang,formal\nbgt,banget\n")
    (lex / "rules.toml").write_text("family = 'insult'\n")
    (lex / "dict.parquet").write_bytes(b"\x00PAR1")

    before = compute_manifest(tmp_path)
    for name in ("alay.csv", "rules.toml", "dict.parquet"):
        assert f"data/raw/lexicons/{name}" in before, f"{name} must be hashed"

    (lex / "alay.csv").write_text("slang,formal\nbgt,banget\ngk,tidak\n")
    after = compute_manifest(tmp_path)
    assert before["data/raw/lexicons/alay.csv"] != after["data/raw/lexicons/alay.csv"]


def test_pycache_is_ignored(tmp_path):
    src = tmp_path / "src" / "phrases"
    src.mkdir(parents=True)
    (src / "mine.py").write_text("x = 1\n")
    (src / "__pycache__").mkdir()
    (src / "__pycache__" / "mine.cpython-311.pyc").write_bytes(b"\x00")
    manifest = compute_manifest(tmp_path)
    assert "src/phrases/mine.py" in manifest
    assert not any("__pycache__" in k for k in manifest)
