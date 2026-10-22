"""The consensus CLI (build contract §4): join the three vote parquets on
comment_id, call the frozen `consense()` per comment, write `consensus.parquet`
(every tier, R9 platform carried through) and `ambiguous.parquet` (R2b: excluded
from training, never deleted).
"""
from __future__ import annotations

import pandas as pd
import pytest

from src import consensus
from src.schemas import HarmSignal, Label


# --- fixture helpers ----------------------------------------------------------

def _write_harm_votes(path, annotator_id: str, rows: list[tuple[str, HarmSignal]]) -> None:
    frame = pd.DataFrame({
        "comment_id": [cid for cid, _ in rows],
        "annotator_id": [annotator_id] * len(rows),
        "signal": [sig.value for _, sig in rows],
        "raw_json": ["{}"] * len(rows),
        "confidence": [1.0] * len(rows),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _write_label_votes(path, annotator_id: str, rows: list[tuple[str, Label]]) -> None:
    frame = pd.DataFrame({
        "comment_id": [cid for cid, _ in rows],
        "annotator_id": [annotator_id] * len(rows),
        "label": [lbl.value for _, lbl in rows],
        "raw_json": ["{}"] * len(rows),
        "confidence": [1.0] * len(rows),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _write_normalized(path, rows: dict[str, tuple[str, str]]) -> None:
    """rows: comment_id -> (platform, text_normalized)."""
    frame = pd.DataFrame({
        "comment_id": list(rows),
        "platform": [v[0] for v in rows.values()],
        "text_normalized": [v[1] for v in rows.values()],
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


HARM, NO_HARM = HarmSignal.HARM, HarmSignal.NO_HARM_EVIDENCE


def _build_success_fixture(root) -> None:
    """4 comments spanning HIGH/MEDIUM/AMBIGUOUS, each on a different platform."""
    _write_harm_votes(root / consensus.VOTES_A1_PATH, "a1", [
        ("c1", HARM), ("c2", NO_HARM), ("c3", HARM), ("c4", NO_HARM),
    ])
    _write_harm_votes(root / consensus.VOTES_A2_PATH, "a2", [
        ("c1", HARM), ("c2", NO_HARM), ("c3", NO_HARM), ("c4", NO_HARM),
    ])
    _write_label_votes(root / consensus.VOTES_A3_PATH, "a3", [
        ("c1", Label.HARASSMENT_PERSONAL),
        ("c2", Label.LEGITIMATE_CRITICISM),
        ("c3", Label.HARASSMENT_PERSONAL),
        ("c4", Label.HARASSMENT_PERSONAL),
    ])
    _write_normalized(root / consensus.NORMALIZED_PATH, {
        "c1": ("youtube", "goblok banget lu"),
        "c2": ("instagram", "videonya jelek audionya pecah"),
        "c3": ("x", "goblok sih"),
        "c4": ("youtube", "dasar goblok"),
    })


# --- R3: the held-out run writes beside the development one, never over it ----

def test_split_paths_keep_the_documented_dev_paths_unchanged():
    dev = consensus.split_paths("dev")
    assert dev["consensus"] == consensus.CONSENSUS_PATH
    assert dev["ambiguous"] == consensus.AMBIGUOUS_PATH
    assert dev["a1"] == consensus.VOTES_A1_PATH


def test_split_paths_namespace_every_heldout_input_and_output():
    held = consensus.split_paths("heldout")
    assert held["consensus"] == "data/processed/consensus_heldout.parquet"
    assert held["ambiguous"] == "data/processed/ambiguous_heldout.parquet"
    assert all(p.startswith("data/interim/heldout/") for p in
               (held["a1"], held["a2"], held["a3"]))
    # nothing the held-out run writes may collide with a development artefact
    assert not set(held.values()) & set(consensus.split_paths("dev").values())


def test_an_unknown_split_is_an_operator_error():
    with pytest.raises(SystemExit):
        consensus.split_paths("prod")


def test_r3_a_heldout_run_does_not_overwrite_the_development_consensus(tmp_path):
    """Overwriting `consensus.parquet` with Channel E labels would silently turn every
    dev-side number - the phrase families, the baseline arms, the dev column of the
    transfer table - into a held-out number. No frozen file changes, so the freeze
    cannot detect it. This is why the two runs have separate paths at all.
    """
    _build_success_fixture(tmp_path)
    consensus.run({}, tmp_path, split="dev")
    dev_before = pd.read_parquet(tmp_path / consensus.CONSENSUS_PATH)

    held = consensus.split_paths("heldout")
    _write_harm_votes(tmp_path / held["a1"], "a1", [("h1", HARM)])
    _write_harm_votes(tmp_path / held["a2"], "a2", [("h1", HARM)])
    _write_label_votes(tmp_path / held["a3"], "a3", [("h1", Label.HARASSMENT_PERSONAL)])
    summary = consensus.run({}, tmp_path, split="heldout")

    dev_after = pd.read_parquet(tmp_path / consensus.CONSENSUS_PATH)
    heldout_frame = pd.read_parquet(tmp_path / held["consensus"])

    assert summary["split"] == "heldout"
    assert list(dev_after["comment_id"]) == list(dev_before["comment_id"])
    assert list(heldout_frame["comment_id"]) == ["h1"]
    assert "h1" not in set(dev_after["comment_id"])


def test_the_heldout_run_uses_the_same_frozen_consense_function(tmp_path):
    """Same code, same rules, different files - that is what makes the transfer
    comparison honest rather than two systems being compared."""
    held = consensus.split_paths("heldout")
    _write_harm_votes(tmp_path / held["a1"], "a1", [("h1", HARM)])
    _write_harm_votes(tmp_path / held["a2"], "a2", [("h1", HARM)])
    _write_label_votes(tmp_path / held["a3"], "a3", [("h1", Label.HARASSMENT_PERSONAL)])
    _write_normalized(tmp_path / consensus.NORMALIZED_PATH, {"h1": ("youtube", "goblok")})

    consensus.run({}, tmp_path, split="heldout")
    frame = pd.read_parquet(tmp_path / held["consensus"])
    # A3 asserts harm and both corroborators agree -> HIGH, exactly as on dev
    assert frame.iloc[0]["tier"] == "high"


def test_cli_exposes_the_split_flag(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(consensus, "load_config", lambda p: {})
    monkeypatch.setattr(consensus, "run",
                        lambda cfg, split="dev": seen.update(split=split))
    consensus.main(["--config", "x.yaml", "--split", "heldout"])
    assert seen["split"] == "heldout"


def test_cli_defaults_to_the_development_split(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(consensus, "load_config", lambda p: {})
    monkeypatch.setattr(consensus, "run",
                        lambda cfg, split="dev": seen.update(split=split))
    consensus.main(["--config", "x.yaml"])
    assert seen["split"] == "dev"


# --- happy path: joins, writes both outputs, carries platform through --------

def test_run_writes_consensus_with_every_tier_and_carries_platform(tmp_path):
    _build_success_fixture(tmp_path)
    summary = consensus.run({}, root=tmp_path)

    assert summary["n_consensus"] == 4
    assert summary["n_missing_a3"] == 0

    frame = pd.read_parquet(tmp_path / consensus.CONSENSUS_PATH)
    assert set(frame["comment_id"]) == {"c1", "c2", "c3", "c4"}
    assert "platform" in frame.columns

    by_id = frame.set_index("comment_id")
    assert by_id.loc["c1", "tier"] == "high"
    assert by_id.loc["c1", "label"] == "harassment_personal"
    assert by_id.loc["c1", "platform"] == "youtube"
    assert by_id.loc["c2", "platform"] == "instagram"
    assert by_id.loc["c2", "label"] == "legitimate_criticism"
    assert by_id.loc["c3", "tier"] == "medium"
    assert by_id.loc["c3", "platform"] == "x"
    # c4: A3 says harm, zero corroboration -> AMBIGUOUS, label null (R1: never
    # coerced), but the row is still WRITTEN - phrase mining needs it for n_ambiguous.
    assert by_id.loc["c4", "tier"] == "ambiguous"
    assert pd.isna(by_id.loc["c4", "label"])


def test_ambiguous_records_are_written_not_dropped(tmp_path):
    """R2b: ambiguous comments are excluded from training but never deleted."""
    _build_success_fixture(tmp_path)
    consensus.run({}, root=tmp_path)

    ambiguous = pd.read_parquet(tmp_path / consensus.AMBIGUOUS_PATH)
    assert list(ambiguous["comment_id"]) == ["c4"]
    assert ambiguous.iloc[0]["text_normalized"] == "dasar goblok"
    assert ambiguous.iloc[0]["platform"] == "youtube"
    assert pd.isna(ambiguous.iloc[0]["label"])

    # still present in consensus.parquet - "never deleted"
    consensus_frame = pd.read_parquet(tmp_path / consensus.CONSENSUS_PATH)
    assert "c4" in set(consensus_frame["comment_id"])


# --- missing A3: skipped with a counted warning ------------------------------

def test_missing_a3_vote_is_skipped_with_a_counted_warning(tmp_path, capsys):
    _write_harm_votes(tmp_path / consensus.VOTES_A1_PATH, "a1", [("skip1", HARM)])
    _write_harm_votes(tmp_path / consensus.VOTES_A2_PATH, "a2", [("skip1", HARM)])
    _write_label_votes(tmp_path / consensus.VOTES_A3_PATH, "a3", [])
    _write_normalized(tmp_path / consensus.NORMALIZED_PATH, {"skip1": ("youtube", "x")})

    summary = consensus.run({}, root=tmp_path)

    assert summary["n_missing_a3"] == 1
    assert summary["n_consensus"] == 0
    frame = pd.read_parquet(tmp_path / consensus.CONSENSUS_PATH)
    assert "skip1" not in set(frame["comment_id"])
    assert "WARNING" in capsys.readouterr().out


# --- missing A1 or A2: an error, not a skip -----------------------------------

def test_a_comment_with_a3_but_missing_a2_is_a_hard_error(tmp_path):
    _write_harm_votes(tmp_path / consensus.VOTES_A1_PATH, "a1", [("err1", HARM)])
    _write_harm_votes(tmp_path / consensus.VOTES_A2_PATH, "a2", [])   # missing
    _write_label_votes(tmp_path / consensus.VOTES_A3_PATH, "a3", [
        ("err1", Label.HARASSMENT_PERSONAL),
    ])
    _write_normalized(tmp_path / consensus.NORMALIZED_PATH, {"err1": ("youtube", "x")})

    with pytest.raises(SystemExit):
        consensus.run({}, root=tmp_path)


def test_a_comment_with_a3_but_missing_a1_is_a_hard_error(tmp_path):
    _write_harm_votes(tmp_path / consensus.VOTES_A1_PATH, "a1", [])   # missing
    _write_harm_votes(tmp_path / consensus.VOTES_A2_PATH, "a2", [("err2", HARM)])
    _write_label_votes(tmp_path / consensus.VOTES_A3_PATH, "a3", [
        ("err2", Label.HARASSMENT_PERSONAL),
    ])
    _write_normalized(tmp_path / consensus.NORMALIZED_PATH, {"err2": ("youtube", "x")})

    with pytest.raises(SystemExit):
        consensus.run({}, root=tmp_path)


# --- graceful degradation when platform hasn't landed yet --------------------

def test_missing_platform_column_degrades_to_unknown_with_a_warning(tmp_path, capsys):
    _write_harm_votes(tmp_path / consensus.VOTES_A1_PATH, "a1", [("c1", HARM)])
    _write_harm_votes(tmp_path / consensus.VOTES_A2_PATH, "a2", [("c1", HARM)])
    _write_label_votes(tmp_path / consensus.VOTES_A3_PATH, "a3", [
        ("c1", Label.HARASSMENT_PERSONAL),
    ])
    # normalized parquet with NO platform column - e.g. normalize lane not yet R9-ready
    frame = pd.DataFrame({"comment_id": ["c1"], "text_normalized": ["goblok"]})
    (tmp_path / consensus.NORMALIZED_PATH).parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(tmp_path / consensus.NORMALIZED_PATH, index=False)

    consensus.run({}, root=tmp_path)
    captured = pd.read_parquet(tmp_path / consensus.CONSENSUS_PATH)
    assert captured.iloc[0]["platform"] == "unknown"
    assert "WARNING" in capsys.readouterr().out


# --- missing input files are loud operator errors -----------------------------

def test_missing_normalized_parquet_raises(tmp_path):
    _write_harm_votes(tmp_path / consensus.VOTES_A1_PATH, "a1", [("c1", HARM)])
    _write_harm_votes(tmp_path / consensus.VOTES_A2_PATH, "a2", [("c1", HARM)])
    _write_label_votes(tmp_path / consensus.VOTES_A3_PATH, "a3", [
        ("c1", Label.HARASSMENT_PERSONAL),
    ])
    with pytest.raises(SystemExit):
        consensus.run({}, root=tmp_path)


def test_missing_vote_parquet_raises(tmp_path):
    with pytest.raises(SystemExit):
        consensus.run({}, root=tmp_path)


# --- CLI entry point: `python -m src.consensus --config ...` -----------------

def test_main_reads_config_and_runs_end_to_end(tmp_path, monkeypatch, capsys):
    _build_success_fixture(tmp_path)
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "local.yaml").write_text("phrases: {}\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    exit_code = consensus.main(["--config", "config/local.yaml"])

    assert exit_code == 0
    assert (tmp_path / consensus.CONSENSUS_PATH).exists()
    assert (tmp_path / consensus.AMBIGUOUS_PATH).exists()


def test_main_missing_config_is_a_loud_operator_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        consensus.main(["--config", "config/does_not_exist.yaml"])
