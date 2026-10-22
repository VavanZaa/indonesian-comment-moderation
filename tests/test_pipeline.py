"""The pipeline driver (build contract §9).

The load-bearing property is a NEGATIVE one: the driver must not freeze. `freeze lock`
is a claim about timing - that the system which scored the held-out figure is the system
that existed before anyone looked at it - and only a person can know that is true. A
driver that locked automatically would emit a lockfile on the run you did *after*
noticing the normalizer mangles a slang form in the held-out data, which is precisely
the failure R3 exists to detect.
"""
from __future__ import annotations

import pytest

from src import pipeline
from src.pipeline import PRODUCER, STAGE_NAMES, STAGES, Stage, check_requirements


# ---------------------------------------------------------------------------
# R3 / D19 - the driver stops before the freeze
# ---------------------------------------------------------------------------

def test_r3_no_stage_locks_or_verifies_the_freeze():
    """If `freeze` were in the stage table, every run would re-lock and the freeze
    would certify nothing."""
    modules = [s.module for s in STAGES]
    assert not any("freeze" in m for m in modules)
    assert not any("report" in m for m in modules), (
        "the held-out run is not a batch stage - it follows a deliberate lock")


def test_r3_the_driver_prints_the_two_manual_commands_and_does_not_run_them(capsys):
    pipeline._print_freeze_instructions("config/local.yaml")
    out = capsys.readouterr().out
    assert "python -m src.freeze lock" in out
    assert "python -m src.report heldout --config config/local.yaml" in out
    assert "will not lock for you" in out


def test_r3_the_reason_for_not_automating_is_written_down_in_the_source():
    """A future reader will be tempted to 'improve' this by adding one line. The
    justification has to survive in the file, not only in a commit message."""
    import pathlib

    source = (pathlib.Path(pipeline.__file__)).read_text(encoding="utf-8")
    assert "timing" in source.lower()
    assert "Do not \"improve\" it" in source


def test_r3_a_successful_run_ends_with_the_freeze_instructions(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(pipeline, "run_stage", lambda stage, argv, root: 0)
    monkeypatch.setattr(pipeline, "check_requirements", lambda stage, root: [])
    code = pipeline.run("config/local.yaml", root=tmp_path)
    assert code == 0
    assert "python -m src.freeze lock" in capsys.readouterr().out


def test_a_failed_run_does_not_print_freeze_instructions(monkeypatch, capsys, tmp_path):
    """Telling someone to lock a half-built system is worse than saying nothing."""
    monkeypatch.setattr(pipeline, "run_stage", lambda stage, argv, root: 1)
    monkeypatch.setattr(pipeline, "check_requirements", lambda stage, root: [])
    code = pipeline.run("config/local.yaml", root=tmp_path)
    assert code == 1
    assert "python -m src.freeze lock" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# the stage table
# ---------------------------------------------------------------------------

def test_stage_order_matches_the_documented_pipeline():
    assert STAGE_NAMES == (
        "ingest", "normalize", "annotate", "consensus", "mine", "families", "baseline",
    )


def test_every_required_artefact_has_a_named_producer():
    """A missing input must name the stage that makes it, not surface a bare
    FileNotFoundError from three modules down."""
    for stage in STAGES:
        for rel in stage.requires:
            assert rel in PRODUCER, f"{stage.name} requires {rel} with no known producer"


def test_requirements_are_produced_by_an_earlier_stage_never_a_later_one():
    """A cycle here would mean the order is wrong and the run could never succeed."""
    position = {name: i for i, name in enumerate(STAGE_NAMES)}
    for stage in STAGES:
        for rel in stage.requires:
            producer = PRODUCER[rel]
            if producer in position:
                assert position[producer] < position[stage.name], (
                    f"{stage.name} requires {rel} from the later stage {producer}")


def test_baseline_runs_after_families_because_it_reads_the_learned_filters():
    assert STAGE_NAMES.index("baseline") > STAGE_NAMES.index("families")


# ---------------------------------------------------------------------------
# missing inputs
# ---------------------------------------------------------------------------

def test_a_missing_input_names_the_stage_that_produces_it(tmp_path):
    stage = next(s for s in STAGES if s.name == "consensus")
    problems = check_requirements(stage, tmp_path)
    assert problems
    assert all("produced by stage 'annotate'" in p for p in problems)


def test_a_present_input_is_not_reported_missing(tmp_path):
    stage = next(s for s in STAGES if s.name == "normalize")
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "raw" / "comments_raw.parquet").write_bytes(b"\x00")
    assert check_requirements(stage, tmp_path) == []


def test_the_run_skips_rather_than_crashes_when_inputs_are_missing(tmp_path, capsys):
    """`--from mine` on an empty tree should explain itself, not raise."""
    code = pipeline.run("config/local.yaml", from_="mine", root=tmp_path)
    out = capsys.readouterr().out
    assert code == 1
    assert "inputs not ready" in out
    assert "produced by stage 'families'" in out or "produced by stage 'consensus'" in out


def test_the_static_list_is_flagged_as_checked_in_not_generated(tmp_path):
    """Telling someone to 'run the stage that produces it' would be wrong here - the
    creator's blocked-word list is authored, not computed."""
    assert "not generated" in PRODUCER["data/raw/static_profanity_id.txt"]


# ---------------------------------------------------------------------------
# stage selection
# ---------------------------------------------------------------------------

def test_from_and_to_select_an_inclusive_range():
    selected = pipeline._slice_stages("normalize", "consensus")
    assert [s.name for s in selected] == ["normalize", "annotate", "consensus"]


def test_to_alone_runs_from_the_beginning():
    assert [s.name for s in pipeline._slice_stages(None, "normalize")] == \
        ["ingest", "normalize"]


def test_from_alone_runs_to_the_end():
    assert [s.name for s in pipeline._slice_stages("families", None)] == \
        ["families", "baseline"]


def test_an_inverted_range_is_rejected():
    with pytest.raises(SystemExit):
        pipeline._slice_stages("baseline", "ingest")


# ---------------------------------------------------------------------------
# fixture mode
# ---------------------------------------------------------------------------

def test_fixture_mode_swaps_the_ingest_module_and_nothing_else(monkeypatch, tmp_path):
    """The whole point of the shared ingest schema: downstream stages cannot tell
    whether the corpus came from the live connectors or the offline generator."""
    seen: list[str] = []
    monkeypatch.setattr(pipeline, "run_stage",
                        lambda stage, argv, root: seen.append(stage.module) or 0)
    monkeypatch.setattr(pipeline, "check_requirements", lambda stage, root: [])

    pipeline.run("config/local.yaml", fixture=True, to="normalize", root=tmp_path)
    assert seen[0] == pipeline.INGEST_FIXTURE

    seen.clear()
    pipeline.run("config/local.yaml", fixture=False, to="normalize", root=tmp_path)
    assert seen[0] == pipeline.INGEST_LIVE


def test_live_mode_uses_the_multi_platform_collector():
    """R9/D35 - `collect` fans out to YouTube, Instagram and X."""
    assert pipeline.INGEST_LIVE == "src.ingest.collect"


def test_the_backend_flag_reaches_only_the_annotate_stage(monkeypatch, tmp_path):
    calls: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(pipeline, "run_stage",
                        lambda stage, argv, root: calls.append((stage.name, list(argv))) or 0)
    monkeypatch.setattr(pipeline, "check_requirements", lambda stage, root: [])

    pipeline.run("config/local.yaml", backend="stub", root=tmp_path)
    for name, argv in calls:
        if name == "annotate":
            assert "--backend" in argv and "stub" in argv
        else:
            assert "--backend" not in argv


def test_every_stage_receives_the_config_path(monkeypatch, tmp_path):
    calls: list[list[str]] = []
    monkeypatch.setattr(pipeline, "run_stage",
                        lambda stage, argv, root: calls.append(list(argv)) or 0)
    monkeypatch.setattr(pipeline, "check_requirements", lambda stage, root: [])

    pipeline.run("config/custom.yaml", root=tmp_path)
    assert calls and all(a[:2] == ["--config", "config/custom.yaml"] for a in calls)


# ---------------------------------------------------------------------------
# failure handling
# ---------------------------------------------------------------------------

def test_a_failing_stage_stops_the_run_by_default(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(pipeline, "check_requirements", lambda stage, root: [])
    monkeypatch.setattr(pipeline, "run_stage",
                        lambda stage, argv, root: 1 if stage.name == "normalize" else 0)
    assert pipeline.run("config/local.yaml", root=tmp_path) == 1
    out = capsys.readouterr().out
    assert "[normalize] FAILED" in out
    assert "[annotate]" not in out, "downstream stages must not run on stale inputs"


def test_keep_going_continues_past_a_failure(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(pipeline, "check_requirements", lambda stage, root: [])
    monkeypatch.setattr(pipeline, "run_stage",
                        lambda stage, argv, root: 1 if stage.name == "normalize" else 0)
    assert pipeline.run("config/local.yaml", root=tmp_path, stop_on_error=False) == 1
    assert "[annotate]" in capsys.readouterr().out


def test_a_stage_calling_sys_exit_is_reported_not_propagated(tmp_path):
    """Several stages raise SystemExit on a bad config. The driver must turn that into
    a stage failure it can report, not let it kill the run mid-table."""
    class FakeModule:
        @staticmethod
        def main(argv):
            raise SystemExit(3)

    import sys

    sys.modules["src._fake_stage"] = FakeModule            # type: ignore[assignment]
    try:
        stage = Stage(name="fake", module="src._fake_stage")
        assert pipeline.run_stage(stage, [], tmp_path) == 3
    finally:
        del sys.modules["src._fake_stage"]


def test_a_stage_returning_none_is_treated_as_success(tmp_path):
    """`src.ingest.fixture.main` returns None; everything else returns an int."""
    class FakeModule:
        @staticmethod
        def main(argv):
            return None

    import sys

    sys.modules["src._fake_none"] = FakeModule             # type: ignore[assignment]
    try:
        assert pipeline.run_stage(Stage("fake", "src._fake_none"), [], tmp_path) == 0
    finally:
        del sys.modules["src._fake_none"]


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------

def test_cli_accepts_the_documented_flags(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(pipeline, "run", lambda config, **kw: captured.update(
        config=config, **kw) or 0)
    code = pipeline.main([
        "--config", "config/local.yaml", "--fixture",
        "--from", "normalize", "--to", "families", "--seed", "7",
    ])
    assert code == 0
    assert captured["fixture"] is True
    assert captured["from_"] == "normalize"
    assert captured["to"] == "families"
    assert captured["seed"] == 7


def test_cli_rejects_an_unknown_stage_name():
    with pytest.raises(SystemExit):
        pipeline.main(["--from", "not_a_stage"])
