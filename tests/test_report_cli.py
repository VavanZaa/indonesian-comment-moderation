"""The reporting CLI: the held-out run (R3) and the screen check (AGENTS.md §5).

Two guarantees are load-bearing here and neither is a matter of taste:

- `heldout` must REFUSE to run against a drifted freeze. A held-out figure scored by a
  system that was touched after seeing it is not a held-out result, and the whole
  strongest-claim-in-the-project rests on it.
- Nothing that reaches a display artefact may name a real person or handle (R8, R9).
  On screen a figure is `Figure A`.
"""
from __future__ import annotations

import json

import pytest

from src import freeze, report
from src.report import (
    ATTRIBUTION,
    _figure_label,
    _mirror_sample,
    build_provenance,
    collection_mix,
    run_heldout,
    screen_check,
)


# ---------------------------------------------------------------------------
# a minimal but complete repo on disk
# ---------------------------------------------------------------------------

ACCOUNT = "UC_realchannel_handle"          # must never appear in an output artefact
FIGURE = "figure_seleb_asli"               # ditto


#: Both dev families clear `n_all >= 25` on purpose. Below minimum support every
#: family is REJECTED regardless of its rate, so a smaller fixture would test the
#: support floor rather than the contrast this file is about.
N_PER_FAMILY = 30

#: Appears ONLY in the held-out split, so `--phrase` cannot resolve it against a known
#: dev family and must fall through to the ad hoc dev+heldout scan (D27).
HELDOUT_ONLY_PHRASE = "mending mati"


def _corpus_rows():
    rows = []
    for i in range(N_PER_FAMILY * 2):
        harmful = i < N_PER_FAMILY
        text = "dasar goblok" if harmful else "videonya jelek"
        rows.append({
            "comment_id": f"d{i}",
            "platform": ["youtube", "instagram", "x"][i % 3],
            "figure_id": FIGURE,
            "source_account": ACCOUNT,
            "post_id": f"post{i % 2}",
            "post_url": f"https://example.test/{i}",
            "author_hash": f"hash{i:04d}",
            "published_at": "2026-04-01T00:00:00Z",
            "text_original": text,
            "text_normalized": text,
            "split": "dev",
            "genre": "entertainment",
            "collection_mode": "api" if i % 3 == 0 else "import",
        })
    for i in range(5):
        rows.append({
            "comment_id": f"h{i}",
            "platform": "youtube",
            "figure_id": "figure_heldout",
            "source_account": "UC_heldout_account",
            "post_id": "postE",
            "post_url": "https://example.test/e",
            "author_hash": f"hh{i:04d}",
            "published_at": "2026-05-01T00:00:00Z",
            "text_original": f"{HELDOUT_ONLY_PHRASE} aja",
            "text_normalized": f"{HELDOUT_ONLY_PHRASE} aja",
            "split": "heldout",
            "genre": "entertainment",
            "collection_mode": "api",
        })
    return rows


def _consensus_rows():
    out = []
    for i in range(N_PER_FAMILY * 2):
        label = "harassment_personal" if i < N_PER_FAMILY else "legitimate_criticism"
        out.append({"comment_id": f"d{i}", "label": label, "tier": "high",
                    "weight": 1.0, "platform": ["youtube", "instagram", "x"][i % 3]})
    for i in range(5):
        out.append({"comment_id": f"h{i}", "label": "harassment_personal",
                    "tier": "high", "weight": 1.0, "platform": "youtube"})
    return out


@pytest.fixture
def repo(tmp_path):
    """A repo root with every artefact the report CLI reads."""
    import pandas as pd

    (tmp_path / "data" / "interim").mkdir(parents=True)
    (tmp_path / "data" / "processed").mkdir(parents=True)

    pd.DataFrame(_corpus_rows()).to_parquet(
        tmp_path / "data" / "interim" / "comments_normalized.parquet", index=False)
    pd.DataFrame(_consensus_rows()).to_parquet(
        tmp_path / "data" / "processed" / "consensus.parquet", index=False)

    harmful_ids = [f"d{i}" for i in range(N_PER_FAMILY)]
    criticism_ids = [f"d{i}" for i in range(N_PER_FAMILY, N_PER_FAMILY * 2)]
    families = {"families": [
        {"name": "GOBLOK", "canonical_key": "goblok", "variants": ["goblok"],
         "member_ids": harmful_ids, "inflation_factor": 1.0},
        {"name": "JELEK", "canonical_key": "jelek", "variants": ["jelek"],
         "member_ids": criticism_ids, "inflation_factor": 1.0},
    ]}
    recommendations = {
        "decision_rule": "L_H >= 0.85 AND C <= 0.10 AND n_all >= 25",
        "thresholds": {"recommend_threshold": 0.85, "max_criticism_share": 0.10,
                       "min_matches": 25},
        "families": [
            {"name": "GOBLOK", "variants": ["goblok"], "n_all": len(harmful_ids),
             "verdict": "SUGGESTED"},
            {"name": "JELEK", "variants": ["jelek"], "n_all": len(criticism_ids),
             "verdict": "REJECTED"},
        ],
    }
    for name, payload in (("phrase_families.json", families),
                          ("recommendations.json", recommendations)):
        (tmp_path / "data" / "processed" / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# D37 - screen_check's printed verdict must follow the CONFIGURED threshold, not
# PhraseScore.recommend()'s hardcoded 0.85 default (the frozen screen_check()
# function's own bug; run_screen_check must correct for it, not inherit it)
# ---------------------------------------------------------------------------

def _boundary_repo(tmp_path):
    """A single family at 23/30 harmful (p_harmful_lower ~= 0.767) - between the old
    0.85 default and D37's 0.60, so its verdict must be able to flip depending on
    which threshold is actually configured."""
    import pandas as pd

    (tmp_path / "data" / "interim").mkdir(parents=True)
    (tmp_path / "data" / "processed").mkdir(parents=True)

    ids = [f"b{i}" for i in range(30)]
    rows = [{
        "comment_id": cid, "platform": "youtube", "figure_id": FIGURE,
        "source_account": ACCOUNT, "post_id": "post0", "post_url": "https://example.test/0",
        "author_hash": f"hash{i:04d}", "published_at": "2026-04-01T00:00:00Z",
        "text_original": "dasar tolol", "text_normalized": "dasar tolol",
        "split": "dev", "genre": "entertainment", "collection_mode": "api",
    } for i, cid in enumerate(ids)]
    pd.DataFrame(rows).to_parquet(
        tmp_path / "data" / "interim" / "comments_normalized.parquet", index=False)

    consensus_rows = [
        {"comment_id": cid, "label": "harassment_personal" if i < 23 else "benign_other",
         "tier": "high", "weight": 1.0, "platform": "youtube"}
        for i, cid in enumerate(ids)
    ]
    pd.DataFrame(consensus_rows).to_parquet(
        tmp_path / "data" / "processed" / "consensus.parquet", index=False)

    families = {"families": [
        {"name": "TOLOL", "canonical_key": "tolol", "variants": ["tolol"],
         "member_ids": ids, "inflation_factor": 1.0},
    ]}
    recommendations = {
        "decision_rule": "L_H >= 0.6 AND C <= 0.1 AND n_all >= 25",
        "thresholds": {"recommend_threshold": 0.60, "max_criticism_share": 0.10, "min_matches": 25},
        "families": [{"name": "TOLOL", "variants": ["tolol"], "n_all": 30, "verdict": "SUGGESTED"}],
    }
    for name, payload in (("phrase_families.json", families),
                          ("recommendations.json", recommendations)):
        (tmp_path / "data" / "processed" / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return tmp_path


def test_screen_check_verdict_follows_the_configured_threshold_not_085(tmp_path):
    """Regression (D37): the printed/written verdict used to always come from
    ScreenCheckItem.verdict, which the frozen screen_check() computes via
    score.recommend() with NO arguments - always 0.85, regardless of config. A
    boundary family (~0.767 lower bound) proves run_screen_check now applies the
    real configured threshold instead."""
    repo = _boundary_repo(tmp_path)

    strict = report.run_screen_check({"phrases": {"recommend_threshold": 0.85}}, repo, top=1, samples=5)
    lenient = report.run_screen_check({"phrases": {"recommend_threshold": 0.60}}, repo, top=1, samples=5)

    assert strict["families"][0]["verdict"] == "REJECTED"
    assert lenient["families"][0]["verdict"] == "SUGGESTED"


def test_screen_check_verdict_uses_060_when_config_omits_the_threshold(tmp_path):
    """An operator whose config doesn't set phrases.recommend_threshold at all must
    get the CURRENT project default (0.60, D37) here too, not the superseded 0.85 -
    this is the same fallback discipline as src/phrases/grouping.py::run()."""
    repo = _boundary_repo(tmp_path)
    payload = report.run_screen_check({}, repo, top=1, samples=5)
    assert payload["families"][0]["verdict"] == "SUGGESTED"


# ---------------------------------------------------------------------------
# R3 - the freeze gate on the held-out run
# ---------------------------------------------------------------------------

def test_r3_heldout_refuses_to_run_when_the_freeze_has_drifted(repo, monkeypatch, capsys):
    """This is the whole point of the freeze. A drifted surface means the held-out
    figure is no longer unseen, and scoring it anyway produces a number that looks like
    generalisation and is not."""
    monkeypatch.setattr(freeze, "verify", lambda root=None: ["src/normalize/text.py"])
    called = []
    monkeypatch.setattr(report, "heldout_report",
                        lambda *a, **k: called.append(1) or {})

    code = run_heldout({}, repo)

    assert code == 1, "a drifted freeze must exit non-zero"
    assert not called, "nothing may be scored after drift is detected"
    err = capsys.readouterr().err
    assert "FREEZE VIOLATED" in err
    assert "src/normalize/text.py" in err


def test_r3_the_freeze_check_runs_before_anything_else(repo, monkeypatch):
    """Order matters: verify first, score second. If scoring ran first the refusal
    would be cosmetic."""
    order: list[str] = []
    monkeypatch.setattr(freeze, "verify",
                        lambda root=None: order.append("verify") or ["drifted"])
    monkeypatch.setattr(report, "heldout_report",
                        lambda *a, **k: order.append("score") or {})
    run_heldout({}, repo)
    assert order == ["verify"], "scoring must not be reached at all"


def test_r3_missing_lockfile_stops_the_run(repo, monkeypatch):
    """`freeze.verify` raises when no lock exists. The held-out run must not proceed
    on an unfrozen system."""
    monkeypatch.setattr(freeze, "verify", lambda root=None: (_ for _ in ()).throw(
        SystemExit("freeze.lock.json not found")))
    with pytest.raises(SystemExit):
        run_heldout({}, repo)


def test_r3_a_clean_freeze_lets_the_run_proceed(repo, monkeypatch):
    monkeypatch.setattr(freeze, "verify", lambda root=None: [])
    monkeypatch.setattr(report, "heldout_report", lambda *a, **k: {
        "n_heldout_comments": 3, "n_families": 1, "n_verdict_matches": 1,
        "n_verdict_flips": 0, "families": [],
    })
    assert run_heldout({}, repo) == 0
    assert (repo / "data" / "processed" / "heldout_report.json").exists()


# ---------------------------------------------------------------------------
# missing held-out labels must REFUSE, never report zeros as a result
# ---------------------------------------------------------------------------

def _write_heldout_consensus(repo, ids):
    import pandas as pd

    rows = [{"comment_id": cid, "label": "harassment_personal", "tier": "high",
             "weight": 1.0, "platform": "youtube"} for cid in ids]
    pd.DataFrame(rows).to_parquet(
        repo / "data" / "processed" / "consensus_heldout.parquet", index=False)


def test_an_unannotated_heldout_split_is_refused_not_reported_as_zero(repo):
    """THE failure this guard exists for.

    Unlabelled matches are silently skipped when a family is scored, so an
    unannotated held-out split makes every family score n_all=0, every lower bound
    0.0, and every SUGGESTED family "flip" to REJECTED. The report would then state,
    in the shape of a real result, that nothing transferred - the exact opposite of
    demo scene 4, from missing data rather than evidence. Nothing upstream catches it:
    the freeze is intact, the corpus is present, the arithmetic is right.
    """
    with pytest.raises(SystemExit) as exc:
        report.heldout_report({}, repo)
    message = str(exc.value)
    assert "has not been annotated" in message
    # and it must say exactly how to fix it, in the right order
    assert "--split heldout" in message
    assert "src.annotate.run" in message and "src.consensus" in message


def test_the_refusal_explains_that_the_heldout_run_follows_the_lock(repo):
    """R3 - the held-out figure is scored by the FROZEN system, so its annotate and
    consensus runs come after `freeze lock`, not inside the development pipeline."""
    with pytest.raises(SystemExit) as exc:
        report.heldout_report({}, repo)
    assert "freeze verify" in str(exc.value)


def test_partial_heldout_labels_are_also_refused(repo):
    """A handful of gaps is normal - `src.consensus` legitimately skips a comment whose
    A3 vote was dropped off-schema. Most of the split missing is not."""
    _write_heldout_consensus(repo, ["h0"])          # 1 of 5
    with pytest.raises(SystemExit) as exc:
        report.heldout_report({}, repo)
    assert "labels only 1 of 5" in str(exc.value)


def test_a_fully_labelled_heldout_split_produces_a_report(repo):
    _write_heldout_consensus(repo, [f"h{i}" for i in range(5)])
    payload = report.heldout_report({}, repo)
    assert payload["n_heldout_comments"] == 5
    assert payload["n_families"] == 2


def test_a_flip_on_thin_heldout_evidence_is_not_reported_as_a_failure_to_transfer(repo):
    """Minimum support is part of the decision rule, and the held-out split is smaller.
    A family with 5 held-out matches is REJECTED without its rate ever being measured.
    Counting that beside a genuine rate change would say "the filters did not transfer"
    when the truth is "there was not enough held-out data to ask" - and it would say it
    on camera, in demo scene 4.
    """
    _write_heldout_consensus(repo, [f"h{i}" for i in range(5)])
    payload = report.heldout_report({}, repo)

    reasons = payload["flip_reasons"]
    assert reasons["insufficient_heldout_support"] >= 1
    assert reasons["different_harm_rate"] == 0

    goblok = next(f for f in payload["families"] if f["name"] == "GOBLOK")
    assert goblok["transfer"]["flip_reason"] == "insufficient_heldout_support"
    assert goblok["transfer"]["heldout_meets_min_support"] is False
    assert reasons["insufficient_heldout_support"] + reasons["different_harm_rate"] == \
        payload["n_verdict_flips"]


def test_the_flip_note_explains_the_distinction_in_words(repo):
    """The JSON is read by people who did not write it."""
    _write_heldout_consensus(repo, [f"h{i}" for i in range(5)])
    note = report.heldout_report({}, repo)["flip_reasons"]["note"]
    assert "not automatically a failure to transfer" in note


def test_heldout_labels_come_from_the_heldout_consensus_not_the_dev_one(repo):
    """If this read `consensus.parquet`, the transfer table would be scoring Channel E
    matches against development labels - a comparison of a split with itself."""
    _write_heldout_consensus(repo, [f"h{i}" for i in range(5)])
    payload = report.heldout_report({}, repo)
    fam = next(f for f in payload["families"] if f["name"] == "GOBLOK")
    # the held-out corpus contains only `mending mati`, so GOBLOK has no matches there
    assert fam["heldout"]["n_all"] == 0
    # while its dev numbers are untouched and still come from recommendations.json
    assert fam["dev"]["n_all"] == N_PER_FAMILY


# ---------------------------------------------------------------------------
# D24 / D27 - the screen check covers what goes on camera, not what passed
# ---------------------------------------------------------------------------

def test_d24_screen_check_includes_rejected_families_not_only_suggested(repo):
    """`jelek -> REJECTED` is demo scene 2. A screen check that only audited suggested
    phrases would never look at the scene the project is built around."""
    payload = report.run_screen_check({}, repo, top=15, samples=5)
    verdicts = {f["phrase"]: f["verdict"] for f in payload["families"]}
    assert "JELEK" in verdicts and verdicts["JELEK"] == "REJECTED"
    assert "GOBLOK" in verdicts and verdicts["GOBLOK"] == "SUGGESTED"


def test_d27_an_arbitrary_phrase_can_be_named_including_from_the_heldout_split(repo):
    """Phrases surfaced from Channel E also appear on camera, so `--phrase` scans the
    held-out split too. This is the one deliberate, named exception to dev-only reads.

    `mending mati` exists ONLY in the held-out split, so a dev-only screen check could
    never surface it and the ad hoc path is the only way it reaches the manual pass.
    """
    payload = report.run_screen_check({}, repo, top=0, samples=5,
                                      extra_phrases=[HELDOUT_ONLY_PHRASE])
    names = [f["phrase"] for f in payload["families"]]
    assert HELDOUT_ONLY_PHRASE in names
    entry = next(f for f in payload["families"] if f["phrase"] == HELDOUT_ONLY_PHRASE)
    assert entry["n_all"] == 5, "the ad hoc scan must reach the held-out rows"
    assert "ad hoc" in entry["source"]


def test_a_named_phrase_that_is_a_known_variant_resolves_to_its_family(repo):
    """`--phrase goblok` must surface the GOBLOK FAMILY, not a lone variant. The
    recommendation unit is the family (D33), so auditing one variant in isolation would
    show numbers that differ from the ones on screen."""
    payload = report.run_screen_check({}, repo, top=0, samples=5,
                                      extra_phrases=["goblok"])
    names = [f["phrase"] for f in payload["families"]]
    assert names == ["GOBLOK"]


def test_screen_check_never_filters_by_verdict(repo):
    """`screen_check()` itself takes arbitrary pairs with no verdict logic at all."""
    from src.phrases.mine import PhraseScore
    from src.schemas import Comment

    rejected = PhraseScore("jelek", n_harmful=0, n_non_harmful=30, n_ambiguous=0,
                           n_criticism=30)
    comments = [Comment(f"c{i}", "ch", "v", "h", "2026-01-01", "jelek", "jelek")
                for i in range(30)]
    items = screen_check([(rejected, comments)], samples_per_phrase=20)
    assert items[0].verdict == "REJECTED"
    assert len(items[0].sampled) == 20


def test_the_automatic_top_n_shows_suggested_families_before_rejected_ones(repo):
    """Ranked by support alone the list fills with the corpus's commonest words - all
    obviously rejected - and the 25-minute pass never looks at a single filter that
    will actually hide someone's comment. Order changes; membership does not."""
    payload = report.run_screen_check({}, repo, top=2, samples=5)
    assert [f["phrase"] for f in payload["families"]][0] == "GOBLOK"


def test_reordering_did_not_become_verdict_filtering(repo):
    """D24 - rejected families must still be reachable by the automatic selection."""
    payload = report.run_screen_check({}, repo, top=15, samples=5)
    verdicts = {f["verdict"] for f in payload["families"]}
    assert verdicts == {"SUGGESTED", "REJECTED"}


def test_screen_check_samples_twenty_by_default_for_the_power_it_buys(repo):
    """T5 - 20 samples catch a truly-70%-precise phrase 99.9% of the time."""
    payload = report.run_screen_check({}, repo, top=15, samples=20)
    assert payload["samples_per_phrase"] == 20


def test_mirror_sample_matches_the_frozen_sampling_exactly(repo):
    """Provenance is attached by re-deriving which comment each sample came from.
    If `screen_check()`'s sampling is ever edited without updating the mirror, the
    provenance line would silently describe the wrong comment. This catches that."""
    from src.phrases.mine import PhraseScore
    from src.schemas import Comment

    score = PhraseScore("x", 5, 5, 0, 0)
    comments = [Comment(f"c{i}", "ch", "v", "h", "2026-01-01", f"text {i}", f"text {i}")
                for i in range(10)]
    pairs = [(score, comments)]
    items = screen_check(pairs, samples_per_phrase=4, seed=7)
    mirrored = _mirror_sample(pairs, 4, seed=7)
    by_id = {c.comment_id: c.text_normalized for c in comments}
    assert [by_id[cid] for cid in mirrored[0]] == items[0].sampled


# ---------------------------------------------------------------------------
# R8 / R9 - what may reach a display artefact
# ---------------------------------------------------------------------------

def test_r9_no_real_account_or_figure_id_reaches_a_display_artefact(repo):
    """Naming a public figure beside harassment examples is the UU PDP exposure R8
    exists to avoid, and three platforms makes it worse rather than better."""
    payload = report.run_screen_check({}, repo, top=15, samples=5)
    blob = json.dumps(payload, ensure_ascii=False)
    assert ACCOUNT not in blob
    assert FIGURE not in blob
    assert "UC_heldout_account" not in blob


def test_r8_no_author_hash_reaches_a_display_artefact(repo):
    payload = report.run_screen_check({}, repo, top=15, samples=5)
    blob = json.dumps(payload, ensure_ascii=False)
    for row in _corpus_rows():
        assert row["author_hash"] not in blob


def test_r9_provenance_is_pseudonymized_but_still_says_which_app_and_post(repo):
    """The reviewer needs the context - a comment reads differently on X than on a
    YouTube reply chain - without the artefact naming anybody."""
    payload = report.run_screen_check({}, repo, top=15, samples=5)
    seen = [s["provenance"] for f in payload["families"] for s in f["samples"]]
    assert seen, "samples must carry provenance"
    for line in seen:
        assert "Figure " in line
        assert "post " in line
        assert ACCOUNT not in line and FIGURE not in line


def test_provenance_prefers_an_upstream_pseudonymized_column():
    """Ingest pseudonymizes at source (build contract §1); this must not re-derive
    and override what ingest already wrote."""
    rows = [{"comment_id": "c1", "provenance": "X · Figure C · post 2 of 9",
             "figure_id": FIGURE, "post_id": "p", "platform": "x"}]
    assert build_provenance(rows)["c1"] == "X · Figure C · post 2 of 9"


def test_provenance_pseudonymizes_a_pre_migration_corpus_too():
    """A corpus with only `channel_id`/`video_id` must still never emit the raw id."""
    rows = [{"comment_id": "c1", "channel_id": ACCOUNT, "video_id": "vid123"}]
    line = build_provenance(rows)["c1"]
    assert ACCOUNT not in line and "vid123" not in line
    assert "Figure A" in line


def test_figure_labels_are_short_stable_and_content_free():
    assert _figure_label(0) == "A"
    assert _figure_label(25) == "Z"
    assert _figure_label(26) == "AA"


# ---------------------------------------------------------------------------
# R9 - disclosing the collection mix
# ---------------------------------------------------------------------------

def test_r9_collection_mix_discloses_platforms_and_live_versus_imported():
    """An X arm built from 400 imported replies does not support the same claim as
    9,000 live YouTube comments, and the asymmetry belongs on the face of the report."""
    mix = collection_mix(_corpus_rows())
    assert mix["platform_data_available"] is True
    assert mix["collection_mode_data_available"] is True
    assert set(mix["comments_per_platform"]) == {"youtube", "instagram", "x"}
    assert "import" in mix["collection_mode_per_platform"]["instagram"]


def test_r9_a_corpus_without_platform_says_so_rather_than_inventing_a_mix():
    mix = collection_mix([{"comment_id": "c1"}])
    assert mix["platform_data_available"] is False
    assert "not yet present" in mix["note"]


def test_screen_check_payload_carries_the_collection_mix(repo):
    payload = report.run_screen_check({}, repo, top=15, samples=5)
    assert payload["collection_mix"]["platform_data_available"] is True


# ---------------------------------------------------------------------------
# R5 - reporting language
# ---------------------------------------------------------------------------

def test_r5_every_figure_is_attributed_to_the_automated_system(repo):
    payload = report.run_screen_check({}, repo, top=15, samples=5)
    assert payload["according_to"] == ATTRIBUTION
    assert "ground truth" in payload["note"]


def test_markdown_is_laid_out_for_the_manual_pass(repo):
    """~25 minutes of proofreading: numbered samples, provenance, a box to tick."""
    payload = report.run_screen_check({}, repo, top=15, samples=5)
    md = report._screen_check_markdown(payload)
    assert ATTRIBUTION in md
    assert "| # | comment" in md and "provenance" in md
    assert "☐" in md, "each sample needs somewhere to record the verdict"
    assert ACCOUNT not in md and FIGURE not in md


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------

def test_cli_requires_a_subcommand():
    with pytest.raises(SystemExit):
        report.main([])


def test_cli_exposes_both_documented_subcommands(capsys):
    for command in ("heldout", "screen_check"):
        with pytest.raises(SystemExit):
            report.main([command, "--help"])
        assert command in capsys.readouterr().out
