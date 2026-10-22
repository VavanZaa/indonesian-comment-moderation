"""The three baseline arms (AGENTS.md R6, D18, T4, R9, build contract §7).

The arms exist so the improvement DECOMPOSES. Reporting only `ours - baseline_0` books
the normalizer's gain as learning, and catching `g0blok` then proves nothing except that
you have a normalizer. These tests pin that decomposition, the parallelism that makes it
meaningful, and the sampling design behind every precision figure.
"""
from __future__ import annotations

import json

import pytest

from src import baseline
from src.baseline import (
    ARMS,
    ArmSpec,
    CorpusRow,
    TermMatcher,
    build_comparison,
    classify,
    read_static_list,
    run_arm,
    suggested_family_terms,
    tokenize,
)


# ---------------------------------------------------------------------------
# a tiny corpus, built so every assertion below is checkable by hand
# ---------------------------------------------------------------------------

def _rows() -> list[CorpusRow]:
    """`g0blok` is leetspeak that only the NORMALIZED column resolves to `goblok`.

    That single row is the entire normalizer effect in miniature: baseline_0 reads the
    original text and misses it, baseline_1 reads the normalized text and catches it,
    and the static list is identical in both.
    """
    return [
        # harmful, plain spelling - every arm catches this one
        CorpusRow("c1", "dasar goblok banget", "dasar goblok banget", "youtube"),
        # harmful, leetspeak - only the arms reading text_normalized catch it
        CorpusRow("c2", "g0blok lu", "goblok lu", "instagram"),
        # legitimate criticism containing `jelek` - the static list hides it, we must not
        CorpusRow("c3", "videonya jelek banget", "videonya jelek banget", "youtube"),
        CorpusRow("c4", "editingnya jelek", "editingnya jelek", "x"),
        # harmful phrase the learned list knows and the static list does not
        CorpusRow("c5", "mending mati aja", "mending mati aja", "x"),
        # benign, nothing fires
        CorpusRow("c6", "keren banget kak", "keren banget kak", "youtube"),
        # ambiguous - still hidden by any filter that matches it (R1)
        CorpusRow("c7", "wkwk goblok lu", "wkwk goblok lu", "instagram"),
    ]


LABELS = {
    "c1": "harmful",
    "c2": "harmful",
    "c3": "legitimate_criticism",
    "c4": "legitimate_criticism",
    "c5": "harmful",
    "c6": "benign_other",
    "c7": "ambiguous",
}

STATIC_TERMS = ["goblok", "jelek"]          # the naive creator's list
LEARNED_TERMS = ["goblok", "mending mati"]  # what the system suggested


def _results(audit_samples: int = 20, seed: int = 0):
    rows = _rows()
    terms_for = {"static": STATIC_TERMS, "learned": LEARNED_TERMS, "hybrid": LEARNED_TERMS}
    return {
        spec.arm_id: run_arm(spec, rows, LABELS, terms_for[spec.filters],
                             audit_samples=audit_samples, seed=seed)
        for spec in ARMS
    }


def _comparison(**kw) -> dict:
    return build_comparison(
        _results(),
        static_terms=STATIC_TERMS,
        learned_terms=LEARNED_TERMS,
        suggested_families=["GOBLOK", "MENDING MATI"],
        static_list_path="data/raw/static_profanity_id.txt",
        audit_samples=kw.get("audit_samples", 20),
    )


# ---------------------------------------------------------------------------
# D18 - the arms are PARALLEL, not a pipeline stage
# ---------------------------------------------------------------------------

def test_d18_no_baseline_output_can_reach_the_recommender():
    """The mistake D18 exists to prevent: making a baseline an input to scoring.

    Implementing the architecture diagram literally - baseline feeding the recommender -
    makes the comparison circular. Nothing under `src/phrases/` may import this module
    or read its artefact.
    """
    import pathlib

    phrases = pathlib.Path(__file__).resolve().parents[1] / "src" / "phrases"
    for path in phrases.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "baseline_comparison" not in source, (
            f"{path.name} reads the baseline artefact - that is the D18 circularity")
        assert "import baseline" not in source and "from ..baseline" not in source, (
            f"{path.name} imports the baseline module - D18 forbids that edge")


def test_d18_an_arm_cannot_observe_another_arm():
    """`run_arm` is handed one text column and one term list and nothing else, so the
    arms are independent by construction. Order of evaluation must not matter."""
    forward = _results()
    reversed_specs = list(reversed(ARMS))
    rows = _rows()
    terms_for = {"static": STATIC_TERMS, "learned": LEARNED_TERMS, "hybrid": LEARNED_TERMS}
    backward = {
        spec.arm_id: run_arm(spec, rows, LABELS, terms_for[spec.filters])
        for spec in reversed_specs
    }
    for arm_id in ("baseline_0", "baseline_1", "ours", "hybrid"):
        assert forward[arm_id].fired_ids == backward[arm_id].fired_ids
        assert forward[arm_id].hidden == backward[arm_id].hidden


# ---------------------------------------------------------------------------
# R6 - the decomposition
# ---------------------------------------------------------------------------

def test_r6_three_arms_exist_and_share_the_normalizer():
    assert [s.arm_id for s in ARMS] == ["baseline_0", "baseline_1", "ours", "hybrid"]
    by_id = {s.arm_id: s for s in ARMS}
    assert by_id["baseline_0"].text_field == "text_original"
    assert by_id["baseline_1"].text_field == "text_normalized"
    assert by_id["ours"].text_field == "text_normalized"
    assert by_id["hybrid"].text_field == "text_normalized"
    # baseline_1 and ours differ ONLY in the term list; baseline_0 and baseline_1 ONLY
    # in the text column. That is what makes each delta attributable to one cause.
    assert by_id["baseline_0"].filters == by_id["baseline_1"].filters == "static"
    assert by_id["ours"].filters == "learned"
    assert by_id["hybrid"].filters == "hybrid"


# ---------------------------------------------------------------------------
# hybrid arm (2026-09-09, D37 follow-up) - a word match is a candidate only;
# the comment's own label confirms before it counts as hidden
# ---------------------------------------------------------------------------

def test_hybrid_fires_only_when_a_term_matches_AND_the_comment_is_labelled_harmful():
    results = _results()
    ours, hybrid = results["ours"], results["hybrid"]

    # ours (word match alone) fires on c1, c2, c5 (harmful) AND c7 (ambiguous, "goblok")
    assert set(ours.fired_ids) == {"c1", "c2", "c5", "c7"}
    # hybrid drops c7 - the word matched, but c7's own label is not "harmful"
    assert set(hybrid.fired_ids) == {"c1", "c2", "c5"}


def test_hybrid_has_identical_recall_to_ours_and_zero_collateral_by_construction():
    """The property documented above ARMS in src/baseline.py: this is not an empirical
    finding, it follows from the firing rule (hits AND bucket == 'harmful')."""
    results = _results()
    ours, hybrid = results["ours"], results["hybrid"]

    assert hybrid.harmful_caught == ours.harmful_caught
    assert hybrid.hidden["legitimate_criticism"] == 0
    assert hybrid.hidden["ambiguous"] == 0
    assert hybrid.hidden["benign_other"] == 0
    assert hybrid.comments_hidden < ours.comments_hidden, (
        "hybrid must hide a strict subset when ours has any non-harmful-bucket hits"
    )
    assert hybrid.precision()["lower"] == 1.0, (
        "trivially perfect by construction (gated on the same label used to score "
        "it) - NOT evidence of an independently accurate classifier"
    )


def test_confirmation_effect_decomposition_shows_collateral_removed_at_no_recall_cost():
    payload = _comparison()
    block = payload["decompositions"]["confirmation_effect"]
    assert block["definition"] == "hybrid - ours"
    d = block["deltas"]
    assert d["harmful_caught"] == 0, "recall must be unchanged (mathematical identity)"
    assert d["legitimate_criticism_hidden"] <= 0
    assert d["comments_hidden"] < 0, "hybrid must hide fewer comments than ours here"

    arm_ids = [a["arm"] for a in payload["arms"]]
    assert "hybrid" in arm_ids
    assert payload["headline"]["hybrid"] == 0, "no legitimate criticism reaches hybrid"


def test_r6_normalizer_effect_is_reported_separately_from_learning_effect():
    payload = _comparison()
    decomp = payload["decompositions"]
    assert decomp["normalizer_effect"]["definition"] == "baseline_1 - baseline_0"
    assert decomp["learning_effect"]["definition"] == "ours - baseline_1"
    # they must be different objects with independently computed deltas
    assert decomp["normalizer_effect"]["deltas"] is not decomp["learning_effect"]["deltas"]


def test_r6_normalizer_effect_is_exactly_the_leetspeak_row():
    """`g0blok` is caught by baseline_1 and missed by baseline_0, and nothing else
    differs between them - so the normalizer effect is +1 hidden, +1 harmful caught."""
    r = _results()
    assert "c2" not in r["baseline_0"].fired_ids, "original text does not resolve g0blok"
    assert "c2" in r["baseline_1"].fired_ids, "normalized text does"
    delta = _comparison()["decompositions"]["normalizer_effect"]["deltas"]
    assert delta["comments_hidden"] == 1
    assert delta["harmful_caught"] == 1


def test_r6_learning_effect_finds_a_phrase_the_static_list_never_had():
    r = _results()
    assert "c5" not in r["baseline_1"].fired_ids, "`mending mati` is not on the static list"
    assert "c5" in r["ours"].fired_ids, "the learned list has it"


# ---------------------------------------------------------------------------
# demo scene 2 - the contrast the whole project is built around
# ---------------------------------------------------------------------------

def test_scene_2_static_list_hides_jelek_and_the_learned_filters_do_not():
    """"The static list blocks `jelek`, this system rejects it." Both criticism rows
    must be hidden by the static arms and untouched by ours."""
    r = _results()
    for cid in ("c3", "c4"):
        assert cid in r["baseline_0"].fired_ids
        assert cid in r["baseline_1"].fired_ids
        assert cid not in r["ours"].fired_ids

    assert r["baseline_0"].criticism_hidden == 2
    assert r["baseline_1"].criticism_hidden == 2
    assert r["ours"].criticism_hidden == 0


def test_headline_metric_is_collateral_damage_not_accuracy():
    """The number that carries the demo is collateral hidden, not a precision score.

    D38 widened this from `legitimate_criticism_hidden` to `collateral_hidden`
    (criticism + public_figure_invective). Reporting only the criticism half would
    now UNDERSTATE the damage, because the political comments a blocked-word list
    hides moved into the invective bucket - they did not stop being collateral.
    """
    headline = _comparison()["headline"]
    assert headline["metric"] == "collateral_hidden"
    assert headline["ours"] < headline["baseline_0"]
    # and both halves must stay separately visible, not blended away
    assert set(headline["breakdown"]["baseline_0"]) == {
        "legitimate_criticism", "public_figure_invective"}


def test_public_figure_invective_is_collateral_not_harm():
    """D38's two load-bearing properties: it must not inflate the harm rate, and it
    MUST gate R0's C term - a phrase firing on crude political speech is destroying
    speech aimed at power exactly as one firing on ordinary criticism is."""
    from src.schemas import COLLATERAL, HARMFUL, Label

    assert Label.PUBLIC_FIGURE_INVECTIVE not in HARMFUL
    assert Label.PUBLIC_FIGURE_INVECTIVE in COLLATERAL
    assert Label.LEGITIMATE_CRITICISM in COLLATERAL

    assert baseline.classify("public_figure_invective") == "public_figure_invective"
    assert "public_figure_invective" in baseline.BUCKETS


# ---------------------------------------------------------------------------
# T4 - precision is conditioned on the FIRED stratum, never the stream
# ---------------------------------------------------------------------------

def test_t4_precision_denominator_is_the_fired_stratum_not_the_stream():
    """A filter firing at 3% means a random sample of the stream is almost all
    non-fired comments. Dividing by the stream would report a precision nobody can act
    on. This bit an earlier version of the plan."""
    ours = _results()["ours"]
    fired = ours.comments_hidden
    assert fired < ours.n_stream, "the fixture must have non-fired comments to be a test"

    # ours fires on c1 (harmful), c2 (harmful), c5 (harmful), c7 (ambiguous)
    assert fired == 4
    p = ours.precision()
    assert p["lower"] == pytest.approx(3 / 4)      # ambiguous counted against us
    assert p["upper"] == pytest.approx(4 / 4)      # ambiguous counted for us
    assert p["resolved"] == pytest.approx(3 / 3)   # resolved only
    # and emphatically NOT the stream denominator
    assert p["lower"] != pytest.approx(3 / ours.n_stream)


def test_t4_precision_is_an_interval_whose_width_is_the_ambiguous_share():
    """Same identity as R1: upper - lower is exactly the ambiguity share of the
    fired stratum. A bare point estimate is never enough."""
    ours = _results()["ours"]
    p = ours.precision()
    ambiguous_share = ours.hidden["ambiguous"] / ours.comments_hidden
    assert p["upper"] - p["lower"] == pytest.approx(ambiguous_share)


def test_ambiguous_matches_are_counted_because_a_filter_still_hides_them():
    """R1/R2b - you do not get to exclude a comment from a consequence it receives."""
    ours = _results()["ours"]
    assert "c7" in ours.fired_ids
    assert ours.hidden["ambiguous"] == 1


def test_audit_sample_is_drawn_from_the_fired_stratum(tmp_path):
    payload = _comparison()
    for arm in payload["arms"]:
        fired = set(_results()[arm["arm"]].fired_ids)
        for cid in arm.get("audit_sample", []):
            assert cid in fired, "an audit sample must come from what the arm fired on"


# ---------------------------------------------------------------------------
# R9 - per-platform breakdown
# ---------------------------------------------------------------------------

def test_r9_every_arm_reports_a_platform_breakdown():
    payload = _comparison()
    for arm in payload["arms"]:
        assert "by_platform" in arm, f"{arm['arm']} has no per-platform view"
        assert set(arm["by_platform"]) == {"youtube", "instagram", "x"}


def test_r9_platform_breakdown_sums_to_the_pooled_numbers():
    """The two views are the same counters, so they cannot drift apart."""
    for arm_id, result in _results().items():
        by_platform = result.platform_breakdown()
        assert sum(p["n_stream"] for p in by_platform.values()) == result.n_stream
        assert sum(p["comments_hidden"] for p in by_platform.values()) == result.comments_hidden
        assert sum(p["harmful_caught"] for p in by_platform.values()) == result.harmful_caught


def test_r9_a_one_platform_family_is_visible_rather_than_averaged_away():
    """`mending mati` only appears on X in this corpus. Pooled, it is one row in seven;
    per platform it is the entire X harmful catch. That is the point of the breakdown."""
    ours = _results()["ours"].platform_breakdown()
    assert ours["x"]["harmful_caught"] == 1
    assert ours["x"]["comments_hidden"] == 1


# ---------------------------------------------------------------------------
# matching rules shared by all three arms
# ---------------------------------------------------------------------------

def test_matching_is_whole_token_so_jelek_does_not_match_jeleknya():
    m = TermMatcher(["jelek"])
    assert m.fires("videonya jelek banget")
    assert not m.fires("jeleknya kebangetan")


def test_digits_are_part_of_a_token_so_g0blok_is_not_goblok_on_original_text():
    """If the tokenizer split on digits, baseline_0 would catch `g0blok` for free and
    the normalizer effect would silently vanish."""
    assert tokenize("g0blok") == ["g0blok"]
    assert not TermMatcher(["goblok"]).fires("g0blok lu")


def test_multi_word_terms_match_as_a_phrase():
    m = TermMatcher(["mending mati"])
    assert m.fires("mending mati aja")
    assert not m.fires("mending pergi, mati lampu")


def test_matching_is_case_insensitive_in_every_arm():
    """Otherwise plain capitalisation would be booked as a normalizer effect."""
    assert TermMatcher(["goblok"]).fires("DASAR GOBLOK")


# ---------------------------------------------------------------------------
# reading the two filter lists
# ---------------------------------------------------------------------------

def test_static_list_does_not_split_on_an_inline_hash(tmp_path):
    """`g#blok` is a censored variant, not a comment. Only a leading `#` is a comment."""
    path = tmp_path / "static.txt"
    path.write_text("# creator list\ngoblok\ng#blok\n\njelek\n", encoding="utf-8")
    assert read_static_list(path) == ["goblok", "g#blok", "jelek"]


def test_learned_terms_read_the_verdict_and_never_recompute_the_rule():
    """Build contract §6: the verdict comes from `PhraseScore.recommend()`. If this
    module applied its own threshold, the demo and the comparison could disagree."""
    recs = {"families": [
        {"name": "GOBLOK", "verdict": "SUGGESTED", "variants": ["goblok", "g0blok"],
         "p_harmful_lower": 0.0},        # deliberately contradicts the verdict
        {"name": "JELEK", "verdict": "REJECTED", "variants": ["jelek"],
         "p_harmful_lower": 0.99},       # ditto, the other way
    ]}
    terms, names = suggested_family_terms(recs)
    assert names == ["GOBLOK"]
    assert "jelek" not in terms
    assert set(terms) == {"goblok", "g0blok"}


def test_a_variant_shared_by_two_families_is_not_double_counted():
    recs = {"families": [
        {"name": "A", "verdict": "SUGGESTED", "variants": ["goblok", "tolol"]},
        {"name": "B", "verdict": "SUGGESTED", "variants": ["GOBLOK", "bego"]},
    ]}
    terms, _ = suggested_family_terms(recs)
    assert len(terms) == len({t.casefold() for t in terms}) == 3


# ---------------------------------------------------------------------------
# label bucketing
# ---------------------------------------------------------------------------

def test_a_null_label_is_ambiguous_not_benign():
    """Coercing the undecided tier to a class is exactly what R1 forbids."""
    assert classify(None) == "ambiguous"
    assert classify("", "ambiguous") == "ambiguous"
    assert classify("harassment_personal", "ambiguous") == "ambiguous"


def test_criticism_is_never_folded_into_benign_other():
    """`legitimate_criticism` is load-bearing; merging it makes scene 2 unproducible."""
    assert classify("legitimate_criticism") == "legitimate_criticism"
    assert classify("benign_other") == "benign_other"
    assert classify("harassment_personal") == "harmful"
    assert classify("sexual_harm") == "harmful"
    assert classify("threat_privacy") == "harmful"


# ---------------------------------------------------------------------------
# R5 / R8 - what the artefact is allowed to say and contain
# ---------------------------------------------------------------------------

def test_r5_every_figure_is_attributed_to_the_automated_system():
    payload = _comparison()
    assert payload["according_to"] == baseline.ATTRIBUTION
    assert payload["headline"]["according_to"] == baseline.ATTRIBUTION
    blob = json.dumps(payload)
    assert "accurate for Indonesian" not in blob


def test_r8_the_corpus_loader_uses_an_allowlist_so_no_handle_can_leak():
    assert baseline.CORPUS_COLUMNS == ("comment_id", "text_original", "text_normalized")
    assert "author_hash" not in baseline.CORPUS_COLUMNS
    blob = json.dumps(_comparison())
    for forbidden in ("author", "handle", "@"):
        assert forbidden not in blob.lower() or forbidden == "@", blob[:200]


def test_the_artefact_states_the_known_limitation_it_cannot_fix():
    """F2 - no stance detection, so a comment condemning a phrase is still matched."""
    limitations = " ".join(_comparison()["limitations"]).lower()
    assert "stance" in limitations
