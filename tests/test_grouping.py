"""Family grouping (build contract §6): candidates -> canonicalize_variant fold ->
optional rapidfuzz merge -> `score_family` over the deduplicated union.

Injects its own fold so these tests are independent of `src/normalize/text.py`,
which another agent is writing in parallel (build contract §2 pins the name;
`group_candidates(fold=...)` exists specifically so tests need not import it).
"""
from __future__ import annotations

import re

import pytest

from src.phrases.grouping import (
    group_candidates,
    platform_support,
    recommendations_payload,
    score_families,
)
from src.schemas import ConsensusRecord, Label, Tier

_WEIGHT = {Tier.HIGH: 1.0, Tier.MEDIUM: 0.5, Tier.AMBIGUOUS: 0.0}


_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b"})


def _fold(text: str) -> str:
    """A tiny stand-in for `canonicalize_variant`: leet-map digits back to
    letters, strip non-alphanumerics, collapse any repeated character to one -
    enough to fold g0blok / goblokkk onto "goblok" without depending on the
    normalize lane."""
    s = text.lower().translate(_LEET)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return re.sub(r"(.)\1+", r"\1", s)


def _rec(cid: str, label: Label | None, tier: Tier = Tier.HIGH) -> ConsensusRecord:
    return ConsensusRecord(cid, label, tier, _WEIGHT[tier])


def _ambiguous(cid: str) -> ConsensusRecord:
    return ConsensusRecord(cid, None, Tier.AMBIGUOUS, 0.0)


# --- grouping: fold, union, singleton family --------------------------------

def test_group_candidates_folds_spelling_variants_into_one_family():
    matches = {
        "goblok": ["c1", "c2"],
        "g0blok": ["c3"],
        "goblokkk": ["c1", "c4"],   # c1 also matched "goblok"
    }
    families = group_candidates(matches, fold=_fold, merge_similarity=None)
    assert len(families) == 1
    fam = families[0]
    assert fam.name == "GOBLOK"
    assert set(fam.member_ids) == {"c1", "c2", "c3", "c4"}


def test_a_phrase_that_groups_with_nothing_is_a_family_of_size_one():
    matches = {"anjir": [f"c{i}" for i in range(10)]}
    families = group_candidates(matches, fold=_fold, merge_similarity=None)
    assert len(families) == 1
    assert families[0].variants == ["anjir"]
    assert families[0].n_all == 10


def test_family_merge_by_rapidfuzz_combines_near_identical_keys():
    """rapidfuzz.fuzz.ratio('mantap', 'mantab') == 83.33: below the default 92
    cutoff (no merge) but above a relaxed 80 cutoff (merges)."""
    matches = {"mantap": ["c1"], "mantab": ["c2"]}   # _fold leaves these distinct
    default = group_candidates(matches, fold=_fold)
    assert len(default) == 2

    relaxed = group_candidates(matches, fold=_fold, merge_similarity=80.0)
    assert len(relaxed) == 1
    assert set(relaxed[0].member_ids) == {"c1", "c2"}


# --- a comment matching two variants of one family contributes ONCE ----------

def test_a_comment_matching_two_variants_contributes_once_through_scoring():
    """`build_family`/`score_family` are already proven not to double count in
    isolation (tests/test_families.py); this exercises the same guarantee
    through grouping's own path from raw candidate match sets."""
    matches = {"goblok": ["c1", "c2"], "goblokkk": ["c1"]}
    families = group_candidates(matches, fold=_fold, merge_similarity=None)
    records = {
        "c1": _rec("c1", Label.HARASSMENT_PERSONAL),
        "c2": _rec("c2", Label.HARASSMENT_PERSONAL),
    }
    [(fam, score)] = score_families(families, matches, records)
    assert fam.n_all == 2, "c1 counted once despite matching two variants"
    assert score.n_all == 2
    assert score.n_harmful == 2


# --- singletons go through score_family, same as multi-variant families ------

def test_singleton_family_is_scored_by_score_family_not_skipped():
    matches = {"tolol": [f"c{i}" for i in range(30)]}
    families = group_candidates(matches, fold=_fold, merge_similarity=None)
    assert len(families[0].variants) == 1
    records = {f"c{i}": _rec(f"c{i}", Label.HARASSMENT_PERSONAL) for i in range(26)}
    records.update({f"c{i}": _rec(f"c{i}", Label.BENIGN_OTHER) for i in range(26, 30)})
    [(_, score)] = score_families(families, matches, records)
    assert score.n_all == 30
    assert score.recommend()


# --- the worked examples ------------------------------------------------------

def _anjir_records() -> dict[str, ConsensusRecord]:
    """docs/04-architecture-mapping.md §6 / mine.py's PhraseScore docstring:
    40 harmful / 10 criticism / 50 ambiguous over 100 matches."""
    records: dict[str, ConsensusRecord] = {}
    for i in range(40):
        records[f"c{i}"] = _rec(f"c{i}", Label.HARASSMENT_PERSONAL)
    for i in range(40, 50):
        records[f"c{i}"] = _rec(f"c{i}", Label.LEGITIMATE_CRITICISM)
    for i in range(50, 100):
        records[f"c{i}"] = _ambiguous(f"c{i}")
    return records


def test_anjir_worked_example_round_trips_and_is_rejected():
    records = _anjir_records()
    matches = {"anjir": list(records)}
    families = group_candidates(matches, fold=_fold, merge_similarity=None)
    [(fam, score)] = score_families(families, matches, records)

    assert score.n_all == 100
    assert score.p_harmful_lower() == pytest.approx(0.400)
    assert score.p_harmful_resolved() == pytest.approx(0.800)
    assert score.p_harmful_upper() == pytest.approx(0.900)
    assert score.ambiguity_share == pytest.approx(0.500)
    assert not score.recommend()

    payload = recommendations_payload([(fam, score)], records, {})
    out = payload["families"][0]
    assert out["verdict"] == "REJECTED"
    assert out["p_harmful_lower"] == pytest.approx(0.400)


def test_mending_mati_worked_example_is_suggested():
    """build contract §5's own worked JSON example: 118 harmful / 4 non-harmful
    (2 of those criticism) / 4 ambiguous over 126 matches -> lower ~0.937,
    SUGGESTED. Matches docs/04-architecture-mapping.md §6's table exactly."""
    ids = [f"c{i}" for i in range(126)]
    records: dict[str, ConsensusRecord] = {}
    it = iter(ids)
    for _ in range(118):
        cid = next(it)
        records[cid] = _rec(cid, Label.HARASSMENT_PERSONAL)
    for _ in range(2):
        cid = next(it)
        records[cid] = _rec(cid, Label.LEGITIMATE_CRITICISM)
    for _ in range(2):
        cid = next(it)
        records[cid] = _rec(cid, Label.BENIGN_OTHER)
    for cid in it:
        records[cid] = _ambiguous(cid)

    matches = {"mending mati": ids}
    families = group_candidates(matches, fold=_fold, merge_similarity=None)
    [(fam, score)] = score_families(families, matches, records)

    assert (score.n_harmful, score.n_non_harmful, score.n_ambiguous, score.n_criticism) \
        == (118, 4, 4, 2)
    assert score.p_harmful_lower() == pytest.approx(118 / 126, abs=1e-3)
    assert score.recommend()

    payload = recommendations_payload([(fam, score)], records, {})
    assert payload["families"][0]["verdict"] == "SUGGESTED"


# --- verdict comes from PhraseScore.recommend, never recomputed --------------

def test_verdict_agrees_with_phrase_score_recommend():
    matches = {"tolol": [f"c{i}" for i in range(30)]}
    families = group_candidates(matches, fold=_fold, merge_similarity=None)
    records = {f"c{i}": _rec(f"c{i}", Label.HARASSMENT_PERSONAL) for i in range(26)}
    records.update({f"c{i}": _rec(f"c{i}", Label.BENIGN_OTHER) for i in range(26, 30)})
    scored = score_families(families, matches, records)
    payload = recommendations_payload(scored, records, {})
    for (_, score), out in zip(scored, payload["families"]):
        assert (out["verdict"] == "SUGGESTED") == score.recommend()


def test_verdict_respects_configured_thresholds_not_hardcoded_defaults():
    """Same counts, different config thresholds must be able to flip the verdict -
    proof the payload calls recommend() with the passed-in thresholds."""
    ids = [f"c{i}" for i in range(30)]
    records = {cid: _rec(cid, Label.HARASSMENT_PERSONAL) for cid in ids[:23]}
    records.update({cid: _rec(cid, Label.BENIGN_OTHER) for cid in ids[23:]})
    matches = {"x": ids}
    families = group_candidates(matches, fold=_fold, merge_similarity=None)
    scored = score_families(families, matches, records)

    strict = recommendations_payload(scored, records, {}, threshold=0.85)
    lenient = recommendations_payload(scored, records, {}, threshold=0.70)
    assert strict["families"][0]["verdict"] == "REJECTED"
    assert lenient["families"][0]["verdict"] == "SUGGESTED"


def test_decision_rule_reflects_the_actual_thresholds_passed_in():
    """Regression (D37): decision_rule used to be a hardcoded '0.85' literal that
    stayed put even when the real threshold moved - a reader trusting this field
    would see a rule that was not the one actually applied. It must always describe
    whatever thresholds were actually used to compute the verdicts beside it.

    `recommendations_payload`'s own default stays 0.85 (matching the un-modifiable
    `PhraseScore.recommend()` default it must agree with - see
    test_verdict_agrees_with_phrase_score_recommend). The project's actual 0.60
    (AGENTS.md R0 / D37) is a `run()`-level config fallback, not this function's
    default - this test proves the STRING follows whatever threshold is passed in,
    at any value, rather than being a hand-written literal that can drift from it."""
    scored: list = []
    records: dict = {}

    payload_current_default = recommendations_payload(scored, records, {})
    assert payload_current_default["decision_rule"] == "L_H >= 0.85 AND C <= 0.1 AND n_all >= 25"

    payload_project_default = recommendations_payload(scored, records, {}, threshold=0.60)
    assert payload_project_default["decision_rule"] == "L_H >= 0.6 AND C <= 0.1 AND n_all >= 25"

    payload_custom = recommendations_payload(
        scored, records, {}, threshold=0.75, max_criticism=0.2, min_matches=50,
    )
    assert payload_custom["decision_rule"] == "L_H >= 0.75 AND C <= 0.2 AND n_all >= 50"


# --- training weights do not leak into grouping's own scoring path -----------

def test_score_families_ignores_training_weight_tier():
    ids_high = [f"h{i}" for i in range(20)]
    ids_medium = [f"m{i}" for i in range(20)]
    fams_high = group_candidates({"goblok": ids_high}, fold=_fold, merge_similarity=None)
    fams_medium = group_candidates({"goblok": ids_medium}, fold=_fold, merge_similarity=None)
    records_high = {cid: _rec(cid, Label.HARASSMENT_PERSONAL, Tier.HIGH) for cid in ids_high}
    records_medium = {cid: _rec(cid, Label.HARASSMENT_PERSONAL, Tier.MEDIUM) for cid in ids_medium}

    [(_, score_high)] = score_families(fams_high, {"goblok": ids_high}, records_high)
    [(_, score_medium)] = score_families(fams_medium, {"goblok": ids_medium}, records_medium)

    assert score_high.p_harmful_lower() == score_medium.p_harmful_lower() == 1.0
    assert score_high.n_harmful == score_medium.n_harmful == 20


# --- platform_support (R9/D35) -----------------------------------------------

def test_platform_support_sums_to_the_pooled_n_all():
    ids = [f"c{i}" for i in range(60)]
    records = {cid: _rec(cid, Label.HARASSMENT_PERSONAL) for cid in ids}
    platform_map = {cid: ("youtube" if i % 3 else "instagram") for i, cid in enumerate(ids)}
    matches = {"goblok": ids}
    families = group_candidates(matches, fold=_fold, merge_similarity=None)
    [(fam, score)] = score_families(families, matches, records)

    support = platform_support(fam.member_ids, records, platform_map)
    assert sum(v["n_all"] for v in support.values()) == score.n_all


def test_platform_support_is_reported_and_never_gates_the_verdict():
    """The pooled family fails (23/30 lower bound < 0.85), and both platform
    subsets individually are even weaker - but there is no per-platform key that
    could have overridden the pooled verdict (R0: exactly three terms)."""
    ids = [f"c{i}" for i in range(30)]
    records = {cid: _rec(cid, Label.HARASSMENT_PERSONAL) for cid in ids[:23]}
    records.update({cid: _rec(cid, Label.BENIGN_OTHER) for cid in ids[23:]})
    platform_map = {cid: "x" for cid in ids[:5]}
    platform_map.update({cid: "youtube" for cid in ids[5:]})
    matches = {"goblok": ids}
    families = group_candidates(matches, fold=_fold, merge_similarity=None)
    scored = score_families(families, matches, records)

    payload = recommendations_payload(scored, records, platform_map)
    out = payload["families"][0]
    assert set(out["platform_support"]) == {"x", "youtube"}
    for entry in out["platform_support"].values():
        assert set(entry) == {"n_all", "p_harmful_lower"}
    assert out["verdict"] == "REJECTED", "the pooled score decides, not a platform slice"


def test_platform_support_falls_back_to_unknown_with_an_empty_map():
    ids = [f"c{i}" for i in range(25)]
    records = {cid: _rec(cid, Label.HARASSMENT_PERSONAL) for cid in ids}
    matches = {"x": ids}
    families = group_candidates(matches, fold=_fold, merge_similarity=None)
    [(fam, score)] = score_families(families, matches, records)
    support = platform_support(fam.member_ids, records, {})
    assert support == {"unknown": {"n_all": 25, "p_harmful_lower": 1.0}}
