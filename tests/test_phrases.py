"""Enforces AGENTS.md R1: the harmful rate is an interval, and decisions use its
lower end. If these fail, the numbers on screen are claims you cannot support."""
import pytest

from src.phrases.mine import PhraseScore, score_phrase
from src.schemas import ConsensusRecord, Label, Tier


def test_bounds_identity_upper_minus_lower_is_ambiguity_share():
    """The exact identity. This is why ambiguity_share is the width of the identified
    set and not a decorative extra column."""
    for n_h, n_nh, n_amb in [(40, 10, 50), (310, 60, 90), (118, 4, 4), (160, 120, 120)]:
        n_all = n_h + n_nh + n_amb
        lower, upper = n_h / n_all, (n_h + n_amb) / n_all
        assert upper - lower == pytest.approx(n_amb / n_all)


def test_ambiguous_matches_are_in_the_operational_denominator():
    """A blocked-words list hides every match. n_all is what gets hidden."""
    s = PhraseScore("anjir", n_harmful=40, n_non_harmful=10, n_ambiguous=50, n_criticism=10)
    assert s.n_all == 100
    assert s.n_resolved == 50


def test_resolved_rate_is_optimistic_and_must_not_drive_the_decision():
    """'anjir' looks recommendable on resolved labels alone and is not.

    Named for the quantity, not the assumption: D20 abandoned "MAR" as a label because
    it names a missingness mechanism rather than a rate.
    """
    s = PhraseScore("anjir", 40, 10, 50, 10)
    assert s.p_harmful_resolved() > 0.75, "the resolved-only rate looks good"
    assert s.p_harmful_lower() < 0.50, "lower bound does not"
    assert not s.recommend(), "decisions use the lower bound"


def test_a_separate_ambiguity_gate_would_be_unreachable():
    """Why there is no max_ambiguity parameter.

    An earlier version had `ambiguity_share <= 0.25` as a third condition, and a test
    that appeared to exercise it using (600, 0, 400, 0) - a phrase whose LOWER BOUND
    was already 0.60 and which therefore failed on the first condition anyway. The
    test passed for the wrong reason and hid the redundancy.

    Deciding on the lower bound IS the ambiguity gate: n_ambiguous <= n_all - n_harmful,
    so L_H >= 0.85 forces ambiguity_share <= 0.15 directly.
    """
    worst = 0.0
    for n_all in range(10, 601, 10):
        for n_h in range(n_all + 1):
            s = PhraseScore("x", n_h, 0, n_all - n_h, 0)
            if s.p_harmful_lower() >= 0.85:
                worst = max(worst, s.ambiguity_share)
    assert worst <= 0.15, "passing the harm threshold already bounds ambiguity at 0.15"


def test_minimum_support_is_enforced_in_the_rule():
    """Without n_min, five all-harmful matches score 0.857 and get suggested on noise."""
    noise = PhraseScore("rare", n_harmful=5, n_non_harmful=0, n_ambiguous=0, n_criticism=0)
    assert noise.p_harmful_lower() == pytest.approx(1.0), "a perfect but meaningless rate"
    assert not noise.recommend(), "5 matches is not evidence - this is what min support is for"
    real = PhraseScore("common", n_harmful=100, n_non_harmful=0, n_ambiguous=0, n_criticism=0)
    assert real.recommend()


def test_criticism_share_denominator_is_all_matches():
    """C(f) = n_criticism / n_all. A filter hides ambiguous matches too, so
    collateral damage is measured against everything it would hide."""
    s = PhraseScore("x", n_harmful=50, n_non_harmful=25, n_ambiguous=25, n_criticism=20)
    assert s.criticism_share == pytest.approx(20 / 100)
    assert s.criticism_share != pytest.approx(20 / 75), "not over resolved only"


def test_ambiguous_records_are_not_coerced_to_benign():
    r = ConsensusRecord("c", None, Tier.AMBIGUOUS, 0.0)
    assert r.is_harmful is None, "R2: undecided stays undecided, never False"


def test_score_phrase_counts_each_bucket():
    def rec(label, tier=Tier.HIGH):
        return ConsensusRecord("c", label, tier, 1.0)
    matches = [
        rec(Label.HARASSMENT_PERSONAL), rec(Label.SEXUAL_HARM),
        rec(Label.LEGITIMATE_CRITICISM), rec(Label.BENIGN_OTHER),
        ConsensusRecord("c", None, Tier.AMBIGUOUS, 0.0),
    ]
    s = score_phrase("x", matches)
    assert (s.n_harmful, s.n_non_harmful, s.n_ambiguous, s.n_criticism) == (2, 2, 1, 1)
    assert s.n_all == 5 and s.n_resolved == 4
    assert s.criticism_share == pytest.approx(0.2)


def test_public_figure_invective_counts_as_collateral_in_C():
    """D38, from the real `memang anjing` case: 52% of its matches were comments like
    `pemerintah memang anjing`. Labelled harassment_personal they inflated L_H and left
    C at 0.000, so it reached SUGGESTED. As `public_figure_invective` they must land in
    the non-harmful side AND in C - otherwise the new label would fix the harm rate
    while still letting the phrase through the collateral gate."""
    def rec(label, tier=Tier.HIGH):
        return ConsensusRecord("c", label, tier, 1.0)

    matches = [rec(Label.HARASSMENT_PERSONAL)] + [rec(Label.PUBLIC_FIGURE_INVECTIVE)] * 4
    s = score_phrase("memang anjing", matches)

    assert s.n_harmful == 1, "invective must not count as harm"
    assert s.n_non_harmful == 4
    assert s.n_criticism == 4, "invective must count as collateral in C"
    assert s.criticism_share == pytest.approx(0.8)
    assert not s.recommend(threshold=0.60), "fails on both L_H and C"


def test_criticism_share_blocks_a_high_precision_phrase():
    """Demo scene 2. The conditions genuinely trade off - criticism is a subset of
    non-harmful, so raising criticism_share pulls the harm rate down. These numbers
    sit in the band where a phrase passes on precision and fails on collateral."""
    s = PhraseScore("jelek", n_harmful=350, n_non_harmful=50, n_ambiguous=0, n_criticism=45)
    assert s.p_harmful_lower() > 0.85
    assert s.criticism_share > 0.10
    assert not s.recommend()


def test_display_never_emits_a_bare_point_estimate():
    d = PhraseScore("anjir", 40, 10, 50, 10).display()
    assert "identified range" in d and "ambiguity share" in d
    assert "-" in d["identified range"], "must show an interval, not a single number"


def test_display_verdict_uses_the_same_threshold_recommend_would():
    """Regression (D37): display()'s embedded verdict used to always score against
    this method's own 0.85 default, no matter what threshold the caller actually
    applies via recommend(). A phrase between 0.60 and 0.85 must show SUGGESTED in
    display() when scored at 0.60, matching what recommend(0.60) itself says -
    two verdicts for the same phrase in the same payload is the bug."""
    s = PhraseScore("asu", n_harmful=54, n_non_harmful=2, n_ambiguous=25, n_criticism=1)
    assert not s.recommend()  # default 0.85: L_H ~= 0.667, fails
    assert s.recommend(0.60)  # at 0.60: passes

    assert s.display()["verdict"] == "REJECTED"
    assert s.display(0.60)["verdict"] == "SUGGESTED"
    assert s.display(0.60)["verdict"] == ("SUGGESTED" if s.recommend(0.60) else "REJECTED")


def test_phrase_scores_ignore_training_weights():
    """AGENTS.md R2. Phrase mining must treat HIGH and MEDIUM as equally resolved.

    If someone starts weighting counts by tier, the lower/upper bounds stop being
    bounds on a missingness problem and become an incoherent mixture. This test
    fails the moment fractional weights leak into scoring.
    """
    def rec(label, tier, weight):
        return ConsensusRecord("c", label, tier, weight)

    all_high = [rec(Label.HARASSMENT_PERSONAL, Tier.HIGH, 1.0) for _ in range(4)]
    half_medium = [
        rec(Label.HARASSMENT_PERSONAL, Tier.HIGH, 1.0),
        rec(Label.HARASSMENT_PERSONAL, Tier.HIGH, 1.0),
        rec(Label.HARASSMENT_PERSONAL, Tier.MEDIUM, 0.5),
        rec(Label.HARASSMENT_PERSONAL, Tier.MEDIUM, 0.5),
    ]
    a, b = score_phrase("x", all_high), score_phrase("x", half_medium)
    assert a.n_harmful == b.n_harmful == 4, "MEDIUM counts as a full resolved observation"
    assert a.p_harmful_lower() == b.p_harmful_lower()


def test_bounds_are_sharp_and_unsmoothed():
    """AGENTS.md R1. A Beta prior would break the identification argument.

    With add-one smoothing "anjir" reports lower = 0.402 when the achievable worst
    case is exactly 0.400, and upper = 0.892 when the achievable best case is 0.900.
    A shrunk interval is not an identified set.
    """
    s = PhraseScore("anjir", n_harmful=40, n_non_harmful=10, n_ambiguous=50, n_criticism=10)
    assert s.p_harmful_lower() == pytest.approx(0.400), "attainable if all ambiguous benign"
    assert s.p_harmful_upper() == pytest.approx(0.900), "attainable if all ambiguous harmful"


def test_band_width_equals_ambiguity_share_exactly():
    for n_h, n_amb in [(40, 50), (310, 90), (118, 4), (0, 7)]:
        s = PhraseScore("x", n_h, 0, n_amb, 0)
        assert s.p_harmful_upper() - s.p_harmful_lower() == pytest.approx(s.ambiguity_share)


def test_resolved_rate_is_none_when_nothing_is_resolved():
    """No prior to invent a number with, so say so."""
    s = PhraseScore("x", n_harmful=0, n_non_harmful=0, n_ambiguous=30, n_criticism=0)
    assert s.p_harmful_resolved() is None
    assert s.display()["harmful among resolved"] == "n/a"
