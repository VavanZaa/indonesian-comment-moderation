"""Capability-aware consensus, including the end-to-end path the old test suite missed.

The previous suite fed `Label` values straight into `consense()`, skipping the adapters
entirely, so it never noticed that A1's "not hate" and A2's "no rule fired" were being
turned into positive votes for `benign_other` that outvoted A3 two-to-one. Every
integration test below starts from NATIVE annotator output.
"""
import pytest

from src.annotate.adapters import adapt_a1, adapt_a2, adapt_a3
from src.consensus import consense
from src.schemas import HarmSignal, HarmVote, Label, LabelVote, Tier

NO_HATE = {"HS": 0, "Abusive": 0}
HATE = {"HS": 1, "Abusive": 1}
NO_RULE = {"family": None}
INSULT = {"family": "insult"}


def pipeline(a1_raw, a2_raw, a3_raw):
    """native output -> adapters -> consensus. The path that was never tested."""
    harm = [HarmVote("a1", "c", adapt_a1(a1_raw), a1_raw),
            HarmVote("a2", "c", adapt_a2(a2_raw), a2_raw)]
    return consense(LabelVote("a3", "c", adapt_a3(a3_raw), a3_raw), harm)


# --- the regression that motivated the rewrite (D29) ------------------------

def test_legitimate_criticism_can_win_end_to_end():
    """'videonya jelek, audionya pecah' - all three annotators correct.

    Under the old symmetric 2-of-3 vote this returned MEDIUM benign_other, making
    n_legitimate_criticism identically zero and demo scene 2 unproducible.
    """
    r = pipeline(NO_HATE, NO_RULE, "legitimate_criticism")
    assert r.label is Label.LEGITIMATE_CRITICISM, "the centrepiece of the PoC"
    assert r.tier is Tier.HIGH
    assert r.label is not Label.BENIGN_OTHER


def test_no_harm_evidence_is_an_abstention_not_a_label_vote():
    for raw, adapt in ((NO_HATE, adapt_a1), (NO_RULE, adapt_a2)):
        assert adapt(raw) is HarmSignal.NO_HARM_EVIDENCE
    assert not hasattr(adapt_a1(NO_HATE), "value") or adapt_a1(NO_HATE) != Label.BENIGN_OTHER


# --- the tier table ---------------------------------------------------------

def test_corroborated_harm_is_high():
    r = pipeline(HATE, INSULT, "harassment_personal")
    assert (r.tier, r.label, r.corroboration) == (Tier.HIGH, Label.HARASSMENT_PERSONAL, 2)


def test_partially_corroborated_harm_is_medium():
    r = pipeline(HATE, NO_RULE, "harassment_personal")
    assert (r.tier, r.label, r.corroboration) == (Tier.MEDIUM, Label.HARASSMENT_PERSONAL, 1)


def test_uncorroborated_harm_is_ambiguous():
    """A3 alone asserting harm is the 'LLM over-flags harsh criticism' failure mode.
    Routing it to AMBIGUOUS keeps it out of n_harmful, where it would inflate phrases
    like `jelek` and invert demo scene 2."""
    r = pipeline(NO_HATE, NO_RULE, "harassment_personal")
    assert r.tier is Tier.AMBIGUOUS
    assert r.label is None


def test_flagged_harm_against_a3_criticism_is_ambiguous():
    """Genuine conflict: both harm annotators fire, A3 says criticism."""
    r = pipeline(HATE, INSULT, "legitimate_criticism")
    assert r.tier is Tier.AMBIGUOUS and r.label is None


# --- D39: harm the corroborators cannot perceive ------------------------------

def test_uncorroborated_threat_is_kept_and_sent_to_review_not_dropped():
    """docs/11 §7c. A deliberate threat contains no profanity, so neither corroborator
    fires - and the plain tier table then dropped it to AMBIGUOUS. On real data that
    discarded 13 of 17 direct threats, including six "we have your location" doxxing
    comments. A1/A2 cannot perceive threats, so their silence is not evidence: keep the
    label, hold it at MEDIUM, and flag it for a human."""
    r = pipeline(NO_HATE, NO_RULE, "threat_privacy")
    assert r.label is Label.THREAT_PRIVACY, "must not be discarded to ambiguous"
    assert r.tier is Tier.MEDIUM
    assert r.needs_review is True
    assert r.is_harmful is True, "queued for review, still counted as harmful"


def test_corroborated_threat_is_not_promoted_to_high():
    """The other half of the same finding: threats that DID trip the abuse classifier
    were no more often real (46% vs 43%). A lexical HARM is not evidence about whether
    something is a threat, so corroboration must not raise the tier either."""
    r = pipeline(HATE, INSULT, "threat_privacy")
    assert (r.tier, r.label, r.needs_review) == (Tier.MEDIUM, Label.THREAT_PRIVACY, True)
    assert r.corroboration == 2, "still recorded for audit - it just decides nothing"


def test_d39_leaves_classes_the_corroborators_can_see_untouched():
    """Harassment keeps the original table: A1 IS trained on abuse, so its silence is
    real evidence and uncorroborated harassment still goes to AMBIGUOUS."""
    r = pipeline(NO_HATE, NO_RULE, "harassment_personal")
    assert r.tier is Tier.AMBIGUOUS and r.needs_review is False
    for a3_raw in ("legitimate_criticism", "benign_other", "public_figure_invective"):
        assert pipeline(NO_HATE, NO_RULE, a3_raw).needs_review is False


def test_review_queue_holds_exactly_the_flagged_records(tmp_path):
    import pandas as pd
    from src.consensus import write_review_queue

    def rec(cid, a1, a2, a3):
        harm = [HarmVote("a1", cid, adapt_a1(a1), a1), HarmVote("a2", cid, adapt_a2(a2), a2)]
        return consense(LabelVote("a3", cid, adapt_a3(a3), a3), harm)

    records = [
        rec("t1", NO_HATE, NO_RULE, "threat_privacy"),
        rec("h1", HATE, INSULT, "harassment_personal"),
        rec("b1", NO_HATE, NO_RULE, "benign_other"),
    ]
    ctx = {"t1": {"platform": "youtube", "text_normalized": "kami sudah mendapatkan lokasi anda"}}
    out = tmp_path / "review_queue.parquet"

    n = write_review_queue(out, records, ctx)

    frame = pd.read_parquet(out)
    assert n == 1 and list(frame["comment_id"]) == ["t1"]
    assert frame.loc[0, "text_normalized"] == "kami sudah mendapatkan lokasi anda"
    assert bool(frame.loc[0, "needs_review"]) is True


def test_benign_with_one_harm_flag_is_medium():
    r = pipeline(HATE, NO_RULE, "benign_other")
    assert (r.tier, r.label) == (Tier.MEDIUM, Label.BENIGN_OTHER)


# --- bookkeeping ------------------------------------------------------------

def test_weights_follow_the_tier():
    assert pipeline(HATE, INSULT, "harassment_personal").weight == 1.0
    assert pipeline(HATE, NO_RULE, "harassment_personal").weight == 0.5
    assert pipeline(NO_HATE, NO_RULE, "harassment_personal").weight == 0.0


def test_votes_and_corroboration_are_retained_for_the_ambiguous_reel():
    r = pipeline(HATE, INSULT, "legitimate_criticism")
    assert len(r.votes) == 3, "scene 5 needs to show who said what"
    assert r.corroboration == 2


def test_mismatched_comment_ids_are_rejected():
    with pytest.raises(ValueError):
        consense(LabelVote("a3", "c1", Label.BENIGN_OTHER, None),
                 [HarmVote("a1", "OTHER", HarmSignal.HARM, None)])


def test_missing_harm_votes_are_rejected():
    with pytest.raises(ValueError):
        consense(LabelVote("a3", "c", Label.BENIGN_OTHER, None), [])


def test_a1_subtypes_are_not_mapped_to_poc_labels():
    """D30: gender-directed hate is not the same concept as sexual harm."""
    assert adapt_a1({"HS": 1, "HS_Gender": 1}) is HarmSignal.HARM
    assert not isinstance(adapt_a1({"HS": 1, "HS_Gender": 1}), Label)


def test_a3_off_schema_output_is_rejected():
    with pytest.raises(ValueError):
        adapt_a3("toxic")


def test_mislabelled_clean_criticism_does_not_enter_n_harmful():
    """The docstring in consensus.py used to claim this yielded MEDIUM harassment.

    It does not: with zero corroboration it is AMBIGUOUS, so an A3 error on clean
    criticism never reaches n_harmful and cannot inflate a phrase like `jelek`.
    """
    r = pipeline(NO_HATE, NO_RULE, "harassment_personal")
    assert r.tier is Tier.AMBIGUOUS and r.label is None


def test_the_real_residual_gap_is_criticism_that_trips_the_lexicon():
    """`videonya goblok banget` - criticism of a work, using an insult word.

    A1/A2 corroborate on lexical grounds while A3 supplies the wrong semantic label,
    so this reaches HIGH harassment_personal. Nothing in the ensemble can separate it;
    the screen check and A3's prompt are the only guards (TRAP 1).
    """
    r = pipeline(HATE, INSULT, "harassment_personal")
    assert r.tier is Tier.HIGH and r.corroboration == 2
