"""The union rule. Summing per-variant match counts inflates exactly the families
the demo highlights, because they are the ones with the most spelling variants."""
import pytest

from src.phrases.families import build_family, inflation_factor

MATCHES = {
    "goblok":   ["c1", "c2", "c3", "c4"],
    "goblokkk": ["c1", "c5"],            # c1 also matched "goblok"
    "g0blok":   ["c2", "c6"],            # c2 also matched "goblok"
    "g*blok":   ["c1"],                  # c1 again
}
VARIANTS = list(MATCHES)


def test_family_support_is_the_union_not_the_sum():
    f = build_family("GOBLOK", VARIANTS, MATCHES)
    assert f.n_all == 6, "c1..c6 distinct"
    assert f.n_all != sum(len(v) for v in MATCHES.values()), "sum would be 9"


def test_a_single_comment_with_two_variants_counts_once():
    f = build_family("GOBLOK", ["goblok", "goblokkk"], {"goblok": ["c1"], "goblokkk": ["c1"]})
    assert f.n_all == 1


def test_inflation_factor_reports_what_summing_would_have_cost():
    assert inflation_factor(VARIANTS, MATCHES) == pytest.approx(9 / 6)


def test_empty_family_is_zero_not_an_error():
    assert build_family("EMPTY", [], {}).n_all == 0
    assert inflation_factor([], {}) == 1.0


# --- family-level scoring: the family is the recommendation unit (D33) -------

from src.phrases.families import score_family
from src.schemas import ConsensusRecord, Label, Tier


def _rec(cid, label):
    return ConsensusRecord(cid, label, Tier.HIGH, 1.0)


def test_family_score_does_not_double_count_a_multi_variant_comment():
    """`goblok banget goblokkk` matches two variants and must contribute once.

    Scoring is where this actually matters - the dedup test above only checks n_all
    on the family object, not the numbers that drive the decision rule.
    """
    matches = {"goblok": ["c1", "c2"], "goblokkk": ["c1"]}
    records = {"c1": _rec("c1", Label.HARASSMENT_PERSONAL),
               "c2": _rec("c2", Label.HARASSMENT_PERSONAL)}
    s = score_family("GOBLOK", list(matches), matches, records)
    assert s.n_all == 2, "3 variant-matches, 2 distinct comments"
    assert s.n_harmful == 2


def test_family_membership_rescues_a_low_support_variant():
    """A variant with 8 matches is below n_min=25 alone and clears it in a family.
    This is the point of families; without family scoring they are decoration."""
    rare = {f"r{i}": _rec(f"r{i}", Label.HARASSMENT_PERSONAL) for i in range(8)}
    common = {f"k{i}": _rec(f"k{i}", Label.HARASSMENT_PERSONAL) for i in range(40)}
    records = {**rare, **common}
    matches = {"g*blok": list(rare), "goblok": list(common)}

    alone = score_family("g*blok", ["g*blok"], matches, records)
    assert not alone.recommend(), "8 matches is below minimum support"

    family = score_family("GOBLOK", ["g*blok", "goblok"], matches, records)
    assert family.n_all == 48 and family.recommend()


def test_family_score_is_conservative_when_variants_disagree():
    """Blocking every variant means being judged on every variant's matches."""
    good = {f"h{i}": _rec(f"h{i}", Label.HARASSMENT_PERSONAL) for i in range(40)}
    bad = {f"c{i}": _rec(f"c{i}", Label.LEGITIMATE_CRITICISM) for i in range(20)}
    records = {**good, **bad}
    matches = {"clean": list(good), "dirty": list(bad)}

    assert score_family("clean", ["clean"], matches, records).recommend()
    family = score_family("FAM", ["clean", "dirty"], matches, records)
    assert family.criticism_share > 0.10
    assert not family.recommend(), "the dilution must be visible at family level"
