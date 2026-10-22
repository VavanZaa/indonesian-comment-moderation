"""Phrase families. The FAMILY is the recommendation unit (D33), and the union rule is
what makes that sound.

A family's support is the size of the UNION of its variants' match sets, never the
sum. A comment reading "goblok banget goblokkk" matches two variants of GOBLOK_FAMILY
and must count ONCE:

    M_F = union of M_f for f in F        n_F = |M_F|
    NOT  n_F = sum of |M_f|

Summing inflates support for exactly the families with the most spelling variants,
which are the ones the demo highlights. It would also let a family clear
`n_all >= 25` on a handful of comments.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from ..schemas import ConsensusRecord
from .mine import PhraseScore, score_phrase


@dataclass(slots=True)
class PhraseFamily:
    name: str
    variants: list[str] = field(default_factory=list)
    #: comment ids matched by ANY variant, deduplicated
    member_ids: set[str] = field(default_factory=set)

    @property
    def n_all(self) -> int:
        return len(self.member_ids)

    def score(self, records_by_id: Mapping[str, ConsensusRecord]) -> PhraseScore:
        """Family-level statistics over the DEDUPLICATED union (D33).

        The family is the recommendation unit, so every quantity in the decision rule
        is computed over M_F, not per variant:

            n_all(F) = |M_F|      n_H(F), n_C(F), n_A(F) counted once per comment

        This is what makes families load-bearing rather than decorative. A variant with
        8 matches is below minimum support on its own but contributes to a family that
        clears it. It is also the conservative reading: if you are going to block every
        variant, you should be judged on every variant's matches.
        """
        matches = [records_by_id[cid] for cid in sorted(self.member_ids) if cid in records_by_id]
        return score_phrase(self.name, matches)


def build_family(
    name: str,
    variants: Sequence[str],
    matches_by_variant: Mapping[str, Sequence[str]],
) -> PhraseFamily:
    """matches_by_variant: variant -> comment ids it matched.

    >>> f = build_family("GOBLOK", ["goblok", "goblokkk"],
    ...                  {"goblok": ["c1", "c2"], "goblokkk": ["c1", "c3"]})
    >>> f.n_all            # c1 counted once, not twice
    3
    """
    members: set[str] = set()
    for v in variants:
        members.update(matches_by_variant.get(v, ()))
    return PhraseFamily(name=name, variants=list(variants), member_ids=members)


def score_family(
    name: str,
    variants: Sequence[str],
    matches_by_variant: Mapping[str, Sequence[str]],
    records_by_id: Mapping[str, ConsensusRecord],
) -> PhraseScore:
    """Build and score in one step. A singleton phrase is a family of size one, so the
    recommendation unit is uniform across the pipeline."""
    return build_family(name, variants, matches_by_variant).score(records_by_id)


def inflation_factor(
    variants: Sequence[str], matches_by_variant: Mapping[str, Sequence[str]]
) -> float:
    """sum(|M_f|) / |union M_f| - how much naive summing would have overstated support."""
    total = sum(len(matches_by_variant.get(v, ())) for v in variants)
    union = len({c for v in variants for c in matches_by_variant.get(v, ())})
    return total / union if union else 1.0


# =============================================================================
# CLI (build contract §6). `build_family`/`score_family`/`inflation_factor` above
# are untouched; all new logic lives in `grouping.py` - this is the thin wrapper
# the contract asks for so `python -m src.phrases.families` works.
# =============================================================================

def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    from .candidates import load_config
    from .grouping import run

    parser = argparse.ArgumentParser(prog="python -m src.phrases.families")
    parser.add_argument("--config", default="config/local.yaml")
    args = parser.parse_args(argv)
    run(load_config(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
