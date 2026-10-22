"""Phrase scoring. AGENTS.md R1 lives here - read the class docstring before editing.

The harmful rate of a phrase is PARTIALLY IDENTIFIED, not point-identified, because
ambiguous matches have no trusted class. Report the bounds; decide on the lower one.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Sequence

from ..schemas import COLLATERAL, ConsensusRecord, Label


@dataclass(slots=True)
class PhraseScore:
    """Counts for one candidate phrase over all comments it matches.

    The shipped artefact is a YouTube blocked-words list, which hides EVERY match --
    ambiguous ones included. So `n_all` is the operational denominator: you do not get
    to exclude comments from a consequence they will receive.

    But n_harmful/n_all silently assumes every ambiguous match is non-harmful, and
    n_harmful/n_resolved silently assumes ambiguous matches behave like resolved ones.
    Both are assumptions. The honest object is an interval (Manski bounds):

        lower    = n_harmful / n_all                    all ambiguous non-harmful
        upper    = (n_harmful + n_ambiguous) / n_all    all ambiguous harmful
        resolved = n_harmful / n_resolved               resolved labels only

    with the exact identity

        upper - lower == ambiguity_share

    so `ambiguity_share` is not a side metric - it is the width of the identified set,
    which is what makes "confidence: low" a number rather than a vibe.

    NO SMOOTHING. These are sharp bounds on the OBSERVED proportion, and a Beta prior
    breaks that: with add-one smoothing "anjir" reports a lower bound of 0.402 when the
    achievable worst case is exactly 0.400, and an upper of 0.892 when the achievable
    best case is 0.900. A shrunk interval is not an identified set. Small-sample noise
    is handled by `min_matches`, not by a prior, and generalisation is tested by the
    frozen Channel E run, not assumed by one.

    Worked example, "anjir": 40 harmful / 10 criticism / 50 ambiguous
        resolved 0.800  <- looks recommendable
        lower    0.400  <- is not
        band     [0.400, 0.900], width 0.500 == ambiguity share
    """

    phrase: str
    n_harmful: int
    n_non_harmful: int
    n_ambiguous: int
    n_criticism: int

    # -- denominators --------------------------------------------------------
    @property
    def n_resolved(self) -> int:
        return self.n_harmful + self.n_non_harmful

    @property
    def n_all(self) -> int:
        """Operational denominator: every comment the filter would hide."""
        return self.n_resolved + self.n_ambiguous

    # -- the interval (sharp, unsmoothed) ------------------------------------
    def p_harmful_lower(self) -> float:
        """Worst case: every ambiguous match is non-harmful. Decide on this."""
        return self.n_harmful / self.n_all if self.n_all else 0.0

    def p_harmful_upper(self) -> float:
        """Best case: every ambiguous match is harmful."""
        return (self.n_harmful + self.n_ambiguous) / self.n_all if self.n_all else 1.0

    def p_harmful_resolved(self) -> float | None:
        """Harmful rate among resolved labels only. Report it, never decide on it.

        Returns None when nothing is resolved - the rate is genuinely undefined then,
        and a prior would invent a number rather than admit that.

        This is the estimator you would use if ambiguity were missing-at-random, but
        do not label it "MAR" on screen or in a UI - MAR names an assumption about the
        missingness mechanism, not a rate. Call it "harmful among resolved".
        """
        return self.n_harmful / self.n_resolved if self.n_resolved else None

    @property
    def ambiguity_share(self) -> float:
        """Equals p_harmful_upper - p_harmful_lower in the unsmoothed limit."""
        return self.n_ambiguous / self.n_all if self.n_all else 0.0

    @property
    def criticism_share(self) -> float:
        """C(f) = n_collateral / n_all, where collateral is `schemas.COLLATERAL`.

        Denominator is ALL matches, not resolved ones: this measures collateral
        filtering, and a filter hides ambiguous matches too. Drives demo scene 2.

        `n_criticism` counts every label in `COLLATERAL`, not only
        `legitimate_criticism` (D38, 2026-09-10). The field keeps its name because it
        is the C term of R0 and is written into artefacts that already carry it; what
        widened is the definition of what counts as collateral. See
        `schemas.COLLATERAL` for why `public_figure_invective` belongs in it.
        """
        return self.n_criticism / self.n_all if self.n_all else 0.0

    # -- decision ------------------------------------------------------------
    def recommend(
        self,
        threshold: float = 0.85,
        max_criticism: float = 0.10,
        min_matches: int = 25,
    ) -> bool:
        """suggest(f) <=> L_H(f) >= 0.85  AND  C(f) <= 0.10  AND  n_all >= n_min

        Decide on the LOWER bound: you are deciding whether to hide people's comments,
        so be wrong on the conservative side.

        There is deliberately NO separate ambiguity gate. It would be unreachable:

            L_H = n_H / n_all >= 0.85  =>  ambiguity_share <= 0.15

        directly, since n_ambiguous <= n_all - n_harmful. Deciding on the lower bound
        *is* the ambiguity gate. `ambiguity_share` remains a display metric - it is the
        width of the identified set - but it is not a separate constraint.

        `min_matches` is load-bearing and is what replaces smoothing: unsmoothed, a
        phrase with 5 matches all harmful scores a perfect 1.000, and only the support
        requirement stops it being suggested on pure noise.
        """
        return (
            self.n_all >= min_matches
            and self.p_harmful_lower() >= threshold
            and self.criticism_share <= max_criticism
        )

    def display(
        self,
        threshold: float = 0.85,
        max_criticism: float = 0.10,
        min_matches: int = 25,
    ) -> dict[str, str]:
        """What goes on screen. Never show a bare point estimate.

        Bug fixed 2026-09-09 (D37): `verdict` used to call `self.recommend()` with no
        arguments, always scoring against this method's own 0.85/0.10/25 defaults
        regardless of the thresholds actually configured for the run. The moment R0's
        threshold moved to 0.60, that produced a live contradiction on the exact
        surface this docstring says is "what goes on screen": `recommendations.json`'s
        top-level `verdict` correctly said SUGGESTED at 0.60 while this method's own
        embedded `display.verdict` still said REJECTED, computed at the old 0.85. Any
        caller applying a non-default threshold MUST pass it here too - callers that
        never override it keep this method's original behaviour exactly.
        """
        resolved = self.p_harmful_resolved()
        return {
            "phrase": self.phrase,
            "matches": f"{self.n_all}",
            "harmful among resolved": "n/a" if resolved is None else f"{resolved:.0%}",
            "legitimate criticism": f"{self.criticism_share:.0%}",
            "ambiguity share": f"{self.ambiguity_share:.0%}",
            "identified range": f"{self.p_harmful_lower():.0%} - {self.p_harmful_upper():.0%}",
            "verdict": "SUGGESTED" if self.recommend(threshold, max_criticism, min_matches) else "REJECTED",
        }


def score_phrase(phrase: str, matches: Sequence[ConsensusRecord]) -> PhraseScore:
    n_harm = n_non = n_amb = n_crit = 0
    for rec in matches:
        harmful = rec.is_harmful
        if harmful is None:
            n_amb += 1
        elif harmful:
            n_harm += 1
        else:
            n_non += 1
            # COLLATERAL, not LEGITIMATE_CRITICISM alone (D38): a phrase firing mostly
            # on crude political speech must be gated by R0's C term exactly as one
            # firing on ordinary criticism is. Testing the single label here is what
            # let `memang anjing` reach SUGGESTED at C=0.000 while half its matches
            # were `pemerintah memang anjing`.
            if rec.label in COLLATERAL:
                n_crit += 1
    return PhraseScore(phrase, n_harm, n_non, n_amb, n_crit)


# =============================================================================
# CLI (build contract §5). `PhraseScore`/`score_phrase` above are untouched; all
# new logic lives in `candidates.py` - this is the thin wrapper the contract asks
# for so `python -m src.phrases.mine` works.
# =============================================================================

def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    from .candidates import load_config, run

    parser = argparse.ArgumentParser(prog="python -m src.phrases.mine")
    parser.add_argument("--config", default="config/local.yaml")
    args = parser.parse_args(argv)
    run(load_config(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
