"""AGENTS.md R4 - each annotator's native output mapped to what it can actually say.

TRAP 1 (AGENTS.md §3): Ibrohim & Budi's 12 labels are HS, Abusive and ten HS_* subtypes.
There is NO class for non-hateful negative content. A1 therefore cannot vote
legitimate_criticism, and neither can the rule system.

The first version of this file mapped A1 and A2 into the 5-label schema anyway, turning
"no harm found" into a positive vote for benign_other. Under a symmetric 2-of-3 vote
that made legitimate_criticism unwinnable and C(f) identically zero. A1 and A2 now emit
a HarmSignal instead, and NO_HARM_EVIDENCE is an abstention.

A1's HS_* subtypes are deliberately NOT mapped to PoC labels. HS_Gender is
gender-directed hate, which is not the same concept as sexual_harm; asserting the
equivalence would write a wrong fine label into the record. A3 supplies fine labels.
"""
from __future__ import annotations

from typing import Any

from ..schemas import HarmSignal, Label

#: A1 native columns that constitute harm evidence. Presence of any => HARM.
A1_HARM_KEYS: tuple[str, ...] = (
    "HS", "Abusive", "HS_Individual", "HS_Group", "HS_Religion",
    "HS_Race", "HS_Physical", "HS_Gender", "HS_Other",
)

#: A2 rule families that constitute harm evidence.
#: Keep these BROAD - consensus.py makes A2 coverage load-bearing for the HIGH tier.
A2_HARM_FAMILIES: frozenset[str] = frozenset({
    "insult", "slur", "sexual", "threat", "doxx", "death_wish", "selfharm_encouragement",
})


def adapt_a1(raw: dict[str, int]) -> HarmSignal:
    """raw: {'HS': 1, 'Abusive': 0, 'HS_Individual': 1, ...} binary columns."""
    return HarmSignal.HARM if any(raw.get(k) for k in A1_HARM_KEYS) else HarmSignal.NO_HARM_EVIDENCE


def adapt_a2(raw: dict[str, Any]) -> HarmSignal:
    """raw: {'family': 'insult'} or {'family': None} when nothing fired."""
    return HarmSignal.HARM if raw.get("family") in A2_HARM_FAMILIES else HarmSignal.NO_HARM_EVIDENCE


def adapt_a3(raw: str) -> Label:
    """A3 emits the PoC label directly. Validate rather than trust."""
    token = (raw or "").strip().lower()
    try:
        return Label(token)
    except ValueError as exc:
        raise ValueError(f"A3 emitted an off-schema label: {raw!r}") from exc
