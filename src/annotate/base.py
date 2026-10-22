"""Annotator protocols. Raw native output is always retained (R4)."""
from __future__ import annotations

from typing import Protocol, Sequence

from ..schemas import Comment, HarmVote, LabelVote


class HarmAnnotator(Protocol):
    """A1 and A2. Broad harm corroboration only - they cannot see criticism."""

    annotator_id: str
    can_vote_criticism: bool = False

    def annotate(self, comments: Sequence[Comment]) -> list[HarmVote]: ...


class LabelAnnotator(Protocol):
    """A3. The only annotator emitting the full 5-label schema."""

    annotator_id: str
    can_vote_criticism: bool = True

    def annotate(self, comments: Sequence[Comment]) -> list[LabelVote]:
        """A3 on a 4080: ~1.6 GPU-hours per 100k comments, batched."""
        ...
