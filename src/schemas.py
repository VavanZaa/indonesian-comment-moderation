"""Label schema and record types. Single source of truth for what 'harmful' means."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Label(str, Enum):
    HARASSMENT_PERSONAL = "harassment_personal"
    SEXUAL_HARM = "sexual_harm"
    THREAT_PRIVACY = "threat_privacy"
    LEGITIMATE_CRITICISM = "legitimate_criticism"
    #: D38. Crude/profane hostility aimed at a PUBLIC figure in their public role, or
    #: at an institution (government, police, courts, a company) - as opposed to a
    #: private individual, which is `harassment_personal`. Added 2026-09-10 after the
    #: first real screen check found that 50%+ of the matches behind the top suggested
    #: filters were comments like `prabowo kntl` and `pemerintah memang anjing`:
    #: angry political speech, not interpersonal harassment. The 5-label schema had no
    #: way to say that, so A3 defaulted it into `harassment_personal` and the system
    #: began recommending that those words be blocked - which would have shipped a
    #: political-speech filter wearing a moderation label. Structurally the same gap as
    #: TRAP 1 (docs/09): a missing class does not stay missing, it gets absorbed by the
    #: nearest available one and quietly corrupts every rate computed from it.
    PUBLIC_FIGURE_INVECTIVE = "public_figure_invective"
    BENIGN_OTHER = "benign_other"


#: AGENTS.md §4. P(harmful | phrase) is undefined without this.
HARMFUL: frozenset[Label] = frozenset({
    Label.HARASSMENT_PERSONAL,
    Label.SEXUAL_HARM,
    Label.THREAT_PRIVACY,
})
NON_HARMFUL: frozenset[Label] = frozenset(Label) - HARMFUL

#: The C term in R0. NOT "criticism" narrowly - the set of non-harmful labels whose
#: filtering is COLLATERAL DAMAGE, i.e. speech a creator's blocked-word list should not
#: be destroying. `public_figure_invective` belongs here for the same reason
#: `legitimate_criticism` does: it is negative speech aimed at power, and hiding it is
#: the failure this project exists to demonstrate, not a success.
#:
#: Anything added to this set immediately tightens R0's `C(F) <= 0.10` gate. That is
#: deliberate: a label that is non-harmful but NOT counted here would let a phrase
#: firing overwhelmingly on it sail past the gate, which is exactly the hole D38 closes.
COLLATERAL: frozenset[Label] = frozenset({
    Label.LEGITIMATE_CRITICISM,
    Label.PUBLIC_FIGURE_INVECTIVE,
})

#: Harm labels that A1 and A2 CANNOT perceive, so their votes carry no signal about them
#: in either direction (D39). Consensus neither discounts these for lack of corroboration
#: nor promotes them for having it: A3 is the only capable judge, so they are held at
#: MEDIUM and flagged `needs_review` for a human.
#:
#: This is TRAP 1 / D29 applied to harm. D29 made consensus capability-aware for
#: `legitimate_criticism` - A1/A2 cannot vote it, so their silence must not count
#: against it - and never extended the idea to harm classes they are equally blind to.
#: Measured on all 130 real threat_privacy labels (docs/11 §7c): A1 is trained on
#: Ibrohim & Budi, which has no threat or doxxing class, and A2's threat/doxx lexicon
#: families fired 4 and 8 times in 223,096 comments. Corroboration kept 61 and dropped 69,
#: the two groups were EQUALLY likely to hold a real threat (46% vs 43%), and the drop
#: discarded 13 of 17 DIRECT threats - including six "we have your location" doxxing
#: comments - because a deliberate threat is calm and specific and contains no profanity
#: for an abuse classifier to fire on.
UNCORROBORABLE: frozenset[Label] = frozenset({Label.THREAT_PRIVACY})


class Tier(str, Enum):
    """Consensus is capability-aware, not a majority vote, so these are NOT vote counts.

    A1/A2 only corroborate harm; A3 supplies the label. Criticism reaches HIGH with both
    corroborators silent, because silence is agreement that no harm was found.
    """
    #: A3's label fully consistent with the corroborators
    #: (harm + both corroborate, or non-harm + neither flags)
    HIGH = "high"
    #: partial support: one corroborator agrees, one does not. Also where every
    #: `UNCORROBORABLE` label is held: A3 alone asserts it, no capable corroborator exists,
    #: so it is kept and flagged `needs_review` - neither trusted as HIGH nor dropped (D39)
    MEDIUM = "medium"
    #: strong contradiction, or a harmful assertion by A3 that nothing corroborates
    AMBIGUOUS = "ambiguous"


#: TRAINING ONLY. These weights apply to the optional student classifier (PoC-B) and
#: to nothing else. **Phrase mining must ignore them**: it treats HIGH and MEDIUM as
#: equally resolved observations and AMBIGUOUS as unresolved, because the lower/upper
#: bounds are about *missingness*, and a MEDIUM label is uncertain, not missing.
#: Mixing fractional weights into the bounds makes them uninterpretable.
#: Enforced by tests/test_phrases.py::test_phrase_scores_ignore_training_weights.
TRAINING_WEIGHT: dict[Tier, float] = {Tier.HIGH: 1.0, Tier.MEDIUM: 0.5, Tier.AMBIGUOUS: 0.0}

#: Tiers that count as RESOLVED for phrase mining. Equal footing, no weighting.
RESOLVED_TIERS: frozenset[Tier] = frozenset({Tier.HIGH, Tier.MEDIUM})
TRAIN_TIERS = RESOLVED_TIERS  # same set today; kept separate because the reasons differ


@dataclass(slots=True)
class Comment:
    comment_id: str
    channel_id: str
    video_id: str
    author_hash: str          # R8: hashed at ingest. Raw handles never enter the pipeline.
    published_at: str
    text_original: str        # R7: never overwritten
    text_normalized: str = ""


class HarmSignal(str, Enum):
    """What A1 and A2 are actually capable of saying.

    NO_HARM_EVIDENCE is a SEMANTIC-LABEL ABSTENTION that still carries negative
    corroborating evidence for harm. A hate-speech classifier returning "not hate" and
    a rule system firing nothing are both saying "I found no harm" - silence about
    *which* non-harmful class this is, but a real signal on the harm axis. It is what
    lets legitimate_criticism reach HIGH when both corroborators are silent.

    Treating that silence as a positive vote for benign_other is what made
    legitimate_criticism unwinnable in the first version of this pipeline (D29).
    """
    HARM = "harm"
    NO_HARM_EVIDENCE = "no_harm_evidence"


@dataclass(slots=True)
class HarmVote:
    """A1 and A2. Broad harm corroboration only - no semantic subtype.

    A1's HS_* subtypes are deliberately NOT mapped to PoC labels: gender-directed
    hate and sexual harm are different concepts, and pretending otherwise would put
    a wrong fine label into the record. The native output is retained in `raw`.
    """
    annotator_id: str
    comment_id: str
    signal: HarmSignal
    raw: Any
    confidence: float | None = None


@dataclass(slots=True)
class LabelVote:
    """A3. The only annotator that emits the full 5-label schema."""
    annotator_id: str
    comment_id: str
    label: Label
    raw: Any
    confidence: float | None = None


@dataclass(slots=True)
class ConsensusRecord:
    comment_id: str
    label: Label | None       # None when tier is AMBIGUOUS
    tier: Tier
    weight: float           # TRAINING ONLY - see TRAINING_WEIGHT. Phrase mining ignores it.
    votes: dict[str, str] = field(default_factory=dict)   # annotator_id -> label or signal
    corroboration: int = 0  # how many of A1/A2 signalled HARM
    #: D39. True when the label is one the corroborators cannot perceive
    #: (`UNCORROBORABLE`), so A3 is the only capable judge and a human must confirm it.
    #: The record still counts as harmful - this routes it, it does not demote it.
    needs_review: bool = False

    @property
    def is_harmful(self) -> bool | None:
        """None means undecided - callers MUST NOT coerce this to False.
        Phrase mining counts these in n_all and reports them as the width of the
        identified set (R1)."""
        return None if self.label is None else self.label in HARMFUL
