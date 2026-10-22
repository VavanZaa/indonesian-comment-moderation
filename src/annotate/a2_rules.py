"""A2 - deterministic regex/lexicon harm matcher.

**A2 EMITS A HarmSignal AND NEVER A Label.** (AGENTS.md TRAP 1 / D29.)

A rule system is blind to criticism: it can say "one of my harm families fired" or
"nothing fired", and the second of those is an *abstention on the semantic label* that
still carries negative evidence on the harm axis. Turning it into a positive vote for
`benign_other` is the D29 bug. `Label` is deliberately not imported in this module and
`tests/test_annotate.py` asserts that it never enters the module namespace.

A2's coverage is LOAD-BEARING (`src/consensus.py`): a blatantly harmful phrase A2 does
not match can never reach the HIGH tier, and an A3 harm assertion with no corroboration
is routed to AMBIGUOUS, where it drops out of `n_harmful` and drags a family's `L_H`
down. So matching is deliberately broad:

* every regex family in `data/raw/lexicons/rule_families.toml`;
* every term in `data/raw/lexicons/abusive_id.csv`, whole-word, attributed to the
  `insult` family (§3 of the build contract lists the abusive lexicon among A2's inputs
  but does not assign it a family - see `abusive_family`);
* both are tried against the text as given *and* against a conservative local fold that
  undoes leet spelling, elongation and separator obfuscation. The fold is a safety net
  for text that reached A2 un-normalized; `src/normalize` remains the real normalizer.

R4: the native output is `{"family": ...}` and the mapping to a `HarmSignal` happens
exclusively in `adapters.adapt_a2`. The raw dict travels with the vote and keeps every
family that fired, not just the reported one.
"""
from __future__ import annotations

import csv
import hashlib
import sys
import tomllib
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import regex as re

from ..schemas import Comment, HarmVote
from .adapters import A2_HARM_FAMILIES, adapt_a2

DEFAULT_LEXICON_DIR = Path("data/raw/lexicons")
RULES_FILENAME = "rule_families.toml"
ABUSIVE_FILENAME = "abusive_id.csv"

#: Reporting order when several families fire on one comment: the most specific and most
#: severe first, so `raw["family"]` is stable no matter how the TOML is ordered. Every
#: name here must be in `A2_HARM_FAMILIES` and every family there must appear here -
#: `tests/test_annotate.py` pins both directions, so a new family cannot be silently
#: dropped from A2's coverage.
FAMILY_PRIORITY: tuple[str, ...] = (
    "threat", "doxx", "death_wish", "selfharm_encouragement", "sexual", "slur", "insult",
)

_FLAGS = re.IGNORECASE | re.UNICODE

_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b",
                       "@": "a", "$": "s"})
_REPEAT = re.compile(r"(.)\1{2,}")
_SEPARATED = re.compile(r"\b(?:\w[\W_]{1,2}){2,}\w\b")
_NON_WORD = re.compile(r"[^\w\s]+", _FLAGS)
_SPACES = re.compile(r"\s+")


def fold(text: str) -> str:
    """Conservative de-obfuscation used as a *second* matching pass.

    Not a replacement for `src.normalize.normalize` - it exists so A2 still fires when
    it is handed text that never went through the normalizer (a unit test, an ad-hoc
    call, a corpus written before the normalize stage existed).
    """
    low = text.lower().translate(_LEET)
    low = _SEPARATED.sub(lambda m: re.sub(r"[\W_]+", "", m.group(0)), low)
    low = _NON_WORD.sub(" ", low)
    low = _REPEAT.sub(r"\1", low)
    return _SPACES.sub(" ", low).strip()


class RuleLoadError(RuntimeError):
    """Rule resources are missing or malformed. Never degrade silently: A2 with no
    patterns is A2 that can never corroborate, which quietly disables the HIGH tier."""


class A2Rules:
    """Deterministic rule/lexicon annotator. Implements `base.HarmAnnotator`.

    Runs today with no model download: it needs only the two lexicon files, which are
    part of the frozen surface (R3) and are hashed into `models.lock.json`.
    """

    can_vote_criticism = False           # TRAP 1. Structural, not a configuration choice.

    def __init__(
        self,
        lexicon_dir: str | Path = DEFAULT_LEXICON_DIR,
        *,
        annotator_id: str = "a2",
        rules_path: str | Path | None = None,
        abusive_path: str | Path | None = None,
        abusive_family: str = "insult",
        use_fold: bool = True,
        require_abusive_lexicon: bool = False,
    ) -> None:
        if abusive_family not in A2_HARM_FAMILIES:
            raise ValueError(f"abusive_family {abusive_family!r} is not in A2_HARM_FAMILIES")
        self.annotator_id = annotator_id
        self.lexicon_dir = Path(lexicon_dir)
        self.rules_path = Path(rules_path) if rules_path else self.lexicon_dir / RULES_FILENAME
        self.abusive_path = (
            Path(abusive_path) if abusive_path else self.lexicon_dir / ABUSIVE_FILENAME
        )
        self.abusive_family = abusive_family
        self.use_fold = use_fold
        self.require_abusive_lexicon = require_abusive_lexicon

        self._patterns: dict[str, list[tuple[str, Any]]] = {f: [] for f in FAMILY_PRIORITY}
        self._sources: list[Path] = []
        self._load_rules()
        self._load_abusive()
        if not any(self._patterns.values()):
            raise RuleLoadError(
                f"A2 loaded zero patterns from {self.lexicon_dir} - refusing to run. "
                "A2 with no coverage silently disables the HIGH consensus tier."
            )

    # -- resources -------------------------------------------------------------

    def _load_rules(self) -> None:
        if not self.rules_path.exists():
            raise RuleLoadError(
                f"{self.rules_path} not found. A2 needs the rule families from §2 of "
                "docs/05-build-contract.md, e.g.\n\n"
                '    [insult]\n    patterns = ["\\\\bgoblok\\\\b", "\\\\btolol\\\\b"]\n'
            )
        data = tomllib.loads(self.rules_path.read_text(encoding="utf-8"))
        for family, table in data.items():
            if not isinstance(table, dict) or not ({"patterns", "words"} & set(table)):
                print(f"a2: ignoring non-family table [{family}] in {self.rules_path}",
                      file=sys.stderr)
                continue
            if family not in A2_HARM_FAMILIES:
                raise RuleLoadError(
                    f"[{family}] in {self.rules_path} is not one of A2_HARM_FAMILIES "
                    f"({', '.join(sorted(A2_HARM_FAMILIES))}). Rename it rather than "
                    "adding a family the adapter cannot read as harm evidence."
                )
            for pattern in table.get("patterns", []) or []:
                self._add(family, str(pattern), literal=False)
            # `words` is an accepted convenience: literal terms, wrapped in word
            # boundaries for you. Broad coverage should not require regex fluency.
            for word in table.get("words", []) or []:
                self._add(family, str(word), literal=True)
        self._sources.append(self.rules_path)

    def _load_abusive(self) -> None:
        if not self.abusive_path.exists():
            message = (
                f"a2: {self.abusive_path} not found - running on rule_families.toml "
                "alone. A2's coverage is load-bearing for the HIGH tier (consensus.py); "
                "expect fewer HIGH records."
            )
            if self.require_abusive_lexicon:
                raise RuleLoadError(message)
            print(message, file=sys.stderr)
            return
        with self.abusive_path.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.reader(fh))
        for i, row in enumerate(rows):
            if not row:
                continue
            word = row[0].strip()
            if not word or word.startswith("#"):
                continue
            if i == 0 and word.lower() == "word":          # header, per the contract
                continue
            self._add(self.abusive_family, word, literal=True)
        self._sources.append(self.abusive_path)

    def _add(self, family: str, raw_pattern: str, *, literal: bool) -> None:
        term = raw_pattern.strip()
        if not term:
            return
        expr = rf"\b{re.escape(term)}\b" if literal else term
        try:
            compiled = re.compile(expr, _FLAGS)
        except re.error as exc:
            raise RuleLoadError(f"bad pattern for [{family}]: {raw_pattern!r} ({exc})") from exc
        self._patterns[family].append((term, compiled))

    # -- matching --------------------------------------------------------------

    def match(self, text: str) -> dict[str, Any]:
        """Native A2 output: `{"family": "insult", ...}` or `{"family": None, ...}`.

        Extra keys are retained for R4 (every family that fired, and the terms that
        fired them). `adapt_a2` reads only `family`, so they are inert downstream.
        """
        candidates = [text]
        if self.use_fold:
            folded = fold(text)
            if folded and folded != text:
                candidates.append(folded)

        fired: list[str] = []
        matched: list[str] = []
        for family in FAMILY_PRIORITY:
            for _term, pattern in self._patterns[family]:
                hit = next((m for c in candidates if (m := pattern.search(c))), None)
                if hit is not None:
                    if family not in fired:
                        fired.append(family)
                    # The actual matched substring, not the pattern source - a regex
                    # like `\b(anjing|anjir)\b` should report which alternative fired.
                    matched_text = hit.group(0)
                    if len(matched) < 8 and matched_text not in matched:
                        matched.append(matched_text)
                    break
        return {
            "family": fired[0] if fired else None,
            "families": fired,
            "matched": matched,
        }

    def annotate(self, comments: Sequence[Comment]) -> list[HarmVote]:
        votes: list[HarmVote] = []
        for comment in comments:
            raw = self.match(comment.text_normalized or comment.text_original)
            votes.append(HarmVote(
                annotator_id=self.annotator_id,
                comment_id=comment.comment_id,
                signal=adapt_a2(raw),          # R4: the only native -> signal path
                raw=raw,
                # No confidence. A2 is a deterministic matcher; a probability here would
                # be a fabricated number (R5).
                confidence=None,
            ))
        return votes

    # -- identity for models.lock.json (R3/D31) --------------------------------

    def n_patterns(self) -> dict[str, int]:
        return {f: len(p) for f, p in self._patterns.items()}

    def resource_hashes(self) -> dict[str, str]:
        return {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            for p in sorted(self._sources, key=lambda q: q.name)
        }

    def rules_sha256(self) -> str:
        """One hash over every rule resource A2 actually loaded."""
        h = hashlib.sha256()
        for p in sorted(self._sources, key=lambda q: q.name):
            h.update(p.name.encode("utf-8"))
            h.update(p.read_bytes())
        return h.hexdigest()[:16]

    def model_identity(self) -> dict[str, Any]:
        return {
            "lexicons": ["kamus_alay", "abusive_id"],
            "rules_sha256": self.rules_sha256(),
            "resource_sha256": self.resource_hashes(),
            "families": {f: n for f, n in self.n_patterns().items() if n},
            "backend": "rules",
        }


def family_counts(votes: Iterable[HarmVote]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for v in votes:
        key = (v.raw or {}).get("family") or "(none)"
        counts[key] = counts.get(key, 0) + 1
    return counts
