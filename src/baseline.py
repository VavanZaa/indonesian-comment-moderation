"""The static-dictionary baseline. Four PARALLEL arms (AGENTS.md R6, build contract §7).

    baseline_0   original text  + static list      what a creator has today
    baseline_1   normalized     + static list      isolates what normalization buys
    ours         normalized     + learned filters
    hybrid       normalized     + learned filters, gated on each comment's own label

    baseline_1 - baseline_0 = the normalizer effect
    ours       - baseline_1 = the learning effect
    hybrid     - ours       = the confirmation effect (2026-09-09, D37 follow-up)

Without `baseline_1`, catching `g0blok` proves only that you have a normalizer. The
first three arms exist so the improvement decomposes into those two terms and neither
can be claimed for the other.

`hybrid` answers a different question: a word/phrase match, even a learned one, still
hides some comments its own per-comment label disagrees with (measured directly on the
real corpus: 33% for `asu`, 19% for `kntl` - see docs/10). `hybrid` treats a match as a
CANDIDATE only, and hides a comment iff its own label also says harmful. See the note
above `ARMS` below for exactly what this arm does and does not prove - short version:
identical recall to `ours`, zero collateral, by construction - not a new classifier.

D18 - THE ARMS ARE PARALLEL, NOT A STAGE OF THE PIPELINE.
There is no code path from any baseline output into scoring. This module READS the
learned artefacts (`recommendations.json`) to evaluate the `ours` arm; nothing here is
ever read back by phrase mining, family scoring or the recommendation rule. The
pipeline diagram in `docs/04-architecture-mapping.md` shows both branches meeting at
COMPARE and nowhere else - implementing the diagram literally, with the baseline as an
input to the recommender, would make the comparison circular and meaningless. If you
are about to import `baseline_comparison.json` from anything under `src/phrases/`,
stop: that is the mistake D18 exists to prevent.

R5 - every figure emitted here is "according to our automated annotation system".
R8 - no author handle enters this module: the corpus loader takes an explicit column
allowlist, so there is nothing to leak into the comparison artefact.
R3 - dev rows only. `split == "heldout"` is Channel E and only `src/report.py heldout`
may read it.
R9 - comments now come from YouTube, Instagram and X about the same public figures, and
reach is uneven across them. Every arm's numbers are broken down BY PLATFORM as well as
pooled, so a family that only "works" on one app is visible rather than averaged away.
`platform` is read from the corpus when the ingest lane has migrated to the multi-platform
schema (build contract SS1); until then every row falls into a single `UNKNOWN_PLATFORM`
bucket and the artefact says so explicitly rather than inventing a mix.

    python -m src.baseline --config config/local.yaml
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Iterable, Mapping, Sequence

from .schemas import HARMFUL, Label

#: R5. Repeated inside the artefact so a figure cannot be lifted out of it unlabelled.
ATTRIBUTION = "according to our automated annotation system"

DEFAULT_CONFIG = Path("config/local.yaml")
TEMPLATE_CONFIG = Path("config/poc.yaml")

NORMALIZED_PARQUET = "data/interim/comments_normalized.parquet"
CONSENSUS_PARQUET = "data/processed/consensus.parquet"
RECOMMENDATIONS_JSON = "data/processed/recommendations.json"
STATIC_LIST_DEFAULT = "data/raw/static_profanity_id.txt"
OUTPUT_JSON = "data/processed/baseline_comparison.json"

#: The consensus labels, collapsed to the four buckets the comparison needs.
HARMFUL_VALUES: frozenset[str] = frozenset(l.value for l in HARMFUL)
CRITICISM_VALUE = Label.LEGITIMATE_CRITICISM.value
INVECTIVE_VALUE = Label.PUBLIC_FIGURE_INVECTIVE.value
#: D38 gives `public_figure_invective` its OWN bucket rather than folding it into
#: `legitimate_criticism`. Both are collateral (see `schemas.COLLATERAL`, and
#: `ArmResult.collateral_hidden` which sums them), but a creator deciding whether to
#: deploy a filter should be able to see "this hides 40 substantive critiques AND 300
#: angry comments about the government" as two facts, not one blended number.
BUCKETS = ("harmful", "legitimate_criticism", "public_figure_invective",
           "benign_other", "ambiguous")

#: R8. The only columns that ever leave the corpus loader. An allowlist, not a
#: denylist: a new author-ish column added upstream cannot silently appear here.
CORPUS_COLUMNS = ("comment_id", "text_original", "text_normalized")

#: R9. NOT required - the multi-platform ingest migration (build contract SS1) may not
#: have landed in every lane yet. When `platform` is absent every row is bucketed here
#: instead of guessing, and the artefact says so (`platform_data_available: false`)
#: rather than presenting a single-bucket "mix" as if it were real.
OPTIONAL_CORPUS_COLUMNS = ("platform",)
UNKNOWN_PLATFORM = "unspecified"


# ---------------------------------------------------------------------------
# matching - "a text file and a match function" (R6)
# ---------------------------------------------------------------------------

#: Digits are part of a token on purpose: on ORIGINAL text `g0blok` must be one token
#: that is not equal to `goblok`, which is exactly the miss `baseline_1` recovers.
_TOKEN_RE = re.compile(r"[0-9a-zÀ-ɏ]+")


def tokenize(text: str) -> list[str]:
    """Case-insensitive word tokens. Punctuation and emoji are separators."""
    return _TOKEN_RE.findall(text.casefold())


class TermMatcher:
    """Whole-token matching of a term list against a comment.

    Whole-token, so `jelek` does not match `jeleknya` - the same rule phrase mining
    uses (build contract §5). Every arm matches identically; the ONLY thing that
    varies between `baseline_0` and `baseline_1` is which text column is read, and
    between `baseline_1` and `ours` which terms are in the list. If the arms matched
    by different rules the decomposition would not be a decomposition.

    Case-insensitive on the raw text too. A creator's blocked-word list is not case
    sensitive, and pretending otherwise would book plain capitalisation as a
    "normalizer effect".
    """

    __slots__ = ("terms", "_lengths")

    def __init__(self, terms: Iterable[str]) -> None:
        self.terms: dict[tuple[str, ...], str] = {}
        for term in terms:
            key = tuple(tokenize(term))
            if key:
                self.terms.setdefault(key, term.strip().casefold())
        self._lengths = sorted({len(k) for k in self.terms})

    def __len__(self) -> int:
        return len(self.terms)

    def hits(self, text: str) -> list[str]:
        """Which terms fire on this text. Empty means the filter does not hide it."""
        if not self.terms or not text:
            return []
        toks = tokenize(text)
        found: set[str] = set()
        for n in self._lengths:
            for i in range(len(toks) - n + 1):
                term = self.terms.get(tuple(toks[i:i + n]))
                if term is not None:
                    found.add(term)
        return sorted(found)

    def fires(self, text: str) -> bool:
        return bool(self.hits(text))


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_config(path: Path | str | None = None) -> dict:
    """Read the run config. Falls back to the checked-in template for the default path.

    `config/local.yaml` is gitignored and machine-specific, so a fresh clone has only
    `config/poc.yaml`. Falling back keeps a smoke run possible; an EXPLICIT --config
    that does not exist is an error, because that is a typo, not a fresh clone.
    """
    import yaml

    explicit = path is not None
    p = Path(path) if path is not None else DEFAULT_CONFIG
    if not p.exists():
        if explicit:
            raise SystemExit(f"config not found: {p}")
        if TEMPLATE_CONFIG.exists():
            print(f"warning: {p} not found - falling back to {TEMPLATE_CONFIG} (template)",
                  file=sys.stderr)
            p = TEMPLATE_CONFIG
        else:
            return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _cfg(config: Mapping, *keys, default=None):
    node: object = config
    for k in keys:
        if not isinstance(node, Mapping) or k not in node:
            return default
        node = node[k]
    return node if node is not None else default


def _require(path: Path, produced_by: str) -> Path:
    if not path.exists():
        raise SystemExit(
            f"missing input: {path.as_posix()}\n"
            f"  produce it first with: {produced_by}"
        )
    return path


def _clean(value: object) -> str:
    """Parquet nulls arrive as None or float('nan') depending on the writer."""
    if value is None:
        return ""
    if isinstance(value, float):
        return ""
    return str(value)


@dataclass(frozen=True, slots=True)
class CorpusRow:
    comment_id: str
    text_original: str
    text_normalized: str
    #: R9. `UNKNOWN_PLATFORM` when the corpus predates the multi-platform migration.
    platform: str = UNKNOWN_PLATFORM

    def text(self, field_name: str) -> str:
        return self.text_original if field_name == "text_original" else self.text_normalized


def load_corpus(path: Path, split: str = "dev") -> list[CorpusRow]:
    """Dev rows only (R3), and only the allowlisted columns (R8).

    `split == "heldout"` is Channel E. Nothing downstream of ingest may read it except
    `python -m src.report heldout`, so the filter is not a flag.

    `platform` (R9) is read when present and defaulted to `UNKNOWN_PLATFORM` otherwise -
    still an allowlist, not a denylist: a new author-ish column added upstream cannot
    silently appear here just because it happens to sit next to `platform`.
    """
    import pandas as pd

    df = pd.read_parquet(path)
    missing = [c for c in CORPUS_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"{path.as_posix()} is missing required columns: {missing}")
    if "split" in df.columns:
        df = df[df["split"].astype(str) == split]
    has_platform = "platform" in df.columns
    cols = list(CORPUS_COLUMNS) + (["platform"] if has_platform else [])
    df = df[cols]                          # R8: allowlist, applied before anything else
    rows: list[CorpusRow] = []
    for r in df.itertuples(index=False):
        platform = (_clean(getattr(r, "platform", "")).strip().casefold()
                    if has_platform else "") or UNKNOWN_PLATFORM
        rows.append(CorpusRow(
            str(r.comment_id), _clean(r.text_original), _clean(r.text_normalized),
            platform=platform,
        ))
    return rows


def load_labels(path: Path) -> dict[str, str]:
    """comment_id -> one of BUCKETS, from `consensus.parquet`.

    A null label is the AMBIGUOUS tier (R2b): the system refused to decide, and the
    comment is still hidden by any filter that matches it, so it is counted, never
    dropped (R1 - `n_all` is the operational denominator).
    """
    import pandas as pd

    df = pd.read_parquet(path)
    for col in ("comment_id", "label"):
        if col not in df.columns:
            raise SystemExit(f"{path.as_posix()} is missing required column: {col}")
    tiers = df["tier"].astype(str).tolist() if "tier" in df.columns else [""] * len(df)
    out: dict[str, str] = {}
    for (cid, label), tier in zip(
        zip(df["comment_id"].tolist(), df["label"].tolist()), tiers
    ):
        out[str(cid)] = classify(label, tier)
    return out


def classify(label: object, tier: object = "") -> str:
    value = _clean(label)
    if not value or _clean(tier).lower() == "ambiguous":
        return "ambiguous"
    if value in HARMFUL_VALUES:
        return "harmful"
    if value == CRITICISM_VALUE:
        return "legitimate_criticism"
    if value == INVECTIVE_VALUE:
        return "public_figure_invective"
    return "benign_other"


def read_static_list(path: Path) -> list[str]:
    """One term per line. Only a line STARTING with `#` is a comment.

    Splitting on an inline `#` would silently destroy a censored variant like `g#blok`.
    """
    terms: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            terms.append(line)
    return terms


def suggested_family_terms(recommendations: Mapping) -> tuple[list[str], list[str]]:
    """The learned filter list: every variant of every family the system SUGGESTED.

    The verdict is read from the artefact, never recomputed here (build contract §6:
    `verdict` comes from `PhraseScore.recommend()`), so this module cannot accidentally
    apply a different rule than the one on screen.
    """
    terms: list[str] = []
    names: list[str] = []
    for fam in recommendations.get("families", []):
        if str(fam.get("verdict", "")).upper() != "SUGGESTED":
            continue
        names.append(str(fam.get("name", "")))
        variants = [str(v) for v in fam.get("variants", []) if str(v).strip()]
        if not variants:
            fallback = fam.get("canonical_key") or fam.get("name") or ""
            variants = [str(fallback).lower()] if fallback else []
        terms.extend(variants)
    # dedupe, keep order - a variant shared by two families must not be counted twice
    seen: set[str] = set()
    unique = [t for t in terms if not (t.casefold() in seen or seen.add(t.casefold()))]
    return unique, sorted(names)


# ---------------------------------------------------------------------------
# the arms
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ArmSpec:
    arm_id: str
    text_field: str          # which column this arm reads
    filters: str             # "static" | "learned" - which list this arm is given
    description: str


ARMS: tuple[ArmSpec, ...] = (
    ArmSpec("baseline_0", "text_original", "static",
            "original text + static list - what a creator has today"),
    ArmSpec("baseline_1", "text_normalized", "static",
            "normalized text + static list - isolates what normalization buys"),
    ArmSpec("ours", "text_normalized", "learned",
            "normalized text + learned filter families"),
    ArmSpec("hybrid", "text_normalized", "hybrid",
            "learned filter families as a CANDIDATE trigger only - a comment is hidden "
            "only if it also matches, per its own comment, the harmful bucket"),
)

#: `hybrid`'s fired set is, by construction, exactly the subset of `ours`'s fired set
#: whose own per-comment label is "harmful" (`run_arm` below enforces this). That makes
#: two things true and worth being explicit about, since a reader could otherwise
#: mistake this arm for an independently-validated classifier:
#:
#: 1. `hybrid.harmful_caught == ours.harmful_caught` and `hybrid.criticism_hidden == 0`
#:    and `hybrid.hidden["ambiguous"] == hidden["benign_other"] == 0` EXACTLY, always -
#:    not an empirical finding, a mathematical consequence of the firing rule. Recall
#:    is identical to `ours`; every unit of collateral `ours` had is gone by definition.
#: 2. `hybrid`'s precision is therefore trivially 1.000 and is NOT evidence that per-
#:    comment confirmation is a good classifier - it is gated on the very label used to
#:    score it. What it demonstrates is the CEILING: if a live per-comment check were as
#:    good as the consensus process this whole project already treats as ground truth
#:    (R5 - "according to our automated annotation system", never asserted as fact),
#:    that check would remove 100% of a word-filter's measured collateral at no recall
#:    cost. Deploying it for real means running that check live on the (small) matched
#:    subset - not free access to a label that does not exist for a brand-new comment.


def _precision_from(hidden: Mapping[str, int], n_fired: int) -> dict[str, float | None]:
    """T4 - conditioned on the comments FIRED ON, never on the stream. Shared by the
    pooled and by-platform numbers so they can never silently disagree about method.

        lower    = harmful / fired               all ambiguous non-harmful
        upper    = (harmful + ambiguous) / fired  all ambiguous harmful
        resolved = harmful / resolved-fired
    """
    if not n_fired:
        return {"lower": None, "upper": None, "resolved": None}
    n_amb = hidden.get("ambiguous", 0)
    harmful = hidden.get("harmful", 0)
    resolved = n_fired - n_amb
    return {
        "lower": harmful / n_fired,
        "upper": (harmful + n_amb) / n_fired,
        "resolved": (harmful / resolved) if resolved else None,
    }


def _recall_from(hidden: Mapping[str, int], corpus: Mapping[str, int]) -> float | None:
    total = corpus.get("harmful", 0)
    return (hidden.get("harmful", 0) / total) if total else None


def _criticism_share_from(hidden: Mapping[str, int], corpus: Mapping[str, int]) -> float | None:
    total = corpus.get("legitimate_criticism", 0)
    return (hidden.get("legitimate_criticism", 0) / total) if total else None


@dataclass(slots=True)
class ArmResult:
    spec: ArmSpec
    n_stream: int
    fired_ids: list[str]
    hidden: Counter = field(default_factory=Counter)      # bucket -> count
    corpus: Counter = field(default_factory=Counter)      # bucket -> count over the stream
    term_hits: Counter = field(default_factory=Counter)   # term -> comments it fired on
    #: R9. Same two counters, keyed first by platform. Populated alongside `hidden`/
    #: `corpus` in `run_arm` so the pooled and per-platform views are the same data,
    #: never two independently-computed numbers that could drift apart.
    hidden_by_platform: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    corpus_by_platform: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    n_terms: int = 0
    audit_sample: list[str] = field(default_factory=list)

    # -- the numbers ------------------------------------------------------
    @property
    def comments_hidden(self) -> int:
        return len(self.fired_ids)

    @property
    def harmful_caught(self) -> int:
        return self.hidden["harmful"]

    @property
    def criticism_hidden(self) -> int:
        return self.hidden["legitimate_criticism"]

    @property
    def invective_hidden(self) -> int:
        """D38 - crude speech aimed at public figures/institutions that this arm hides."""
        return self.hidden["public_figure_invective"]

    @property
    def collateral_hidden(self) -> int:
        """Total non-harmful negative speech destroyed: criticism + invective.

        This is the honest headline for "what does the filter cost". Reporting
        `criticism_hidden` alone after D38 would UNDERSTATE the damage, because the
        political comments a blocked-word list hides moved out of that bucket into
        `public_figure_invective` - they did not stop being collateral.
        """
        return self.criticism_hidden + self.invective_hidden

    def precision(self) -> dict[str, float | None]:
        """T4 - conditioned on the comments this arm FIRES ON, never on the stream."""
        return _precision_from(self.hidden, self.comments_hidden)

    def recall(self) -> float | None:
        return _recall_from(self.hidden, self.corpus)

    def criticism_hidden_share(self) -> float | None:
        return _criticism_share_from(self.hidden, self.corpus)

    def fire_rate(self) -> float:
        return self.comments_hidden / self.n_stream if self.n_stream else 0.0

    def platform_breakdown(self) -> dict[str, dict]:
        """R9 - every pooled metric above, recomputed per platform from the SAME
        counters (`hidden_by_platform`/`corpus_by_platform`) the pooled numbers come
        from, so a family that only "works" on one app is visible rather than
        averaged into the pooled figure, and the two views cannot disagree by
        construction.
        """
        out: dict[str, dict] = {}
        platforms = sorted(set(self.corpus_by_platform) | set(self.hidden_by_platform))
        for p in platforms:
            hidden = self.hidden_by_platform.get(p, Counter())
            corpus = self.corpus_by_platform.get(p, Counter())
            n_fired = sum(hidden.values())
            n_stream = sum(corpus.values())
            out[p] = {
                "n_stream": n_stream,
                "comments_hidden": n_fired,
                "harmful_caught": hidden.get("harmful", 0),
                "legitimate_criticism_hidden": hidden.get("legitimate_criticism", 0),
                "legitimate_criticism_hidden_share": _criticism_share_from(hidden, corpus),
                "hidden_breakdown": {b: hidden.get(b, 0) for b in BUCKETS},
                "corpus_breakdown": {b: corpus.get(b, 0) for b in BUCKETS},
                "precision_over_fired_stratum": _precision_from(hidden, n_fired),
                "recall_over_resolved_harmful": _recall_from(hidden, corpus),
                "fire_rate_on_stream": (n_fired / n_stream) if n_stream else 0.0,
            }
        return out


def run_arm(
    spec: ArmSpec,
    rows: Sequence[CorpusRow],
    labels: Mapping[str, str],
    terms: Sequence[str],
    audit_samples: int = 20,
    seed: int = 0,
) -> ArmResult:
    """Evaluate ONE arm. It sees its own text column and its own term list, nothing else
    (plus, for `filters == "hybrid"` only, the SAME per-comment `labels` every arm
    already reads for evaluation - see the note above `ARMS` for exactly what that
    does and does not prove).

    The signature is the R6 guarantee made mechanical: an arm cannot read another arm's
    fired set, counts or artefact, because it is never given one. Run them in any order
    and the numbers are identical.
    """
    matcher = TermMatcher(terms)
    result = ArmResult(spec=spec, n_stream=len(rows), fired_ids=[], n_terms=len(matcher))
    for row in rows:
        bucket = labels.get(row.comment_id)
        if bucket is None:
            continue                      # unlabelled comments are outside the comparison
        result.corpus[bucket] += 1
        result.corpus_by_platform[row.platform][bucket] += 1          # R9
        hits = matcher.hits(row.text(spec.text_field))
        # `hybrid`: a term match is only a CANDIDATE - it hides the comment only if that
        # comment's own label confirms harm, never on the word match alone.
        fires = bool(hits) and (spec.filters != "hybrid" or bucket == "harmful")
        if fires:
            result.fired_ids.append(row.comment_id)
            result.hidden[bucket] += 1
            result.hidden_by_platform[row.platform][bucket] += 1      # R9
            for term in hits:
                result.term_hits[term] += 1
    result.audit_sample = _audit_sample(result.fired_ids, audit_samples, seed)
    return result


def _audit_sample(fired: Sequence[str], k: int, seed: int) -> list[str]:
    """T4 - draw the audit sample from the FIRED stratum, with a recorded propensity.

    A filter firing at 3% means a uniform sample of 460 stream comments contains about
    14 fired items, so a stream sample cannot support a precision claim. Sampling the
    fired stratum can, provided the inclusion probability is written down beside it.
    """
    rng = random.Random(seed)
    pool = sorted(fired)
    return sorted(rng.sample(pool, min(k, len(pool))))


def _arm_json(result: ArmResult, audit_samples: int) -> dict:
    n_fired = result.comments_hidden
    precision = result.precision()
    return {
        "arm": result.spec.arm_id,
        "by_platform": result.platform_breakdown(),   # R9 - pooled numbers below, per app
        "text": result.spec.text_field,
        "filters": result.spec.filters,
        "description": result.spec.description,
        "figures_according_to": ATTRIBUTION,          # R5
        "n_terms": result.n_terms,
        "comments_hidden": n_fired,
        "harmful_caught": result.harmful_caught,
        "legitimate_criticism_hidden": result.criticism_hidden,
        "public_figure_invective_hidden": result.invective_hidden,          # D38
        "collateral_hidden": result.collateral_hidden,                      # D38
        "legitimate_criticism_hidden_share": result.criticism_hidden_share(),
        "hidden_breakdown": {b: result.hidden[b] for b in BUCKETS},
        "precision_over_fired_stratum": {
            "denominator": "the comments this arm hides, not the stream (T4)",
            "n_fired": n_fired,
            "lower": precision["lower"],
            "upper": precision["upper"],
            "resolved": precision["resolved"],
            "note": (
                "an interval, not a point: an ambiguous match has no trusted class and "
                "is hidden anyway (R1). lower assumes every ambiguous match is "
                "non-harmful, upper assumes every one is harmful."
            ),
        },
        "recall_over_resolved_harmful": {
            "value": result.recall(),
            "harmful_caught": result.harmful_caught,
            "n_harmful_in_corpus": result.corpus["harmful"],
            "note": (
                "ambiguous comments are excluded from this denominator - they have no "
                "trusted class. The count of them is in corpus.labels."
            ),
        },
        "sampling_design": {
            "rule": "T4 - precision is conditioned on the fired stratum, never on the stream",
            "fire_rate_on_stream": result.fire_rate(),
            "n_stream": result.n_stream,
            "n_fired": n_fired,
            "audit_sample_size": len(result.audit_sample),
            "audit_inclusion_probability": (
                len(result.audit_sample) / n_fired if n_fired else None
            ),
            "audit_sample_comment_ids": result.audit_sample,
            "why": (
                f"at a fire rate of {result.fire_rate():.1%}, a uniform sample of 460 "
                f"stream comments would contain about {round(460 * result.fire_rate())} "
                "fired items - too few to support a precision claim. The audit sample "
                "above is drawn from the fired stratum instead, and its inclusion "
                "probability is recorded so the stream rate can be recovered."
            ),
            "requested_audit_samples": audit_samples,
        },
        "top_firing_terms": [
            {"term": t, "comments": n} for t, n in result.term_hits.most_common(10)
        ],
    }


def _delta(a: ArmResult, b: ArmResult) -> dict:
    """a - b, for the metrics the decomposition is about."""
    def sub(x: float | None, y: float | None) -> float | None:
        return None if x is None or y is None else x - y

    return {
        "comments_hidden": a.comments_hidden - b.comments_hidden,
        "harmful_caught": a.harmful_caught - b.harmful_caught,
        "legitimate_criticism_hidden": a.criticism_hidden - b.criticism_hidden,
        "public_figure_invective_hidden": a.invective_hidden - b.invective_hidden,  # D38
        "collateral_hidden": a.collateral_hidden - b.collateral_hidden,             # D38
        "precision_lower_over_fired_stratum": sub(
            a.precision()["lower"], b.precision()["lower"]),
        "recall_over_resolved_harmful": sub(a.recall(), b.recall()),
        "legitimate_criticism_hidden_share": sub(
            a.criticism_hidden_share(), b.criticism_hidden_share()),
    }


def build_comparison(
    results: Mapping[str, ArmResult],
    *,
    static_terms: Sequence[str],
    learned_terms: Sequence[str],
    suggested_families: Sequence[str],
    static_list_path: str,
    audit_samples: int,
    split: str = "dev",
) -> dict:
    """Assemble the artefact, with BOTH decompositions stated explicitly:
    normalizer effect vs. learning effect (R6), and pooled vs. by-platform (R9)."""
    b0, b1, ours = results["baseline_0"], results["baseline_1"], results["ours"]
    hybrid = results.get("hybrid")
    any_arm = b0
    # The three arms see the same labelled stream (only the text column and the term
    # list differ), so pooled-by-platform counts are identical across arms; b0 stands
    # in for all three here the same way it already does for `labels` below.
    platform_counts = {
        p: sum(c.values()) for p, c in sorted(any_arm.corpus_by_platform.items())
    }
    platform_data_available = any(p != UNKNOWN_PLATFORM for p in platform_counts)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "according_to": ATTRIBUTION,                       # R5
        "question": (
            "RQ1 - do learned filters beat a static list, and how much of any gap is "
            "normalization rather than learning?"
        ),
        "independence": (
            "D18 - the three arms are parallel. No baseline output is an input to phrase "
            "mining, family scoring or the recommendation rule; they meet only in this "
            "comparison."
        ),
        "split": split,
        "corpus": {
            "split": split,
            "n_comments_scored": any_arm.n_stream,
            "labels": {b: any_arm.corpus[b] for b in BUCKETS},
            "platforms": platform_counts,                          # R9, pooled
            "platform_data_available": platform_data_available,
            "note": (
                "development channels only (R3). Channel E is read by "
                "`python -m src.report heldout` and by nothing else. "
                + ("Every arm below also reports a `by_platform` breakdown (R9)."
                   if platform_data_available else
                   "`platform` is not yet on this corpus (the multi-platform ingest "
                   "migration, build contract SS1, has not landed here yet), so every "
                   "row is counted under a single 'unspecified' platform below.")
            ),
        },
        "filter_lists": {
            "static": {
                "source": static_list_path,
                "n_terms": len(static_terms),
                "arms": ["baseline_0", "baseline_1"],
            },
            "learned": {
                "source": RECOMMENDATIONS_JSON,
                "n_terms": len(learned_terms),
                "n_families_suggested": len(suggested_families),
                "families": list(suggested_families),
                "arms": ["ours"],
            },
        },
        "arms": [_arm_json(results[spec.arm_id], audit_samples)
                for spec in ARMS if spec.arm_id in results],
        "decompositions": {
            "normalizer_effect": {
                "definition": "baseline_1 - baseline_0",
                "means": "what the shared normalizer buys, holding the filter list fixed",
                "deltas": _delta(b1, b0),
            },
            "learning_effect": {
                "definition": "ours - baseline_1",
                "means": "what learned filters buy, holding the normalizer fixed",
                "deltas": _delta(ours, b1),
            },
            "why_both": (
                "Reporting only `ours - baseline_0` would book the normalizer's gain as "
                "learning. Without baseline_1, catching `g0blok` proves only that you "
                "have a normalizer (R6)."
            ),
            **({
                "confirmation_effect": {
                    "definition": "hybrid - ours",
                    "means": (
                        "what requiring per-comment confirmation buys, holding the "
                        "candidate word list fixed. Deliberately NOT independent "
                        "evidence of precision (see the note above ARMS in "
                        "src/baseline.py) - harmful_caught is mathematically "
                        "identical to `ours` by construction; the deltas below show "
                        "collateral removed at zero recall cost, not a new "
                        "classifier's accuracy."
                    ),
                    "deltas": _delta(hybrid, ours),
                },
            } if hybrid is not None else {}),
        },
        "headline": {
            "metric": "collateral_hidden",
            "means": (
                "collateral damage - non-harmful negative speech the arm hides: "
                "legitimate_criticism + public_figure_invective (D38). Reporting only "
                "the criticism half would understate it, since the political comments "
                "a blocked-word list hides now sit in the invective bucket."
            ),
            "baseline_0": b0.collateral_hidden,
            "baseline_1": b1.collateral_hidden,
            "ours": ours.collateral_hidden,
            **({"hybrid": hybrid.collateral_hidden} if hybrid is not None else {}),
            "breakdown": {
                arm.spec.arm_id: {
                    "legitimate_criticism": arm.criticism_hidden,
                    "public_figure_invective": arm.invective_hidden,
                }
                for arm in (b0, b1, ours, *( (hybrid,) if hybrid is not None else () ))
            },
            "according_to": ATTRIBUTION,
        },
        "limitations": [
            "No stance detection: a comment condemning a phrase contains the phrase and "
            "is matched by every arm alike (F2).",
            "Matching is whole-token and case-insensitive in all three arms, so the only "
            "difference between baseline_0 and baseline_1 is the text column.",
            "Labels are automated. Every figure here is " + ATTRIBUTION + ".",
        ],
    }


def write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run(
    config: Mapping,
    root: Path = Path("."),
    audit_samples: int = 20,
    seed: int = 0,
) -> dict:
    static_path = root / str(
        _cfg(config, "baseline", "static_dictionary", default=STATIC_LIST_DEFAULT))
    corpus_path = _require(root / NORMALIZED_PARQUET, "python -m src.normalize.text")
    consensus_path = _require(root / CONSENSUS_PARQUET, "python -m src.consensus")
    rec_path = _require(root / RECOMMENDATIONS_JSON, "python -m src.phrases.families")
    _require(static_path, "add the creator's blocked-word list at " + static_path.as_posix())

    rows = load_corpus(corpus_path)                     # dev only (R3)
    labels = load_labels(consensus_path)
    recommendations = json.loads(rec_path.read_text(encoding="utf-8"))

    # Each arm's term list is built separately and handed to a separate run_arm call.
    static_terms = read_static_list(static_path)
    learned_terms, suggested = suggested_family_terms(recommendations)
    # hybrid uses the SAME candidate terms as `ours` - the difference is entirely in
    # the firing gate (run_arm), not in which words are watched for.
    terms_for = {"static": static_terms, "learned": learned_terms, "hybrid": learned_terms}

    results = {
        spec.arm_id: run_arm(spec, rows, labels, terms_for[spec.filters],
                             audit_samples=audit_samples, seed=seed)
        for spec in ARMS
    }
    return build_comparison(
        results,
        static_terms=static_terms,
        learned_terms=learned_terms,
        suggested_families=suggested,
        static_list_path=static_path.as_posix(),
        audit_samples=audit_samples,
    )


def _print_summary(payload: Mapping) -> None:
    print(f"baseline comparison - all figures {ATTRIBUTION}")
    header = (f"{'arm':<12}{'hidden':>9}{'harmful':>9}{'recall':>9}"
              f"{'crit':>7}{'invect':>8}{'collat':>8}{'precision(lo)':>15}")
    print(header)
    print("-" * len(header))
    for arm in payload["arms"]:
        lo = arm["precision_over_fired_stratum"]["lower"]
        rec = arm["recall_over_resolved_harmful"]["value"]
        print(
            f"{arm['arm']:<12}{arm['comments_hidden']:>9}{arm['harmful_caught']:>9}"
            f"{('n/a' if rec is None else f'{rec:.1%}'):>9}"
            f"{arm['legitimate_criticism_hidden']:>7}"
            f"{arm['public_figure_invective_hidden']:>8}"
            f"{arm['collateral_hidden']:>8}"
            f"{('n/a' if lo is None else f'{lo:.3f}'):>15}"
        )
    for key in ("normalizer_effect", "learning_effect", "confirmation_effect"):
        block = payload["decompositions"].get(key)
        if block is None:
            continue
        d = block["deltas"]
        print(f"\n{key} ({block['definition']}):")
        print(f"  comments hidden              {d['comments_hidden']:+d}")
        print(f"  harmful caught               {d['harmful_caught']:+d}")
        print(f"  legitimate criticism hidden  {d['legitimate_criticism_hidden']:+d}")
        print(f"  public-figure invective hid  {d['public_figure_invective_hidden']:+d}")
        print(f"  TOTAL collateral hidden      {d['collateral_hidden']:+d}")

    platforms = payload["corpus"].get("platforms", {})
    if payload["corpus"].get("platform_data_available") and platforms:
        print("\nby platform (pooled corpus - every arm sees the same labelled stream):")
        for p, n in platforms.items():
            print(f"  {p:<12}{n:>8}")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m src.baseline",
        description="Four parallel baseline arms (R6). Development channels only.",
    )
    ap.add_argument("--config", default=None, help="default: config/local.yaml")
    ap.add_argument("--out", default=OUTPUT_JSON)
    ap.add_argument("--audit-samples", type=int, default=20,
                    help="size of the fired-stratum audit sample per arm (T4)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    config = load_config(args.config)
    payload = run(config, audit_samples=args.audit_samples, seed=args.seed)
    out = Path(args.out)
    write_json(out, payload)
    _print_summary(payload)
    print(f"\nwrote {out.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
