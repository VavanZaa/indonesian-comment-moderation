"""Capability-aware consensus.

A symmetric N-of-M vote across annotators with DIFFERENT CAPABILITIES is wrong here.
A1 (hate-speech classifier) and A2 (rules) structurally cannot emit
`legitimate_criticism`; their adapters previously turned "no harm found" into a
positive vote for `benign_other`, which then outvoted A3 two-to-one. The result was
n_legitimate_criticism ~ 0 for every phrase, which makes C(f) ~ 0 and demo scene 2
impossible. See AGENTS.md TRAP 1.

So aggregation is hierarchical:

    A1, A2  ->  broad harm corroboration  {HARM, NO_HARM_EVIDENCE}
    A3      ->  the full 5-label semantic judgement

NO_HARM_EVIDENCE is a *semantic-label* abstention - it says nothing about which
non-harmful class applies - but it is still negative corroborating evidence for harm,
and it is what lets `legitimate_criticism` reach HIGH when both corroborators are
silent. Abstention on the label, evidence on the harm axis.

    A3 harmful:      2 corroborate -> HIGH | 1 -> MEDIUM | 0 -> AMBIGUOUS
    A3 non-harmful:  0 flag harm   -> HIGH | 1 -> MEDIUM | 2 -> AMBIGUOUS
    A3 UNCORROBORABLE (threat_privacy): always MEDIUM + needs_review    (D39)

Agreement strength sets the tier; full disagreement is ambiguous. The asymmetry is
deliberate: A3 asserting harm with zero corroboration is exactly the "LLM over-flags
harsh criticism" failure mode, and routing it to AMBIGUOUS keeps it out of n_harmful,
where it would inflate phrases like `jelek` and invert demo scene 2. The cost is that
subtle harm A1/A2 cannot see also lands in AMBIGUOUS - conservative, and it enriches
the ambiguous reel rather than corrupting the filter list.

D39 (2026-09-10) narrowed that cost where it stopped being acceptable. "Subtle harm
A1/A2 cannot see" turned out to include most direct threats: a deliberate threat
contains no profanity, so the table above discarded 13 of 17 of them on real data,
including six "we have your location" doxxing comments (docs/11 §7c). The trade-off was
right while the goal was a filter list; it is wrong for a class with asymmetric costs
once the goal is per-comment moderation. So labels in `schemas.UNCORROBORABLE` bypass
the table: A1/A2 cannot perceive them, their votes decide nothing in either direction,
and the record is kept, held at MEDIUM, and routed to a human via `review_queue.parquet`.

This makes A2's rule coverage load-bearing: a blatantly harmful family that A2 does
not match will never reach HIGH. Keep the lexicon families broad.

What this DOES fix, and what it does not. A3 mislabelling clean criticism as
harassment with neither corroborator firing lands in AMBIGUOUS, so it never enters
n_harmful. The residual gap is criticism that trips the harm lexicon:

    "videonya goblok banget"   A1 HARM, A2 HARM, A3 harassment_personal -> HIGH

A1/A2 corroborate on lexical grounds while A3 supplies the wrong semantic label, and
nothing in the ensemble can separate the two - on that boundary it is n=1 and always
was (TRAP 1). This is the specific case the screen check must hunt for, and the reason
A3's prompt distinguishes criticism of a *work* from criticism of a *person*.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from .schemas import (
    ConsensusRecord, HarmSignal, HarmVote, Label, LabelVote, Tier, HARMFUL, TRAINING_WEIGHT,
    UNCORROBORABLE,
)


def consense(label_vote: LabelVote, harm_votes: Sequence[HarmVote]) -> ConsensusRecord:
    """Combine A3's semantic label with A1/A2's harm corroboration."""
    cid = label_vote.comment_id
    if any(v.comment_id != cid for v in harm_votes):
        raise ValueError("votes span multiple comments")
    if not harm_votes:
        raise ValueError("no harm votes: A1/A2 corroboration is required")

    corroboration = sum(v.signal is HarmSignal.HARM for v in harm_votes)
    n = len(harm_votes)
    a3_says_harm = label_vote.label in HARMFUL

    needs_review = False
    if label_vote.label in UNCORROBORABLE:
        # D39. A1/A2 cannot perceive this class, so their votes say nothing about it in
        # either direction. Their silence must not drop it to AMBIGUOUS - that is what
        # discarded 13 of 17 direct threats - and a lexical HARM must not promote it to
        # HIGH, because the corroborated threats were no more often real than the
        # uncorroborated ones (46% vs 43%, docs/11 §7c). A3 is the only capable judge:
        # keep its label, hold it at MEDIUM, and send it to a human. `corroboration`
        # is still recorded below, for audit - it just no longer decides anything here.
        tier, label, needs_review = Tier.MEDIUM, label_vote.label, True
    else:
        agreeing = corroboration if a3_says_harm else n - corroboration
        if agreeing == n:
            tier, label = Tier.HIGH, label_vote.label
        elif agreeing == 0:
            tier, label = Tier.AMBIGUOUS, None
        else:
            tier, label = Tier.MEDIUM, label_vote.label

    votes: dict[str, str] = {label_vote.annotator_id: label_vote.label.value}
    votes.update({v.annotator_id: v.signal.value for v in harm_votes})

    return ConsensusRecord(
        comment_id=cid, label=label, tier=tier,
        weight=TRAINING_WEIGHT[tier], votes=votes, corroboration=corroboration,
        needs_review=needs_review,
    )


# =============================================================================
# CLI (build contract §4). Everything below is new; `consense()` above is
# untouched. `python -m src.consensus --config config/local.yaml` joins the
# three vote parquets on comment_id, calls `consense` per comment, and writes
# `consensus.parquet` (every tier, including AMBIGUOUS with label=None - phrase
# mining needs those rows to compute n_ambiguous) and `ambiguous.parquet`
# (R2b: excluded from training, never deleted, joined to text_normalized for
# demo scene 5).
# =============================================================================

#: repo-relative, posix-style (build contract §0)
VOTES_A1_PATH = "data/interim/votes_a1.parquet"
VOTES_A2_PATH = "data/interim/votes_a2.parquet"
VOTES_A3_PATH = "data/interim/votes_a3.parquet"
NORMALIZED_PATH = "data/interim/comments_normalized.parquet"
CONSENSUS_PATH = "data/processed/consensus.parquet"
AMBIGUOUS_PATH = "data/processed/ambiguous.parquet"

#: The held-out run writes BESIDE the development consensus, never over it (R3).
#: Overwriting `consensus.parquet` with Channel E labels would silently turn every
#: dev-side number - phrase families, the baseline arms, the dev column of the transfer
#: table - into held-out numbers, and the freeze would not catch it because no frozen
#: file changed. `src/annotate/run.py` namespaces the vote parquets the same way.
HELDOUT_VOTES_DIR = "data/interim/heldout"
HELDOUT_CONSENSUS_PATH = "data/processed/consensus_heldout.parquet"
HELDOUT_AMBIGUOUS_PATH = "data/processed/ambiguous_heldout.parquet"

SPLITS: frozenset[str] = frozenset({"dev", "heldout"})

#: D39. Every record flagged `needs_review` - today, every `UNCORROBORABLE` (threat)
#: label - joined to its text so a human can confirm or reject it. Written BESIDE the
#: consensus, never instead of it: these rows also stay in `consensus.parquet` as harmful
#: MEDIUM records. Being queued for review routes a comment; it does not demote it.
REVIEW_PATH = "data/processed/review_queue.parquet"
HELDOUT_REVIEW_PATH = "data/processed/review_queue_heldout.parquet"


def split_paths(split: str = "dev") -> dict[str, str]:
    """Every input and output path for one split. One place, so the two runs cannot
    half-agree about where their data lives."""
    if split not in SPLITS:
        raise SystemExit(f"split must be one of {sorted(SPLITS)}, got {split!r}")
    if split == "dev":
        return {
            "a1": VOTES_A1_PATH, "a2": VOTES_A2_PATH, "a3": VOTES_A3_PATH,
            "consensus": CONSENSUS_PATH, "ambiguous": AMBIGUOUS_PATH,
            "review": REVIEW_PATH,
        }
    return {
        "a1": f"{HELDOUT_VOTES_DIR}/votes_a1.parquet",
        "a2": f"{HELDOUT_VOTES_DIR}/votes_a2.parquet",
        "a3": f"{HELDOUT_VOTES_DIR}/votes_a3.parquet",
        "consensus": HELDOUT_CONSENSUS_PATH, "ambiguous": HELDOUT_AMBIGUOUS_PATH,
        "review": HELDOUT_REVIEW_PATH,
    }

_CONSENSUS_COLUMNS = [
    "comment_id", "label", "tier", "weight", "votes_json", "corroboration", "platform",
    "needs_review",
]
_AMBIGUOUS_COLUMNS = _CONSENSUS_COLUMNS + ["text_normalized"]
_REVIEW_COLUMNS = _CONSENSUS_COLUMNS + ["text_normalized"]


def write_review_queue(
    path: str | Path,
    records: Sequence[ConsensusRecord],
    context: Mapping[str, Mapping[str, str]],
) -> int:
    """`review_queue.parquet` (D39) - the `needs_review` subset, joined to
    `text_normalized`, for a human to confirm or reject.

    This is fix 1 of docs/11 §7c: an uncertain threat goes to a PERSON, never to "not
    harmful". Threat is the one category where the costs are sharply asymmetric - a
    missed "we have your location" is far worse than a false alarm - and A3 alone is
    right about roughly half the time on it, so neither trusting nor discarding it is
    defensible. Writing the queue does not change any label or tier.
    """
    import pandas as pd

    rows = []
    for r in records:
        if not r.needs_review:
            continue
        ctx = context.get(r.comment_id, {})
        row = _record_row(r, ctx.get("platform", "unknown"))
        row["text_normalized"] = ctx.get("text_normalized", "")
        rows.append(row)
    frame = pd.DataFrame(rows, columns=_REVIEW_COLUMNS)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(p, index=False)
    return len(rows)


def load_config(path: str | Path) -> dict:
    """Read a pipeline config. Missing file is an operator error, not a default."""
    import yaml

    p = Path(path)
    if not p.exists():
        raise SystemExit(f"config not found: {p} (copy config/poc.yaml to config/local.yaml)")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _opt_float(value: object) -> float | None:
    import pandas as pd

    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return float(value)


def _read_parquet(path: str | Path, what: str):
    import pandas as pd

    p = Path(path)
    if not p.exists():
        raise SystemExit(f"{p} not found - run `python -m src.annotate.run` first ({what})")
    return pd.read_parquet(p)


def load_harm_votes(path: str | Path) -> dict[str, HarmVote]:
    """A1/A2 output: `comment_id, annotator_id, signal, raw_json, confidence`.

    Already adapted upstream (R4) - `signal` is a `HarmSignal` value, not a raw
    native prediction, so this reconstructs `HarmVote` directly with no re-adapting.
    """
    frame = _read_parquet(path, "A1/A2 votes")
    votes: dict[str, HarmVote] = {}
    for row in frame.itertuples(index=False):
        cid = str(row.comment_id)
        votes[cid] = HarmVote(
            annotator_id=str(row.annotator_id),
            comment_id=cid,
            signal=HarmSignal(str(row.signal)),
            raw=getattr(row, "raw_json", None),
            confidence=_opt_float(getattr(row, "confidence", None)),
        )
    return votes


def load_label_votes(path: str | Path) -> dict[str, LabelVote]:
    """A3 output: `comment_id, annotator_id, label, raw_json, confidence`."""
    frame = _read_parquet(path, "A3 votes")
    votes: dict[str, LabelVote] = {}
    for row in frame.itertuples(index=False):
        cid = str(row.comment_id)
        votes[cid] = LabelVote(
            annotator_id=str(row.annotator_id),
            comment_id=cid,
            label=Label(str(row.label)),
            raw=getattr(row, "raw_json", None),
            confidence=_opt_float(getattr(row, "confidence", None)),
        )
    return votes


def load_comment_context(path: str | Path) -> dict[str, dict[str, str]]:
    """`comment_id -> {"platform", "text_normalized"}` off the normalized comments.

    Required for R2b (`ambiguous.parquet` is joined to `text_normalized`) and R9
    (`consensus.parquet` carries `platform` through). A missing `platform` column -
    `src/normalize/` landing before its R9 support - degrades to "unknown" with a
    warning rather than blocking the run; a missing file is an operator error.
    """
    import pandas as pd

    p = Path(path)
    if not p.exists():
        raise SystemExit(f"{p} not found - run `python -m src.normalize.text` first")
    frame = pd.read_parquet(p)
    has_platform = "platform" in frame.columns
    has_text = "text_normalized" in frame.columns
    if not has_platform:
        print(f"WARNING: {p} has no 'platform' column yet - R9 provenance will read 'unknown'")
    ctx: dict[str, dict[str, str]] = {}
    for row in frame.itertuples(index=False):
        cid = str(row.comment_id)
        platform = getattr(row, "platform", None) if has_platform else None
        text = getattr(row, "text_normalized", None) if has_text else None
        ctx[cid] = {
            "platform": "unknown" if platform is None or pd.isna(platform) else str(platform),
            "text_normalized": "" if text is None or pd.isna(text) else str(text),
        }
    return ctx


def _record_row(rec: ConsensusRecord, platform: str) -> dict:
    return {
        "comment_id": rec.comment_id,
        "label": None if rec.label is None else rec.label.value,
        "tier": rec.tier.value,
        "weight": rec.weight,
        "votes_json": json.dumps(rec.votes, sort_keys=True),
        "corroboration": rec.corroboration,
        "platform": platform,
        "needs_review": rec.needs_review,                                  # D39
    }


def write_consensus(
    path: str | Path,
    records: Sequence[ConsensusRecord],
    context: Mapping[str, Mapping[str, str]],
) -> None:
    """`consensus.parquet` - every tier, including AMBIGUOUS with `label=None`.

    Phrase mining needs the ambiguous rows to compute `n_ambiguous` (R1), so nothing
    here filters by tier.
    """
    import pandas as pd

    rows = [_record_row(r, context.get(r.comment_id, {}).get("platform", "unknown"))
            for r in records]
    frame = pd.DataFrame(rows, columns=_CONSENSUS_COLUMNS)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(p, index=False)


def write_ambiguous(
    path: str | Path,
    records: Sequence[ConsensusRecord],
    context: Mapping[str, Mapping[str, str]],
) -> int:
    """`ambiguous.parquet` (R2b) - the AMBIGUOUS-tier subset, joined to
    `text_normalized`. Excluded from training, never deleted: demo scene 5."""
    import pandas as pd

    rows = []
    for r in records:
        if r.tier is not Tier.AMBIGUOUS:
            continue
        ctx = context.get(r.comment_id, {})
        row = _record_row(r, ctx.get("platform", "unknown"))
        row["text_normalized"] = ctx.get("text_normalized", "")
        rows.append(row)
    frame = pd.DataFrame(rows, columns=_AMBIGUOUS_COLUMNS)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(p, index=False)
    return len(rows)


def run(config: Mapping, root: str | Path = ".", split: str = "dev") -> dict:
    """CLI body for `python -m src.consensus`. Returns a summary for the caller.

    A comment missing its A3 vote is skipped with a counted warning (A3 supplies
    the label, so there is nothing to consense without it). A comment that HAS an
    A3 vote but is missing A1 or A2 is an error: `consense()` requires corroboration
    from both, and silently dropping it would hide a broken annotate run.

    `split="heldout"` reads the held-out vote parquets and writes the held-out
    consensus. Same code, same frozen `consense()`, different files - which is what
    makes the transfer comparison in `src/report.py heldout` an honest one.
    """
    root = Path(root)
    paths = split_paths(split)

    a1_votes = load_harm_votes(root / paths["a1"])
    a2_votes = load_harm_votes(root / paths["a2"])
    a3_votes = load_label_votes(root / paths["a3"])
    context = load_comment_context(root / NORMALIZED_PATH)

    comment_ids = sorted(set(a1_votes) | set(a2_votes) | set(a3_votes))

    records: list[ConsensusRecord] = []
    n_missing_a3 = 0
    for cid in comment_ids:
        label_vote = a3_votes.get(cid)
        if label_vote is None:
            n_missing_a3 += 1
            continue
        if cid not in a1_votes or cid not in a2_votes:
            raise SystemExit(
                f"comment {cid!r} has an A3 vote but is missing A1 or A2 corroboration - "
                "consense() requires both (AGENTS.md TRAP 1 / build contract §4)"
            )
        records.append(consense(label_vote, [a1_votes[cid], a2_votes[cid]]))

    if n_missing_a3:
        print(f"WARNING: {n_missing_a3} comment(s) skipped - no A3 vote")

    write_consensus(root / paths["consensus"], records, context)
    n_ambiguous = write_ambiguous(root / paths["ambiguous"], records, context)
    n_review = write_review_queue(root / paths["review"], records, context)

    summary = {
        "split": split,
        "n_comments": len(comment_ids),
        "n_consensus": len(records),
        "n_missing_a3": n_missing_a3,
        "n_ambiguous": n_ambiguous,
        "n_review": n_review,
        "output": paths["consensus"],
        "ambiguous_output": paths["ambiguous"],
        "review_output": paths["review"],
    }
    print(
        f"consensed {summary['n_consensus']} {split} comments "
        f"({n_ambiguous} ambiguous, kept on disk per R2b) -> {paths['consensus']}"
    )
    print(f"-> {paths['ambiguous']}")
    print(f"-> {paths['review']} ({n_review} flagged for human review - D39, "
          "still counted as harmful)")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m src.consensus")
    parser.add_argument("--config", default="config/local.yaml")
    parser.add_argument(
        "--split", choices=sorted(SPLITS), default="dev",
        help="dev (default) or heldout. The held-out run happens AFTER `freeze lock`, "
             "writes to its own files, and never overwrites the development consensus.",
    )
    args = parser.parse_args(argv)
    run(load_config(args.config), split=args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
