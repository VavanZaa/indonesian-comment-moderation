"""N-gram candidate generation (build contract §5).

The shipped artefact is a YouTube blocked-words list, so matching here is
**whole-token**: `jelek` must not match `jeleknya`. Candidates are therefore built
from token tuples rather than substrings, and the same tuple test is what
`phrase_matches` exposes to callers.

Performance. The corpus is 10k-30k comments and the range is 1..4, so an
O(candidates x comments) rescan is not affordable. Instead this builds an
**inverted index** in one streaming pass per n: each document contributes its
DISTINCT n-grams to a postings dict `ngram -> [document index]`. A comment reading
`goblok banget goblok` posts to `goblok` once, so a posting list is already the
deduplicated match set of that phrase.

On top of that the levels are pruned by downward closure (Apriori): a 3-gram cannot
occur in more documents than either of the 2-grams it contains, so an n-gram is only
counted when both of its (n-1)-subgrams already cleared `min_matches`. Level 1 is the
only unpruned level, and levels 2..4 stay small. Total work is
O(sum_n tokens) set operations rather than O(candidates x comments); the loop also
stops early once a level has no survivors, since no longer n-gram can have any.

n-grams do not cross punctuation or emoji: `goblok. mending mati` yields no
`goblok mending`. Emoji are kept in the pipeline text (they carry tone) but they end
a phrase rather than forming one.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..schemas import ConsensusRecord, Label, Tier
from .mine import PhraseScore, score_phrase

#: repo-relative, posix-style (build contract §0)
NORMALIZED_PATH = "data/interim/comments_normalized.parquet"
CONSENSUS_PATH = "data/processed/consensus.parquet"
CANDIDATES_PATH = "data/processed/phrase_candidates.json"

#: A token is a run of word characters. Anything else (punctuation, emoji) is a
#: phrase boundary, so n-grams never span it.
_TOKEN = re.compile(r"\w+", re.UNICODE)
_BOUNDARY = re.compile(r"[^\w\s]+", re.UNICODE)


@dataclass(slots=True)
class Candidate:
    """One surviving n-gram and the deduplicated set of comments it matched."""
    phrase: str
    n: int
    match_ids: list[str]

    @property
    def n_all(self) -> int:
        return len(self.match_ids)


# --- tokenization -----------------------------------------------------------

def segments(text: str) -> list[list[str]]:
    """Token runs that an n-gram may span, split at punctuation and emoji."""
    return [toks for part in _BOUNDARY.split(text or "") if (toks := _TOKEN.findall(part))]


def tokenize(text: str) -> list[str]:
    """Flat whole-token view of a text. `jeleknya` is one token, not `jelek` + suffix."""
    return _TOKEN.findall(text or "")


def phrase_matches(phrase: str, text: str) -> bool:
    """Whole-token containment test - the same rule candidate generation uses.

    >>> phrase_matches("jelek", "videonya jelek banget")
    True
    >>> phrase_matches("jelek", "jeleknya keliatan")
    False
    """
    needle = tokenize(phrase)
    if not needle:
        return False
    k = len(needle)
    return any(
        seg[i:i + k] == needle
        for seg in segments(text)
        for i in range(len(seg) - k + 1)
    )


# --- candidate generation ---------------------------------------------------

def generate_candidates(
    documents: Sequence[tuple[str, str]],
    ngram_range: tuple[int, int] = (1, 4),
    min_matches: int = 25,
) -> list[Candidate]:
    """Inverted-index n-gram mining over `(comment_id, text_normalized)` pairs.

    Keeps candidates with `n_all >= min_matches`, where `n_all` is the number of
    DISTINCT comments matched - a comment saying the phrase twice counts once, which
    is the same union discipline families use (D32).
    """
    lo, hi = int(ngram_range[0]), int(ngram_range[1])
    lo = max(1, lo)
    ids = [cid for cid, _ in documents]
    docs = [segments(text) for _, text in documents]

    out: list[Candidate] = []
    survivors: set[tuple[str, ...]] | None = None
    for n in range(lo, hi + 1):
        postings: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for doc_idx, doc in enumerate(docs):
            seen: set[tuple[str, ...]] = set()
            for seg in doc:
                for i in range(len(seg) - n + 1):
                    gram = tuple(seg[i:i + n])
                    if gram in seen:
                        continue            # already posted: dedupe within the comment
                    if survivors is not None and (
                        gram[:-1] not in survivors or gram[1:] not in survivors
                    ):
                        continue            # downward closure: cannot clear min_matches
                    seen.add(gram)
                    postings[gram].append(doc_idx)
        kept = {gram for gram, posting in postings.items() if len(posting) >= min_matches}
        for gram in sorted(kept):
            out.append(Candidate(
                phrase=" ".join(gram),
                n=n,
                match_ids=sorted(ids[i] for i in postings[gram]),
            ))
        survivors = kept
        if not kept:
            break                            # no longer n-gram can survive either
    return out


def score_candidates(
    candidates: Iterable[Candidate],
    records_by_id: Mapping[str, ConsensusRecord],
) -> list[tuple[Candidate, PhraseScore]]:
    """Score every survivor through the existing `score_phrase` (R1/R2).

    Consensus tiers are NOT weighted here: HIGH and MEDIUM are equally resolved
    observations and AMBIGUOUS is unresolved. `score_phrase` is the only place the
    bookkeeping lives.
    """
    scored = []
    for cand in candidates:
        matches = [records_by_id[cid] for cid in cand.match_ids if cid in records_by_id]
        scored.append((cand, score_phrase(cand.phrase, matches)))
    return scored


def candidates_payload(
    scored: Sequence[tuple[Candidate, PhraseScore]],
    ngram_range: tuple[int, int],
    min_matches: int,
) -> dict:
    """The `phrase_candidates.json` body. `match_ids` is required: §6 needs the raw
    match sets to build each family's deduplicated union."""
    return {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "ngram_range": [int(ngram_range[0]), int(ngram_range[1])],
        "min_matches": int(min_matches),
        "candidates": [
            {
                "phrase": cand.phrase,
                "n": cand.n,
                "match_ids": list(cand.match_ids),
                "n_harmful": score.n_harmful,
                "n_non_harmful": score.n_non_harmful,
                "n_ambiguous": score.n_ambiguous,
                "n_criticism": score.n_criticism,
            }
            for cand, score in sorted(
                scored, key=lambda pair: (-pair[0].n_all, pair[0].phrase)
            )
        ],
    }


# --- shared I/O -------------------------------------------------------------

def load_config(path: str | Path) -> dict:
    """Read a pipeline config. Missing file is an operator error, not a default."""
    import yaml

    p = Path(path)
    if not p.exists():
        raise SystemExit(f"config not found: {p} (copy config/poc.yaml to config/local.yaml)")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def write_json(path: str | Path, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_consensus_records(path: str | Path) -> dict[str, ConsensusRecord]:
    """Rebuild `ConsensusRecord`s from `consensus.parquet`.

    `label` is null for the ambiguous tier and stays None - never coerced to a class.
    `weight` is carried for completeness and is TRAINING ONLY (R2); nothing in phrase
    mining reads it.
    """
    import pandas as pd

    p = Path(path)
    if not p.exists():
        raise SystemExit(f"{p} not found - run `python -m src.consensus` first")
    frame = pd.read_parquet(p)
    records: dict[str, ConsensusRecord] = {}
    for row in frame.itertuples(index=False):
        raw_label = getattr(row, "label", None)
        label = None if raw_label is None or pd.isna(raw_label) else Label(str(raw_label))
        votes_json = getattr(row, "votes_json", None)
        votes = json.loads(votes_json) if isinstance(votes_json, str) and votes_json else {}
        records[str(row.comment_id)] = ConsensusRecord(
            comment_id=str(row.comment_id),
            label=label,
            tier=Tier(str(row.tier)),
            weight=float(getattr(row, "weight", 0.0) or 0.0),
            votes=votes,
            corroboration=int(getattr(row, "corroboration", 0) or 0),
        )
    return records


def load_platform_map(path: str | Path) -> dict[str, str]:
    """`comment_id -> platform`, read straight off `consensus.parquet` (R9/D35).

    `src.consensus` carries `platform` through from the normalized comments, so
    this is the one place family grouping needs to look for it. A missing column -
    an artefact written before R9 landed - degrades to an empty map rather than
    crashing family grouping: `platform_support` is a display breakdown, not part
    of the (pooled) decision rule, so every comment simply falls back to "unknown".
    """
    import pandas as pd

    p = Path(path)
    if not p.exists():
        raise SystemExit(f"{p} not found - run `python -m src.consensus` first")
    frame = pd.read_parquet(p)
    if "platform" not in frame.columns:
        return {}
    return {
        str(row.comment_id): ("unknown" if pd.isna(row.platform) else str(row.platform))
        for row in frame.itertuples(index=False)
    }


def load_documents(
    path: str | Path, keep_ids: Iterable[str] | None = None
) -> list[tuple[str, str]]:
    """`(comment_id, text_normalized)` for dev rows only.

    Build contract §1: nothing downstream of ingest may read `split == "heldout"`
    except `src.report heldout`. Phrase discovery is dev-only (R3).
    """
    import pandas as pd

    p = Path(path)
    if not p.exists():
        raise SystemExit(f"{p} not found - run `python -m src.normalize.text` first")
    frame = pd.read_parquet(p)
    if "split" in frame.columns:
        frame = frame[frame["split"] == "dev"]
    wanted = None if keep_ids is None else set(keep_ids)
    docs: list[tuple[str, str]] = []
    for row in frame.itertuples(index=False):
        cid = str(row.comment_id)
        if wanted is not None and cid not in wanted:
            continue
        text = getattr(row, "text_normalized", "")
        docs.append((cid, "" if text is None or pd.isna(text) else str(text)))
    return docs


def run(config: Mapping, root: str | Path = ".") -> dict:
    """CLI body for `python -m src.phrases.mine`. Returns a summary for the caller."""
    root = Path(root)
    phrases_cfg = dict(config.get("phrases") or {})
    ngram_range = tuple(phrases_cfg.get("ngram_range") or (1, 4))
    min_matches = int(phrases_cfg.get("min_matches", 25))

    records = load_consensus_records(root / CONSENSUS_PATH)
    # Mine only over comments that HAVE a consensus record, so that the surviving
    # `n_all` is exactly `len(match_ids)` and the min-support filter means what it says.
    documents = load_documents(root / NORMALIZED_PATH, keep_ids=records.keys())

    candidates = generate_candidates(documents, ngram_range, min_matches)
    scored = score_candidates(candidates, records)
    payload = candidates_payload(scored, ngram_range, min_matches)
    out_path = root / CANDIDATES_PATH
    write_json(out_path, payload)

    summary = {
        "n_comments": len(documents),
        "n_candidates": len(candidates),
        "min_matches": min_matches,
        "ngram_range": [int(ngram_range[0]), int(ngram_range[1])],
        "output": CANDIDATES_PATH,
    }
    print(
        f"mined {summary['n_candidates']} candidates "
        f"(n-grams {summary['ngram_range'][0]}..{summary['ngram_range'][1]}, "
        f"min_matches={min_matches}) over {summary['n_comments']} dev comments "
        f"-> {CANDIDATES_PATH}"
    )
    print("Candidates are not recommendations: the unit of decision is the FAMILY (D33). "
          "Run `python -m src.phrases.families` next.")
    return summary
