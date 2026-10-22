"""Family discovery (build contract §6). The FAMILY is the recommendation unit (D33).

Candidates are grouped by `canonicalize_variant` - the aggressive fold that collapses
spelling variants (`g*b l 0 kkk` -> `goblok`) - and near-identical keys may then be
merged with rapidfuzz. A phrase that groups with nothing is a **family of size one**,
so the unit stays uniform: everything, singletons included, is scored by
`score_family` over the DEDUPLICATED UNION of its variants' match sets. Nothing here
sums per-variant counts, and nothing here re-implements the recommendation rule -
the verdict comes from `PhraseScore.recommend`.

`canonicalize_variant` is imported lazily, by the exact name pinned in build contract
§2, so that this module can be tested with an injected fold while `src/normalize/text.py`
is still being written.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..schemas import ConsensusRecord
from .candidates import (
    CANDIDATES_PATH,
    CONSENSUS_PATH,
    load_config,
    load_consensus_records,
    load_platform_map,
    write_json,
)
from .families import inflation_factor, score_family
from .mine import PhraseScore, score_phrase

FAMILIES_PATH = "data/processed/phrase_families.json"
RECOMMENDATIONS_PATH = "data/processed/recommendations.json"

#: rapidfuzz similarity (0-100) above which two canonical keys are treated as the same
#: family. Only keys with the SAME token count are ever compared, and at 92 a single
#: edit merges only long keys - a 4-character key with one substitution scores 75, so
#: `mati` and `hati` can never merge. Set `phrases.family_merge_similarity: null` to
#: disable the stage entirely and group on the canonical fold alone.
DEFAULT_MERGE_SIMILARITY = 92.0


@dataclass(slots=True)
class Family:
    name: str
    canonical_key: str
    variants: list[str] = field(default_factory=list)
    member_ids: list[str] = field(default_factory=list)
    inflation_factor: float = 1.0

    @property
    def n_all(self) -> int:
        return len(self.member_ids)


def default_fold() -> Callable[[str], str]:
    """The family-grouping fold, imported by the name pinned in build contract §2.

    Lazy on purpose: `src/normalize/text.py` is a separate lane, and importing it at
    module scope would make `pytest -q` depend on that lane being finished.
    """
    from ..normalize.text import canonicalize_variant

    return canonicalize_variant


def _merge_keys(keys: Sequence[str], similarity: float | None) -> dict[str, str]:
    """Union-find over canonical keys within a small edit distance.

    Returns key -> representative key. Deterministic: keys are visited in sorted
    order and the representative is always the lexicographically smallest member.
    """
    parent = {k: k for k in keys}

    def find(k: str) -> str:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            lo, hi = (ra, rb) if ra < rb else (rb, ra)
            parent[hi] = lo

    if similarity is not None and float(similarity) > 0 and len(keys) > 1:
        from rapidfuzz import fuzz, process

        cutoff = float(similarity)
        buckets: dict[int, list[str]] = {}
        for k in sorted(keys):
            buckets.setdefault(k.count(" ") + 1, []).append(k)
        for bucket in buckets.values():
            if len(bucket) < 2:
                continue
            for key in bucket:
                for other, _score, _idx in process.extract(
                    key, bucket, scorer=fuzz.ratio, score_cutoff=cutoff, limit=None
                ):
                    if other != key:
                        union(key, other)
    return {k: find(k) for k in sorted(parent)}


def group_candidates(
    matches_by_variant: Mapping[str, Sequence[str]],
    fold: Callable[[str], str] | None = None,
    merge_similarity: float | None = DEFAULT_MERGE_SIMILARITY,
) -> list[Family]:
    """Group phrases into families. A phrase in no group is a family of size one.

    `fold` defaults to `canonicalize_variant`; tests inject their own.
    """
    fold = fold or default_fold()
    by_key: dict[str, list[str]] = {}
    for variant in sorted(matches_by_variant):
        by_key.setdefault(fold(variant), []).append(variant)

    representative = _merge_keys(sorted(by_key), merge_similarity)
    merged: dict[str, list[str]] = {}
    for key, variants in by_key.items():
        merged.setdefault(representative[key], []).extend(variants)

    families: list[Family] = []
    for group in merged.values():
        variants = sorted(set(group))
        # Named after its highest-support variant, uppercased; ties broken
        # lexicographically so the artefact is stable across runs (R3).
        head = min(variants, key=lambda v: (-len(set(matches_by_variant.get(v, ()))), v))
        members = {cid for v in variants for cid in matches_by_variant.get(v, ())}
        families.append(Family(
            name=head.upper(),
            canonical_key=fold(head),
            variants=variants,
            member_ids=sorted(members),
            inflation_factor=inflation_factor(variants, matches_by_variant),
        ))
    families.sort(key=lambda f: (-f.n_all, f.name))
    return families


def score_families(
    families: Sequence[Family],
    matches_by_variant: Mapping[str, Sequence[str]],
    records_by_id: Mapping[str, ConsensusRecord],
) -> list[tuple[Family, PhraseScore]]:
    """Every family - singletons included - goes through `score_family` (D33).

    `score_family` recomputes the union itself, so the numbers behind the decision are
    produced by the frozen implementation rather than by anything in this module.
    """
    return [
        (fam, score_family(fam.name, fam.variants, matches_by_variant, records_by_id))
        for fam in families
    ]


def _round(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(float(value), digits)


def platform_support(
    member_ids: Sequence[str],
    records_by_id: Mapping[str, ConsensusRecord],
    platform_map: Mapping[str, str],
) -> dict[str, dict]:
    """The pooled n_all(F) broken down by platform (R9/D35) - reporting only.

    The decision stays on the pooled score; this exists so a family that only
    exists on one app is visible rather than buried in the pooled number. Every
    id in `member_ids` falls into exactly one platform bucket ("unknown" if
    `platform_map` has no entry for it), so summing `n_all` over the returned
    platforms reproduces the pooled `n_all` exactly - the same restriction
    (`if cid in records_by_id`) is applied here as in `PhraseFamily.score`.
    """
    by_platform: dict[str, list[str]] = {}
    for cid in member_ids:
        by_platform.setdefault(platform_map.get(cid, "unknown"), []).append(cid)

    support: dict[str, dict] = {}
    for platform, ids in sorted(by_platform.items()):
        matches = [records_by_id[cid] for cid in sorted(ids) if cid in records_by_id]
        score = score_phrase("", matches)
        support[platform] = {
            "n_all": score.n_all,
            "p_harmful_lower": _round(score.p_harmful_lower()),
        }
    return support


def families_payload(families: Sequence[Family]) -> dict:
    return {
        "families": [
            {
                "name": fam.name,
                "canonical_key": fam.canonical_key,
                "variants": list(fam.variants),
                "member_ids": list(fam.member_ids),
                "inflation_factor": _round(fam.inflation_factor),
            }
            for fam in families
        ]
    }


def recommendations_payload(
    scored: Sequence[tuple[Family, PhraseScore]],
    records_by_id: Mapping[str, ConsensusRecord],
    platform_map: Mapping[str, str],
    threshold: float = 0.85,
    max_criticism: float = 0.10,
    min_matches: int = 25,
) -> dict:
    """The numbers that go on screen.

    `verdict` is `PhraseScore.recommend(...)` with the configured thresholds - the rule
    is never restated here. Every point estimate ships with its interval and the
    ambiguity share, which is the width of that interval (R1).

    `platform_support` (R9/D35) breaks the pooled `n_all(F)` down per platform.
    The decision itself never reads it - `verdict` is computed on the pooled `score`
    alone - because R0 pins the rule at exactly three terms and a per-platform gate
    would be a fourth.

    `decision_rule` is built FROM the passed-in thresholds, not restated as a literal -
    it used to be a hardcoded "L_H >= 0.85 ..." string that stayed put even when the
    real threshold moved (D37 lowered the project default from 0.85 to 0.60), so a
    reader trusting this field would have seen a rule that was not the one actually
    applied. Never hand-write this string again.
    """
    return {
        "decision_rule": f"L_H >= {threshold:g} AND C <= {max_criticism:g} AND n_all >= {min_matches}",
        "thresholds": {
            "recommend_threshold": float(threshold),
            "max_criticism_share": float(max_criticism),
            "min_matches": int(min_matches),
        },
        "families": [
            {
                "name": fam.name,
                "variants": list(fam.variants),
                "n_all": score.n_all,
                "n_harmful": score.n_harmful,
                "n_non_harmful": score.n_non_harmful,
                "n_ambiguous": score.n_ambiguous,
                "n_criticism": score.n_criticism,
                "p_harmful_lower": _round(score.p_harmful_lower()),
                "p_harmful_upper": _round(score.p_harmful_upper()),
                "p_harmful_resolved": _round(score.p_harmful_resolved()),
                "criticism_share": _round(score.criticism_share),
                "ambiguity_share": _round(score.ambiguity_share),
                "inflation_factor": _round(fam.inflation_factor),
                "verdict": (
                    "SUGGESTED"
                    if score.recommend(threshold, max_criticism, min_matches)
                    else "REJECTED"
                ),
                "platform_support": platform_support(fam.member_ids, records_by_id, platform_map),
                # Must pass the SAME thresholds used for `verdict` above - `display()`
                # computes its own verdict internally and silently falls back to
                # 0.85/0.10/25 if not given the real ones (mine.py, D37).
                "display": score.display(threshold, max_criticism, min_matches),
            }
            for fam, score in scored
        ],
    }


def run(config: Mapping, root: str | Path = ".") -> dict:
    """CLI body for `python -m src.phrases.families`."""
    import json

    root = Path(root)
    phrases_cfg = dict(config.get("phrases") or {})
    # 0.60 matches AGENTS.md R0 as of D37 - an operator whose config omits
    # recommend_threshold gets the CURRENT rule, not the superseded 0.85 one.
    threshold = float(phrases_cfg.get("recommend_threshold", 0.60))
    max_criticism = float(phrases_cfg.get("max_criticism_share", 0.10))
    min_matches = int(phrases_cfg.get("min_matches", 25))
    merge_similarity = phrases_cfg.get("family_merge_similarity", DEFAULT_MERGE_SIMILARITY)

    cand_path = root / CANDIDATES_PATH
    if not cand_path.exists():
        raise SystemExit(f"{cand_path} not found - run `python -m src.phrases.mine` first")
    payload = json.loads(cand_path.read_text(encoding="utf-8"))
    matches_by_variant = {
        c["phrase"]: list(c["match_ids"]) for c in payload.get("candidates", [])
    }
    records = load_consensus_records(root / CONSENSUS_PATH)
    platform_map = load_platform_map(root / CONSENSUS_PATH)

    try:
        fold = default_fold()
    except ImportError as exc:                          # pragma: no cover - lane guard
        raise SystemExit(
            "src/normalize/text.py must provide canonicalize_variant "
            f"(build contract §2): {exc}"
        ) from exc

    families = group_candidates(matches_by_variant, fold=fold,
                                merge_similarity=merge_similarity)
    scored = score_families(families, matches_by_variant, records)

    write_json(root / FAMILIES_PATH, families_payload(families))
    recs = recommendations_payload(
        scored, records, platform_map, threshold, max_criticism, min_matches
    )
    write_json(root / RECOMMENDATIONS_PATH, recs)

    suggested = sum(f["verdict"] == "SUGGESTED" for f in recs["families"])
    singletons = sum(len(f.variants) == 1 for f in families)
    summary = {
        "n_variants": len(matches_by_variant),
        "n_families": len(families),
        "n_singletons": singletons,
        "n_suggested": suggested,
        "n_rejected": len(families) - suggested,
    }
    print(
        f"{summary['n_variants']} candidate phrases -> {summary['n_families']} families "
        f"({singletons} singletons); {suggested} SUGGESTED, "
        f"{summary['n_rejected']} REJECTED"
    )
    print(f"-> {FAMILIES_PATH}\n-> {RECOMMENDATIONS_PATH}")
    print("Every phrase that will appear on camera still needs the manual screen check "
          "(AGENTS.md §5), suggested and rejected alike.")
    return summary
