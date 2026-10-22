"""Reporting and the manual screen check (AGENTS.md §5, R8).

The screen check is editorial verification of what you publish, not annotation. It
accepts ARBITRARY phrases: suggested, rejected, and anything surfaced from Channel E.
Restricting it to the dev-channel outcome list would miss phrases that only appear in
the held-out scene.

--------------------------------------------------------------------------------------
CLI  (build contract §8)

    python -m src.report heldout      --config config/local.yaml
    python -m src.report screen_check --config config/local.yaml --top 15 --samples 20
                                       [--phrase NAME ...]

`screen_check()` and `ScreenCheckItem` above this banner are frozen and MUST NOT
CHANGE (build contract line 12). Everything below is CLI machinery built around them.

R9 - both commands disclose the collection mix (which platforms, and which of those
were collected live vs. imported from an export) and never print a real account name
or figure identifier - on screen a figure is `Figure A`. `Comment` (src/schemas.py) has
no platform/provenance fields and is also frozen, so this module keeps its own
comment_id -> provenance side table instead of extending it.
--------------------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import baseline
from . import consensus as consensus_mod
from . import freeze
from .phrases.candidates import phrase_matches
from .phrases.mine import PhraseScore
from .schemas import Comment


@dataclass(slots=True)
class ScreenCheckItem:
    phrase: str
    verdict: str
    sampled: list[str]          # normalized text only - never author handles (R8)


def screen_check(
    phrases: Iterable[tuple[PhraseScore, Sequence[Comment]]],
    samples_per_phrase: int = 20,
    seed: int = 0,
) -> list[ScreenCheckItem]:
    """Sample matches for every phrase that will appear on camera.

    20 samples catches a phrase whose true precision is 0.70 with probability 0.999
    (0.7**20 = 0.0008). A genuinely 0.97-precise phrase looks clean about half the
    time, which is the expected and acceptable outcome.

    Pass whatever you plan to show: suggested phrases, rejected phrases (jelek is
    demo scene 2), and Channel E phrases. There is no filtering by verdict here.
    """
    rng = random.Random(seed)
    out: list[ScreenCheckItem] = []
    for score, matches in phrases:
        pool = list(matches)
        picked = rng.sample(pool, min(samples_per_phrase, len(pool)))
        out.append(ScreenCheckItem(
            phrase=score.phrase,
            verdict="SUGGESTED" if score.recommend() else "REJECTED",
            sampled=[c.text_normalized or c.text_original for c in picked],
        ))
    return out


# ============================================================================
# CLI machinery below this line. `screen_check`/`ScreenCheckItem` above are frozen
# (build contract §8) and nothing below imports anything they don't already expose.
# ============================================================================

#: R5. Repeated here (matches src/baseline.py) so a figure cannot be lifted out of
#: this module's artefacts unlabelled.
ATTRIBUTION = "according to our automated annotation system"

FAMILIES_JSON = "data/processed/phrase_families.json"
HELDOUT_OUTPUT_JSON = "data/processed/heldout_report.json"
SCREEN_CHECK_JSON = "data/processed/screen_check.json"
SCREEN_CHECK_MD = "data/processed/screen_check.md"

#: A wider column set than `baseline.CORPUS_COLUMNS`'s R8 allowlist. This module needs
#: platform/figure/post columns to build pseudonymized provenance (R9) and the
#: collection-mix disclosure - but see `build_provenance`: real figure/account
#: identifiers are folded into `Figure A`-style labels before anything is written to a
#: display artefact, and `author_hash` (already a one-way hash, never a raw handle) is
#: read only to populate `Comment.author_hash`, which `screen_check()` never emits.
WIDE_CORPUS_COLUMNS: tuple[str, ...] = (
    "comment_id", "text_original", "text_normalized", "split",
    "channel_id", "video_id", "author_hash", "published_at",
    "platform", "figure_id", "source_account", "post_id", "post_url",
    "collection_mode", "provenance",
)

_PLATFORM_DISPLAY = {
    "youtube": "YouTube", "instagram": "Instagram", "x": "X", "tiktok": "TikTok",
}


# ---------------------------------------------------------------------------
# loading - wider than baseline's, for the reasons in WIDE_CORPUS_COLUMNS above
# ---------------------------------------------------------------------------

def _load_wide_corpus(path: Path, split: str | None = None) -> list[dict]:
    """Read whatever of `WIDE_CORPUS_COLUMNS` the corpus actually has.

    An allowlist, same discipline as `baseline.CORPUS_COLUMNS` (R8): a column this
    module does not already know about cannot appear in the rows this function
    returns, no matter what else ingest starts writing.
    """
    import pandas as pd

    df = pd.read_parquet(path)
    if split is not None and "split" in df.columns:
        df = df[df["split"].astype(str) == split]
    cols = [c for c in WIDE_CORPUS_COLUMNS if c in df.columns]
    if "comment_id" not in cols:
        raise SystemExit(f"{path.as_posix()} is missing required column: comment_id")
    df = df[cols]
    out: list[dict] = []
    for row in df.itertuples(index=False):
        out.append({c: baseline._clean(getattr(row, c)) for c in cols})
    return out


def _rows_to_comments(rows: Iterable[Mapping]) -> dict[str, Comment]:
    """Project wide rows onto the frozen `Comment` schema, keyed by comment_id.

    `Comment` predates the multi-platform migration (no platform/figure/post fields),
    so `channel_id`/`video_id` are filled from `figure_id`/`post_id` when the legacy
    names are absent. `screen_check()` only ever reads `.text_normalized`/
    `.text_original` off these objects (R8) - the rest is carried for completeness.
    """
    out: dict[str, Comment] = {}
    for r in rows:
        cid = str(r.get("comment_id") or "")
        if not cid:
            continue
        out[cid] = Comment(
            comment_id=cid,
            channel_id=str(r.get("channel_id") or r.get("figure_id") or ""),
            video_id=str(r.get("video_id") or r.get("post_id") or ""),
            author_hash=str(r.get("author_hash") or ""),
            published_at=str(r.get("published_at") or ""),
            text_original=str(r.get("text_original") or ""),
            text_normalized=str(r.get("text_normalized") or ""),
        )
    return out


def _score_from_buckets(name: str, member_ids: Iterable[str], labels: Mapping[str, str]) -> PhraseScore:
    """Tally the four PhraseScore buckets for a set of comment ids from consensus
    labels alone (`baseline.classify()`'s output), the same bookkeeping
    `src.phrases.mine.score_phrase` does over `ConsensusRecord`s. Used for family
    numbers recomputed in this module (heldout transfer, ad hoc --phrase lookups) so
    they are always built from the SAME member_ids + labels, never a second rule."""
    n_harm = n_non = n_amb = n_crit = 0
    for cid in member_ids:
        bucket = labels.get(str(cid))
        if bucket is None:
            continue
        if bucket == "harmful":
            n_harm += 1
        elif bucket == "ambiguous":
            n_amb += 1
        else:
            n_non += 1
            # D38: mirrors `mine.py::score_phrase` - C counts every COLLATERAL label,
            # not `legitimate_criticism` alone. These two counters must agree, or a
            # phrase's C here would differ from the C that decided its verdict.
            if bucket in ("legitimate_criticism", "public_figure_invective"):
                n_crit += 1
    return PhraseScore(name, n_harm, n_non, n_amb, n_crit)


# ---------------------------------------------------------------------------
# R9 - pseudonymized provenance and the collection-mix disclosure
# ---------------------------------------------------------------------------

def _platform_display(platform: str) -> str:
    key = (platform or "").strip().casefold()
    if not key or key == baseline.UNKNOWN_PLATFORM:
        return "Unknown platform"
    return _PLATFORM_DISPLAY.get(key, key.title())


def _figure_label(index: int) -> str:
    """0->A, 1->B, ..., 25->Z, 26->AA, ... - spreadsheet-column style, so an
    unbounded number of figures still gets a short, stable, content-free label."""
    index += 1
    letters = ""
    while index:
        index, rem = divmod(index - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def _figure_key(row: Mapping) -> str:
    return str(row.get("figure_id") or row.get("channel_id") or "unknown-figure")


def _post_key(row: Mapping) -> str:
    return str(row.get("post_id") or row.get("video_id") or "unknown-post")


def build_provenance(rows: Sequence[Mapping]) -> dict[str, str]:
    """comment_id -> pseudonymized "which post, which app" string (R9/D35).

    Prefers an upstream `provenance` column when ingest has already written one
    (build contract §1 - it is ingest's job to pseudonymize at source). Otherwise
    this derives the same shape itself from whatever figure/post columns exist,
    including the current pre-migration schema (`channel_id`/`video_id` only): real
    figure and post identifiers are mapped to `Figure A`/"post i of N" labels here,
    before anything reaches an output file - never the raw id, regardless of which
    schema generation produced the corpus.
    """
    figures = sorted({_figure_key(r) for r in rows})
    figure_label = {f: _figure_label(i) for i, f in enumerate(figures)}
    posts_by_figure: dict[str, list[str]] = {f: [] for f in figures}
    for r in rows:
        posts_by_figure[_figure_key(r)].append(_post_key(r))
    posts_by_figure = {f: sorted(set(posts)) for f, posts in posts_by_figure.items()}

    out: dict[str, str] = {}
    for r in rows:
        cid = str(r.get("comment_id") or "")
        if not cid:
            continue
        existing = str(r.get("provenance") or "").strip()
        if existing:
            out[cid] = existing
            continue
        f = _figure_key(r)
        posts = posts_by_figure[f]
        idx = posts.index(_post_key(r)) + 1
        out[cid] = (
            f"{_platform_display(str(r.get('platform') or ''))} · "
            f"Figure {figure_label[f]} · post {idx} of {len(posts)}"
        )
    return out


def collection_mix(rows: Sequence[Mapping]) -> dict:
    """R9 - which platforms fed this report, and whether each was collected live
    (`api`/`fixture`) or imported from an export (`import`). Reported from whatever
    columns ingest has actually written; if `platform`/`collection_mode` are not yet
    on disk (the multi-platform migration, build contract §1, may not have landed in
    every lane at once) this says so explicitly rather than inventing a mix - an `x`
    arm built from 400 imported replies does not support the same claim as 9,000 live
    YouTube comments, and that asymmetry belongs on the face of the report.
    """
    platform_available = any(str(r.get("platform") or "").strip() for r in rows)
    mode_available = any(str(r.get("collection_mode") or "").strip() for r in rows)

    counts: dict[str, int] = {}
    modes: dict[str, set[str]] = {}
    for r in rows:
        p = str(r.get("platform") or "").strip().casefold() or baseline.UNKNOWN_PLATFORM
        counts[p] = counts.get(p, 0) + 1
        m = str(r.get("collection_mode") or "").strip().casefold()
        if m:
            modes.setdefault(p, set()).add(m)

    if platform_available:
        note = (
            "comments per platform and, where known, whether each was collected live "
            "or imported from an export (R9) - an uneven corpus is a limitation to "
            "disclose, not to smooth over."
        )
    else:
        note = (
            "the `platform` column is not yet present in this corpus - the ingest "
            "stage has not been migrated to the multi-platform schema (build contract "
            "§1) here yet, so every comment below is counted as a single 'unspecified' "
            "platform rather than a real mix."
        )
    return {
        "platform_data_available": platform_available,
        "collection_mode_data_available": mode_available,
        "comments_per_platform": dict(sorted(counts.items())),
        "collection_mode_per_platform": {p: sorted(m) for p, m in sorted(modes.items())},
        "note": note,
    }


def _round(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(float(value), digits)


def _score_block(
    score: PhraseScore,
    threshold: float = 0.85,
    max_criticism: float = 0.10,
    min_matches: int = 25,
) -> dict:
    """`threshold`/`max_criticism`/`min_matches` must match whatever the DEV verdict
    beside this one was scored with (the frozen lockfile's `recommendations.json`,
    via `heldout_report`'s own `thresholds` block) - comparing a held-out verdict
    computed at this function's bare 0.85 default against a dev verdict computed at
    a different configured threshold (D37 moved the project default to 0.60) would
    silently manufacture "verdict flips" that are really just the two sides being
    scored by different rules, not a transfer failure."""
    return {
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
        "verdict": "SUGGESTED" if score.recommend(threshold, max_criticism, min_matches) else "REJECTED",
    }


# ---------------------------------------------------------------------------
# `heldout` - freeze.verify() first, always (R3)
# ---------------------------------------------------------------------------

def _score_family_over_rows(
    name: str, variants: Sequence[str], rows: Sequence[Mapping], labels: Mapping[str, str],
) -> PhraseScore:
    """Apply one family's variants to `rows` with `phrase_matches` - the SAME
    whole-token matcher `src/phrases/candidates.py` used to build the dev numbers in
    `recommendations.json` - so the dev-vs-heldout comparison differs only in which
    split's data feeds it, never in how a match is defined. A comment counts once if
    ANY variant matches it (the family union rule, D33)."""
    if not variants:
        return PhraseScore(name, 0, 0, 0, 0)
    member_ids = [
        str(r["comment_id"]) for r in rows
        if any(phrase_matches(v, str(r.get("text_normalized") or r.get("text_original") or ""))
               for v in variants)
    ]
    return _score_from_buckets(name, member_ids, labels)


#: How much of the held-out split must carry a consensus label before a transfer number
#: means anything. Not 100%: `src.consensus` legitimately skips a comment whose A3 vote
#: was dropped as off-schema, so a handful of gaps is normal.
MIN_HELDOUT_LABEL_COVERAGE = 0.95


def _load_heldout_labels(root: Path, heldout_rows: Sequence[Mapping]) -> dict[str, str]:
    """Consensus labels for the held-out split, or a refusal.

    THE FAILURE THIS EXISTS TO PREVENT.
    `_score_from_buckets` skips any comment id it has no label for. So if the held-out
    split was never annotated, every family scores n_all=0, every lower bound is 0.0,
    and every SUGGESTED family "flips" to REJECTED. The report then states, confidently
    and in the shape of a real result, that nothing transferred - which is the exact
    opposite of demo scene 4 and is not a finding at all, just absent data.

    Nothing upstream catches it: the freeze is intact, the corpus is present, and the
    arithmetic is correct. So the check belongs here, and it refuses rather than warns.
    """
    dev_path = root / baseline.CONSENSUS_PARQUET
    heldout_path = root / consensus_mod.HELDOUT_CONSENSUS_PATH

    if not heldout_path.exists():
        raise SystemExit(
            f"{heldout_path.as_posix()} not found - the held-out split has not been "
            "annotated.\n\n"
            "The held-out figure must be scored by the FROZEN system, so its annotate "
            "and consensus runs happen AFTER `freeze lock`, not as part of the "
            "development pipeline:\n\n"
            "  python -m src.freeze verify\n"
            "  python -m src.annotate.run --config config/local.yaml --split heldout\n"
            "  python -m src.consensus   --config config/local.yaml --split heldout\n"
            "  python -m src.report heldout --config config/local.yaml\n\n"
            "Those write to their own files and never touch the development votes or "
            f"{dev_path.as_posix()}."
        )

    labels = baseline.load_labels(heldout_path)
    ids = {str(r.get("comment_id")) for r in heldout_rows}
    covered = len(ids & set(labels))
    coverage = covered / len(ids) if ids else 0.0
    if coverage < MIN_HELDOUT_LABEL_COVERAGE:
        raise SystemExit(
            f"{heldout_path.as_posix()} labels only {covered} of {len(ids)} held-out "
            f"comments ({coverage:.1%}).\n\n"
            "Unlabelled matches are silently skipped when a family is scored, so every "
            "family would report a lower bound of 0.0 and appear to have failed to "
            "transfer. That is missing data wearing the costume of a result.\n\n"
            "Re-run the held-out annotate and consensus steps, then try again."
        )
    return labels


def heldout_report(config: Mapping, root: Path = Path(".")) -> dict:
    """Assemble `data/processed/heldout_report.json`. Callers MUST have already
    confirmed `freeze.verify()` returned no drift (`run_heldout` below does this) -
    this function does not check the freeze itself, so it stays a plain, testable
    computation over whatever is on disk.
    """
    root = Path(root)
    normalized_path = baseline._require(root / baseline.NORMALIZED_PARQUET,
                                        "python -m src.normalize.text")
    rec_path = baseline._require(root / baseline.RECOMMENDATIONS_JSON,
                                 "python -m src.phrases.families")

    heldout_rows = _load_wide_corpus(normalized_path, split="heldout")
    if not heldout_rows:
        raise SystemExit(
            f"no split=='heldout' rows in {normalized_path.as_posix()} - the held-out "
            "figure has not been collected. There is nothing to score."
        )

    labels = _load_heldout_labels(root, heldout_rows)
    recommendations = json.loads(rec_path.read_text(encoding="utf-8"))

    # Minimum support is part of the decision rule (R0), and the held-out split is
    # smaller than the development one. So a family can flip to REJECTED purely by
    # having fewer than `min_matches` held-out matches, with a rate nobody measured.
    # Reporting one undifferentiated flip count makes those look like failures to
    # transfer - the opposite of demo scene 4 - so the two causes are counted apart.
    thresholds_block = recommendations.get("thresholds") or {}
    min_matches = int(thresholds_block.get("min_matches", 25))
    # Same thresholds the DEV verdicts beside these were computed with (D37: scoring
    # the two sides against different rules would manufacture fake "verdict flips").
    threshold = float(thresholds_block.get("recommend_threshold", 0.60))
    max_criticism = float(thresholds_block.get("max_criticism_share", 0.10))

    families_out = []
    n_matches = n_flips = 0
    n_flips_support = n_flips_rate = 0
    for fam in recommendations.get("families", []):
        name = str(fam.get("name", ""))
        variants = [str(v) for v in fam.get("variants", []) if str(v).strip()]
        heldout_score = _score_family_over_rows(name, variants, heldout_rows, labels)
        dev_block = {
            "n_all": fam.get("n_all"), "n_harmful": fam.get("n_harmful"),
            "n_non_harmful": fam.get("n_non_harmful"), "n_ambiguous": fam.get("n_ambiguous"),
            "n_criticism": fam.get("n_criticism"),
            "p_harmful_lower": fam.get("p_harmful_lower"), "p_harmful_upper": fam.get("p_harmful_upper"),
            "p_harmful_resolved": fam.get("p_harmful_resolved"),
            "criticism_share": fam.get("criticism_share"), "ambiguity_share": fam.get("ambiguity_share"),
            "verdict": str(fam.get("verdict", "")).upper(),
        }
        heldout_block = _score_block(heldout_score, threshold, max_criticism, min_matches)
        matches = dev_block["verdict"] == heldout_block["verdict"]
        n_matches += matches
        n_flips += not matches

        below_support = heldout_score.n_all < min_matches
        if matches:
            flip_reason = None
        elif below_support:
            flip_reason = "insufficient_heldout_support"
            n_flips_support += 1
        else:
            flip_reason = "different_harm_rate"
            n_flips_rate += 1

        dev_lower = fam.get("p_harmful_lower")
        families_out.append({
            "name": name,
            "variants": variants,
            "dev": dev_block,
            "heldout": heldout_block,
            "transfer": {
                "verdict_matches": matches,
                "flip_reason": flip_reason,
                "heldout_meets_min_support": not below_support,
                "p_harmful_lower_delta": (
                    _round(heldout_score.p_harmful_lower() - dev_lower)
                    if dev_lower is not None else None
                ),
            },
        })

    mix = collection_mix(heldout_rows)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "according_to": ATTRIBUTION,                                      # R5
        "freeze": {"verified": True, "lock_path": str(freeze.LOCK_PATH)},
        "held_out_unit_note": (
            "R9/D35 - the held-out unit is the FIGURE across every platform it posts "
            "to (YouTube, Instagram, X together), not a single channel. Every source "
            "of a held-out figure was held out with it at ingest, so this split is not "
            "one app's comments generalising to another - it is the whole figure, "
            "unseen, across all of them."
        ),
        "collection_mix": mix,                                            # R9
        "n_heldout_comments": len(heldout_rows),
        "n_families": len(families_out),
        "n_verdict_matches": n_matches,
        "n_verdict_flips": n_flips,
        "min_matches": min_matches,
        "flip_reasons": {
            "insufficient_heldout_support": n_flips_support,
            "different_harm_rate": n_flips_rate,
            "note": (
                "A flip is not automatically a failure to transfer. Minimum support is "
                f"part of the decision rule (n_all >= {min_matches}), and the held-out "
                f"split is smaller than the development one ({len(heldout_rows)} "
                "comments), so a family with too few held-out matches is REJECTED "
                "without its rate ever being measured. Only "
                "`different_harm_rate` flips are families that had enough held-out "
                "evidence and behaved differently on it."
            ),
        },
        "families": sorted(families_out, key=lambda f: (-(f["dev"]["n_all"] or 0), f["name"])),
    }


def _print_heldout_summary(payload: Mapping) -> None:
    print(f"heldout report - all figures {ATTRIBUTION}")
    print(f"{payload['n_heldout_comments']} held-out comments, {payload['n_families']} families")
    reasons = payload.get("flip_reasons", {})
    print(f"verdict matches dev: {payload['n_verdict_matches']}   "
          f"flips: {payload['n_verdict_flips']}")
    if reasons:
        print(f"  of those flips: {reasons.get('different_harm_rate', 0)} on a different "
              f"harm rate, {reasons.get('insufficient_heldout_support', 0)} on too few "
              f"held-out matches (< {payload.get('min_matches')}) to judge at all")
    header = f"{'family':<20}{'dev verdict':>14}{'heldout verdict':>18}{'lower delta':>14}"
    print(header)
    print("-" * len(header))
    for fam in payload["families"][:15]:
        delta = fam["transfer"]["p_harmful_lower_delta"]
        print(
            f"{fam['name']:<20}{fam['dev']['verdict']:>14}{fam['heldout']['verdict']:>18}"
            f"{('n/a' if delta is None else f'{delta:+.3f}'):>14}"
        )


def run_heldout(config: Mapping, root: Path = Path(".")) -> int:
    """`python -m src.report heldout` body. R3: `freeze.verify()` runs FIRST, and any
    drift refuses the run before anything else executes - that is the entire point of
    the freeze (AGENTS.md R3, build contract §8)."""
    root = Path(root)
    drifted = freeze.verify(root)                # raises SystemExit if no lockfile yet
    if drifted:
        print("FREEZE VIOLATED - the held-out figure is no longer unseen. These "
              "changed after the lock:", file=sys.stderr)
        for d in drifted:
            print(f"  {d}", file=sys.stderr)
        print("\nEither revert the change, or re-lock deliberately and say so on "
              "camera - do not score Channel E against a system that was touched "
              "after seeing it.", file=sys.stderr)
        return 1

    payload = heldout_report(config, root)
    out_path = root / HELDOUT_OUTPUT_JSON
    baseline.write_json(out_path, payload)
    _print_heldout_summary(payload)
    print(f"\nwrote {out_path.as_posix()}")
    return 0


# ---------------------------------------------------------------------------
# `screen_check` CLI - selection, ad hoc --phrase lookup, provenance, JSON + MD
# ---------------------------------------------------------------------------

def _combined_families(root: Path) -> list[dict]:
    """`phrase_families.json` (variants/member_ids) joined with `recommendations.json`
    (n_all, for ranking) by name. Dev only - the automatic top-N selection never
    reaches Channel E; `--phrase` is the deliberate, named exception (AGENTS.md §5)."""
    fam_path = baseline._require(root / FAMILIES_JSON, "python -m src.phrases.families")
    rec_path = baseline._require(root / baseline.RECOMMENDATIONS_JSON,
                                 "python -m src.phrases.families")
    fam_by_name = {f["name"]: f for f in
                   json.loads(fam_path.read_text(encoding="utf-8")).get("families", [])}
    rec_by_name = {f["name"]: f for f in
                   json.loads(rec_path.read_text(encoding="utf-8")).get("families", [])}
    out = []
    for name, fam in fam_by_name.items():
        member_ids = list(fam.get("member_ids", []))
        rec = rec_by_name.get(name, {})
        out.append({
            "name": name,
            "variants": list(fam.get("variants", [])),
            "member_ids": member_ids,
            "n_all": int(rec.get("n_all", len(member_ids))),
            "verdict": str(rec.get("verdict", "")).upper(),
            "source": "dev",
        })
    return out


def _screen_check_rank(fam: Mapping) -> tuple:
    """Ranking for the automatic top-N.

    Ranking by support alone fills the list with `saya`, `yang`, `kamu` - the most
    common words in the corpus, all obviously REJECTED - and the operator spends the
    25 minutes proofreading stopwords while not one SUGGESTED family, the phrases that
    actually go on screen and actually hide people's comments, is ever looked at.

    So SUGGESTED families come first, then the highest-support rejections (which is
    where `jelek` lives, demo scene 2). This is a change of ORDER, not of membership:
    there is still no verdict filtering (D24), and `--phrase` still reaches anything.
    """
    return (0 if fam.get("verdict") == "SUGGESTED" else 1, -fam["n_all"], fam["name"])


def _mirror_sample(
    pairs: Sequence[tuple[PhraseScore, Sequence[Comment]]], samples_per_phrase: int, seed: int,
) -> list[list[str]]:
    """Recovers which comment_id each `screen_check()` sample came from, WITHOUT
    touching the frozen `screen_check()`/`ScreenCheckItem` (build contract §8: neither
    may change).

    This calls `random.Random(seed).sample(pool, k)` on the exact same `pairs` list,
    in the exact same order, with the exact same per-phrase `k` - i.e. it is the same
    two lines `screen_check()` runs internally, verbatim. `random.sample` selects by
    POSITION, never by the content of the sampled objects, so given identical inputs
    and an identical call sequence it must select the identical positions every time;
    since `pairs` here IS the same list of `Comment` objects passed to `screen_check`
    (not a copy), "the same positions" means the same object references come back.
    `test_screen_check_mirror_matches_frozen_sampling` cross-checks the recovered
    text against `ScreenCheckItem.sampled` byte for byte, so if `screen_check()`'s
    sampling is ever edited without updating this function, that test fails loudly
    instead of provenance silently drifting onto the wrong comment.
    """
    rng = random.Random(seed)
    out: list[list[str]] = []
    for _score, matches in pairs:
        pool = list(matches)
        picked = rng.sample(pool, min(samples_per_phrase, len(pool)))
        out.append([c.comment_id for c in picked])
    return out


def run_screen_check(
    config: Mapping,
    root: Path = Path("."),
    *,
    top: int = 15,
    samples: int = 20,
    seed: int = 0,
    extra_phrases: Sequence[str] = (),
) -> dict:
    root = Path(root)
    # `screen_check()` below is frozen (build contract §8) and always calls
    # `score.recommend()` unparameterized - it can only ever apply this method's own
    # 0.85/0.10/25 defaults, no matter what the config says. That was invisible while
    # the project's real threshold happened to BE 0.85; the moment D37 lowered it to
    # 0.60, `item.verdict` started disagreeing with `recommendations.json`'s own
    # verdict for the exact same family (asu/kntl: SUGGESTED there, REJECTED here).
    # Read the real thresholds here, in the non-frozen CLI layer, and recompute the
    # verdict actually written to the report below instead of trusting `item.verdict`.
    phrases_cfg = dict(config.get("phrases") or {})
    threshold = float(phrases_cfg.get("recommend_threshold", 0.60))
    max_criticism = float(phrases_cfg.get("max_criticism_share", 0.10))
    min_matches = int(phrases_cfg.get("min_matches", 25))

    normalized_path = baseline._require(root / baseline.NORMALIZED_PARQUET,
                                        "python -m src.normalize.text")
    consensus_path = baseline._require(root / baseline.CONSENSUS_PARQUET,
                                       "python -m src.consensus")

    dev_rows = _load_wide_corpus(normalized_path, split="dev")
    labels = baseline.load_labels(consensus_path)
    known = _combined_families(root)

    ranked = sorted(known, key=_screen_check_rank)
    selected: dict[str, dict] = {f["name"]: f for f in ranked[:max(top, 0)]}

    name_index = {f["name"].casefold(): f for f in known}
    variant_index = {
        str(v).casefold(): f for f in known for v in f["variants"]
    }

    heldout_rows: list[dict] = []
    heldout_loaded = False
    for raw in extra_phrases:
        key = raw.strip()
        if not key:
            continue
        hit = name_index.get(key.casefold()) or variant_index.get(key.casefold())
        if hit:
            selected.setdefault(hit["name"], hit)
            continue
        # Not a known dev family: treat NAME as a literal phrase, per AGENTS.md §5
        # ("the screen-check function takes arbitrary phrase ids for exactly this
        # reason") - including a look in Channel E, because "phrases surfaced from
        # Channel E" (D24/D27) are explicitly part of what this command must cover.
        # This is the one deliberate, NAMED exception to dev-only reads in this
        # module; `report.py heldout` is the automatic one (build contract §1).
        if not heldout_loaded:
            heldout_loaded = True
            if normalized_path.exists():
                heldout_rows = _load_wide_corpus(normalized_path, split="heldout")
        pool = dev_rows + heldout_rows
        member_ids = [
            str(r["comment_id"]) for r in pool
            if phrase_matches(key, str(r.get("text_normalized") or r.get("text_original") or ""))
        ]
        selected[key] = {
            "name": key, "variants": [key], "member_ids": member_ids,
            "n_all": len(member_ids),
            "source": f"ad hoc --phrase (dev+heldout scan, {len(member_ids)} matches)",
        }

    all_rows = dev_rows + heldout_rows
    comments_by_id = _rows_to_comments(all_rows)
    provenance = build_provenance(all_rows)

    ordered_names = sorted(selected, key=lambda n: (-selected[n]["n_all"], n))
    pairs: list[tuple[PhraseScore, list[Comment]]] = []
    sources: dict[str, str] = {}
    scores: dict[str, PhraseScore] = {}
    for name in ordered_names:
        fam = selected[name]
        score = _score_from_buckets(name, fam["member_ids"], labels)
        matches = [comments_by_id[cid] for cid in fam["member_ids"] if cid in comments_by_id]
        pairs.append((score, matches))
        sources[name] = fam.get("source", "dev")
        scores[name] = score

    items = screen_check(pairs, samples_per_phrase=samples, seed=seed)
    picked_ids = _mirror_sample(pairs, samples, seed)

    mix = collection_mix(all_rows)
    families_out = []
    for (score, _matches), item, ids in zip(pairs, items, picked_ids):
        samples_out = []
        for i, (text, cid) in enumerate(zip(item.sampled, ids), start=1):
            samples_out.append({
                "n": i,
                "text_normalized": text,                                   # R8
                "provenance": provenance.get(cid, "Unknown platform · Figure ? · post ? of ?"),
                "mark": "",                                                # for the manual pass
            })
        families_out.append({
            "phrase": item.phrase,
            # NOT item.verdict - see the comment at the top of this function. Same
            # `score` object screen_check() itself sampled from, just scored against
            # the real configured thresholds instead of PhraseScore.recommend()'s
            # hardcoded 0.85/0.10/25 defaults.
            "verdict": "SUGGESTED" if score.recommend(threshold, max_criticism, min_matches) else "REJECTED",
            "source": sources.get(item.phrase, "dev"),
            "n_all": score.n_all,
            "identified_range": {
                "lower": _round(score.p_harmful_lower()), "upper": _round(score.p_harmful_upper()),
            },
            "ambiguity_share": _round(score.ambiguity_share),
            "criticism_share": _round(score.criticism_share),
            "n_sampled": len(samples_out),
            "samples": samples_out,
        })

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "according_to": ATTRIBUTION,                                       # R5
        "top": top,
        "samples_per_phrase": samples,
        "seed": seed,
        "collection_mix": mix,                                             # R9
        "note": (
            "editorial verification of what gets published, not annotation "
            "(AGENTS.md §5). Every verdict above is according to our automated "
            "annotation system, never asserted as ground truth - that is what this "
            "manual pass is for. Suggested and rejected families are both included; "
            "there is no verdict filtering."
        ),
        "families": families_out,
    }


def _screen_check_markdown(payload: Mapping) -> str:
    lines: list[str] = []
    lines.append("# Screen check — manual verification (~25 minutes)")
    lines.append("")
    lines.append(f"Generated: {payload['generated_at']}  ")
    lines.append(f"All figures {payload['according_to']}.  ")
    lines.append(
        f"top={payload['top']}  samples_per_phrase={payload['samples_per_phrase']}  "
        f"seed={payload['seed']}"
    )
    lines.append("")
    lines.append("## Collection mix (R9)")
    lines.append("")
    mix = payload["collection_mix"]
    if mix["platform_data_available"]:
        for p, n in mix["comments_per_platform"].items():
            modes = ", ".join(mix["collection_mode_per_platform"].get(p, [])) or "unknown"
            lines.append(f"- **{p}**: {n} comments (collection mode: {modes})")
    else:
        lines.append(f"- {mix['note']}")
    lines.append("")
    lines.append(payload["note"])
    lines.append("")
    for fam in payload["families"]:
        rng = fam["identified_range"]
        lo = "n/a" if rng["lower"] is None else f"{rng['lower']:.0%}"
        hi = "n/a" if rng["upper"] is None else f"{rng['upper']:.0%}"
        amb = "n/a" if fam["ambiguity_share"] is None else f"{fam['ambiguity_share']:.0%}"
        lines.append(f"## `{fam['phrase']}` -> **{fam['verdict']}** ({fam['source']})")
        lines.append("")
        lines.append(f"- matches (n_all): {fam['n_all']}")
        lines.append(f"- identified range: {lo} - {hi}")
        lines.append(f"- ambiguity share: {amb}")
        lines.append("")
        lines.append("| # | comment (normalized text only, R8) | provenance | mark |")
        lines.append("|---|---|---|---|")
        for s in fam["samples"]:
            text = s["text_normalized"].replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {s['n']} | {text} | {s['provenance']} | ☐ |")
        lines.append("")
    return "\n".join(lines)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# argparse dispatch
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m src.report",
        description="Reporting CLI: the held-out run (R3) and the manual screen "
                    "check (AGENTS.md §5).",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    ho = sub.add_parser("heldout", help="score the frozen families against Channel E")
    ho.add_argument("--config", default=None, help="default: config/local.yaml")

    sc = sub.add_parser("screen_check", help="sample matches for the manual pass")
    sc.add_argument("--config", default=None, help="default: config/local.yaml")
    sc.add_argument("--top", type=int, default=15)
    sc.add_argument("--samples", type=int, default=20)
    sc.add_argument("--seed", type=int, default=0)
    sc.add_argument("--phrase", action="append", default=[],
                    help="include this family/phrase regardless of --top; repeatable. "
                         "Looked up in phrase_families.json first, then treated as a "
                         "literal phrase over dev + Channel E (AGENTS.md §5).")

    args = ap.parse_args(argv)
    root = Path(".")

    if args.command == "heldout":
        config = baseline.load_config(args.config)
        return run_heldout(config, root)

    if args.command == "screen_check":
        config = baseline.load_config(args.config)
        payload = run_screen_check(
            config, root, top=args.top, samples=args.samples, seed=args.seed,
            extra_phrases=args.phrase,
        )
        json_path = root / SCREEN_CHECK_JSON
        md_path = root / SCREEN_CHECK_MD
        baseline.write_json(json_path, payload)
        _write_text(md_path, _screen_check_markdown(payload))
        print(f"screen check - all figures {ATTRIBUTION}")
        print(f"{len(payload['families'])} phrases, {payload['samples_per_phrase']} "
              f"samples each")
        for fam in payload["families"]:
            print(f"  {fam['phrase']:<20} {fam['verdict']:<10} n_all={fam['n_all']}")
        print(f"\nwrote {json_path.as_posix()}\nwrote {md_path.as_posix()}")
        return 0

    return 2                                                       # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
