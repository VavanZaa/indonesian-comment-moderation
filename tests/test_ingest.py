"""Tests for `src/ingest/fixture.py` and `src/ingest/__init__.py` - the seeded offline
corpus (build contract §1, AGENTS.md R3/R8/R9/D35).

Deliberately split into two tiers:

  - Most tests exercise `generate_records` (and the small pure helpers around it)
    directly. That function has NO import-time dependency on `src.ingest.base` - a
    different lane owns that module and may still be writing it - so this tier passes
    regardless of whether `base.py` exists yet.
  - A handful of integration tests need `base.build_row` / `write_corpus` /
    `write_dev_manifest` to check the real on-disk contract (schema, hashing, the R3/R9
    dev-manifest leak). They go through the `all_rows` fixture below, which
    `pytest.importorskip`s `src.ingest.base` - those tests SKIP (not fail) while that
    module is still being written elsewhere in the repo, and exercise the real path once
    it lands.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from src.ingest.fixture import (
    HARMFUL_INTENTS,
    PLATFORMS,
    SCENE_PROBES,
    SPLIT_DEV,
    SPLIT_HELDOUT,
    _author_hash_salt,
    _figure_alias_map,
    _figures_from_config,
    _load_config,
    build,
    generate_records,
    goblok_platform_harmful_share,
    scene_report,
    to_rows,
)

# The operator's local trial is currently three YouTube-only dev figures. Synthetic
# scene tests need the versioned five-figure demo template, independent of that trial.
CONFIG_PATH = "config/poc.yaml"

#: The exact on-disk schema, build contract §1. Hardcoded here (not imported from
#: `src.ingest.base`) so this file's schema check does not depend on that module's
#: existence or internal column ordering - only on the contract everyone builds against.
CONTRACT_COLUMNS = frozenset({
    "comment_id", "platform", "figure_id", "source_account", "post_id", "post_url",
    "parent_id", "author_hash", "published_at", "text_original", "split", "genre",
    "collection_mode", "provenance", "like_count",
})


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def cfg():
    return _load_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def all_records(cfg):
    return generate_records(cfg, seed=0)


@pytest.fixture(scope="module")
def dev_records(all_records):
    return [r for r in all_records if r.split == SPLIT_DEV]


@pytest.fixture(scope="module")
def heldout_records(all_records):
    return [r for r in all_records if r.split == SPLIT_HELDOUT]


@pytest.fixture(scope="module")
def all_rows(cfg, all_records):
    """Base-dependent. Any test using this fixture SKIPS cleanly if `src.ingest.base`
    is not importable yet, rather than failing the whole suite."""
    pytest.importorskip("src.ingest.base")
    return to_rows(all_records, cfg)


# --------------------------------------------------------------------------- config / figures parsing
def test_config_figures_block_shape(cfg):
    figures = _figures_from_config(cfg)
    dev = [f for f in figures if f["split"] == SPLIT_DEV]
    heldout = [f for f in figures if f["split"] == SPLIT_HELDOUT]
    assert len(dev) == 4, "D6/build contract: 4 dev figures"
    assert len(heldout) == 1
    for f in figures:
        assert 2 <= len(f["sources"]) <= 3, f"{f['id']} should have 2-3 platform sources"
        assert {s["platform"] for s in f["sources"]} <= set(PLATFORMS)


def test_figures_from_config_raises_on_split_conflict_R9():
    """R9: a figure may not have sources in both splits; ingest must raise."""
    bad_cfg = {
        "figures": [
            {"id": "figure_x", "genre": "news", "split": "dev",
             "sources": [{"platform": "youtube", "account": "a"}]},
            {"id": "figure_x", "genre": "news", "split": "heldout",
             "sources": [{"platform": "instagram", "account": "b"}]},
        ]
    }
    with pytest.raises(RuntimeError, match="R9"):
        _figures_from_config(bad_cfg)


def test_figure_alias_map_pseudonymizes_dev_then_heldout_in_order():
    figures = _figures_from_config({
        "figures": [
            {"id": "figure_a", "split": "dev", "genre": "x",
             "sources": [{"platform": "youtube", "account": "a"}]},
            {"id": "figure_b", "split": "dev", "genre": "x",
             "sources": [{"platform": "youtube", "account": "b"}]},
            {"id": "figure_e", "split": "heldout", "genre": "x",
             "sources": [{"platform": "youtube", "account": "e"}]},
        ]
    })
    alias = _figure_alias_map(figures)
    assert alias == {"figure_a": "Figure A", "figure_b": "Figure B", "figure_e": "Figure C"}


def test_author_hash_salt_required():
    with pytest.raises(SystemExit):
        _author_hash_salt({"privacy": {}})


# --------------------------------------------------------------------------- determinism (D6)
def test_fixture_is_deterministic(cfg):
    def key(r):
        return (r.platform, r.figure_id, r.post_id, r.native_id, r.parent_native_id,
                r.author_id, r.published_at, r.text, r.group, r.intent, r.split)

    a = generate_records(cfg, seed=0)
    b = generate_records(cfg, seed=0)
    assert [key(r) for r in a] == [key(r) for r in b]


def test_different_seed_changes_output(cfg):
    a = generate_records(cfg, seed=0)
    b = generate_records(cfg, seed=1)
    assert [r.text for r in a] != [r.text for r in b]


# --------------------------------------------------------------------------- corpus shape (D6)
def test_corpus_size_within_d6_target(all_records):
    assert 10_000 <= len(all_records) <= 30_000


def test_split_partitions_by_figure(all_records):
    """No figure may appear in both splits."""
    splits_by_figure: dict[str, set[str]] = {}
    for r in all_records:
        splits_by_figure.setdefault(r.figure_id, set()).add(r.split)
    for fid, splits in splits_by_figure.items():
        assert len(splits) == 1, f"figure {fid} appears in both splits: {splits}"


def test_no_heldout_ids_leak_into_dev_by_construction_R3_R9(all_records):
    """R3/R9 leak precursor: the fixture's own generated ids must never collide across
    splits within a platform. The real dev-manifest leak test lives in the
    base-dependent tier below."""
    dev_ids = {(r.platform, r.native_id) for r in all_records if r.split == SPLIT_DEV}
    heldout_ids = {(r.platform, r.native_id) for r in all_records if r.split == SPLIT_HELDOUT}
    assert dev_ids.isdisjoint(heldout_ids)


# --------------------------------------------------------------------------- platforms (R9/D35)
def test_all_three_platforms_present_in_dev(dev_records):
    assert {r.platform for r in dev_records} == set(PLATFORMS)


def test_x_comments_always_open_with_at_handle(dev_records):
    x_rows = [r for r in dev_records if r.platform == "x"]
    assert x_rows
    assert all(r.text.startswith("@") for r in x_rows)


def test_x_non_scene_comments_are_clipped_short(dev_records):
    """'short and clipped' (R9): the core message (handles stripped back off) is capped
    for anything that is not carrying a scene-critical phrase."""
    x_filler = [r for r in dev_records if r.platform == "x" and r.group == "filler"]
    assert x_filler
    for r in x_filler:
        core_words = [w for w in r.text.split(" ") if not w.startswith("@")]
        assert len(core_words) <= 5


def test_instagram_comments_are_hashtag_and_emoji_heavy(dev_records):
    ig_rows = [r for r in dev_records if r.platform == "instagram"]
    assert ig_rows
    assert all("#" in r.text for r in ig_rows)


def test_youtube_comments_are_unclipped_and_have_no_at_handle(dev_records):
    yt_filler = [r for r in dev_records if r.platform == "youtube" and r.group == "filler"]
    assert yt_filler
    assert all(not r.text.startswith("@") for r in yt_filler)
    assert any(len(r.text.split(" ")) > 5 for r in yt_filler), "youtube keeps the longer prose"


def test_goblok_family_harmful_rate_differs_by_platform(dev_records):
    """R9: at least one family must have a genuinely different harmful rate on one
    platform, so `platform_support` has something real to report."""
    shares = goblok_platform_harmful_share(dev_records, split=SPLIT_DEV)
    assert set(shares) == set(PLATFORMS)
    assert max(shares.values()) - min(shares.values()) >= 0.10, shares


def test_reply_chains_present_and_well_formed(dev_records):
    """Reply chains via parent_id: a reply must point at a real, earlier comment on the
    same post."""
    native_by_post: dict[tuple[str, str], dict[str, str]] = {}
    for r in dev_records:
        native_by_post.setdefault((r.platform, r.post_id), {})[r.native_id] = r.published_at

    replies = [r for r in dev_records if r.parent_native_id is not None]
    assert replies, "expected some comments to be generated as replies"
    for r in replies:
        siblings = native_by_post[(r.platform, r.post_id)]
        assert r.parent_native_id in siblings, "parent must be a real comment on the same post"
        assert r.published_at >= siblings[r.parent_native_id], "reply must not precede its parent"


# --------------------------------------------------------------------------- demo scenes
def test_scene_1_mending_mati_high_harm_zero_criticism(dev_records):
    group = [r for r in dev_records if r.group == "death_wish"]
    assert group
    harmful = sum(1 for r in group if r.intent in HARMFUL_INTENTS)
    criticism = sum(1 for r in group if r.intent == "legitimate_criticism")
    assert harmful / len(group) > 0.85
    assert criticism == 0
    texts = " ".join(r.text.lower() for r in group)
    assert "mending mati" in texts


def test_f2_condemnation_of_mending_mati_present(dev_records):
    """docs/04-architecture-mapping.md finding F2: a comment condemning the phrase while
    containing it must be findable in the screen check."""
    texts = " ".join(r.text.lower() for r in dev_records if r.group == "death_wish")
    assert 'bilang "mending mati"' in texts


def test_scene_probes_present_in_dev(all_records):
    scenes = scene_report(all_records)
    for key in SCENE_PROBES:
        assert scenes[SPLIT_DEV][key] > 0, f"missing dev probe: {key}"


def test_scene_2_jelek_mostly_criticism_not_recommended(dev_records):
    group = [r for r in dev_records if r.group == "jelek"]
    assert group
    n = len(group)
    n_harmful = sum(1 for r in group if r.intent in HARMFUL_INTENTS)
    n_criticism = sum(1 for r in group if r.intent == "legitimate_criticism")
    lower = n_harmful / n
    criticism_share = n_criticism / n
    assert criticism_share > 0.10, "must fail the criticism gate - this is the whole scene"
    # roughly reproduces docs/04-architecture-mapping.md §6: jelek lower ~= 0.172
    assert 0.05 <= lower <= 0.30


def test_scene_3_goblok_character_variants_all_present(dev_records):
    texts = " ".join(r.text.lower() for r in dev_records if r.group == "goblok")
    for variant in ("goblok", "g0blok", "goblokkk", "g*blok", "g o b l o k"):
        assert variant in texts, f"missing goblok variant: {variant}"


def test_scene_4_heldout_family_transfers_in_different_surface_wording(heldout_records):
    assert heldout_records
    texts = " ".join(r.text.lower() for r in heldout_records if r.group == "goblok")
    for variant in ("g0bl0k", "goblokk", "g#blok", "g.o.b.l.o.k"):
        assert variant in texts, f"missing held-out goblok obfuscation: {variant}"

    death = [r for r in heldout_records if r.group == "death_wish"]
    assert death
    harmful = sum(1 for r in death if r.intent in HARMFUL_INTENTS)
    assert harmful / len(death) > 0.85, "the family must transfer, not merely exist"
    assert "mending mati" in " ".join(r.text.lower() for r in death)


def test_scene_5_ambiguous_reel(dev_records):
    ingroup = [r for r in dev_records if r.group == "ingroup"]
    assert ingroup
    assert all(r.intent == "ambiguous" for r in ingroup)
    texts = " ".join(r.text.lower() for r in dev_records)
    assert "wkwkwk goblok lu" in texts


def test_anjir_worked_example_ratio_matches_mine_py(dev_records):
    """src/phrases/mine.py PhraseScore docstring: 40% harmful / 10% criticism /
    50% ambiguous -> lower 0.400, resolved 0.800, upper 0.900."""
    group = [r for r in dev_records if r.group == "anjir"]
    n = len(group)
    assert n > 0
    counts = {
        i: sum(1 for r in group if r.intent == i)
        for i in ("harassment_personal", "legitimate_criticism", "ambiguous")
    }
    assert counts["harassment_personal"] / n == pytest.approx(0.40, abs=0.02)
    assert counts["legitimate_criticism"] / n == pytest.approx(0.10, abs=0.02)
    assert counts["ambiguous"] / n == pytest.approx(0.50, abs=0.02)


# --------------------------------------------------------------------------- base-dependent (schema, R8, R3/R9 manifest)
def test_output_schema_matches_contract(all_rows):
    assert all_rows
    for row in all_rows[:1000]:
        assert set(row) == CONTRACT_COLUMNS


def test_collection_mode_is_fixture(all_rows):
    assert all(row["collection_mode"] == "fixture" for row in all_rows)


def test_r8_author_hash_is_not_raw_and_is_16_hex_chars(all_records, all_rows):
    raw_ids = {r.author_id for r in all_records}
    for row in all_rows[:1000]:
        assert row["author_hash"] not in raw_ids
        assert len(row["author_hash"]) == 16
        int(row["author_hash"], 16)  # must actually be hex


def test_r8_author_hash_matches_documented_algorithm(cfg, all_records, all_rows):
    """sha256(salt + author_id).hexdigest()[:16] - build contract §1."""
    from src.ingest import base

    salt = _author_hash_salt(cfg)
    sample = all_records[0]
    sample_comment_id = base.make_comment_id(sample.platform, sample.native_id)
    row = next(r for r in all_rows if r["comment_id"] == sample_comment_id)
    expected = hashlib.sha256((salt + sample.author_id).encode("utf-8")).hexdigest()[:16]
    assert row["author_hash"] == expected


def test_no_raw_handle_survives_into_any_output_row(all_records, all_rows):
    """R8, broader sweep: no output row's text/author_hash should ever equal a raw
    synthetic author id."""
    raw_ids = {r.author_id for r in all_records}
    for row in all_rows[:1000]:
        assert row["author_hash"] not in raw_ids
        assert row["text_original"] not in raw_ids


def test_parent_id_references_a_real_comment_id(all_rows):
    by_id = {row["comment_id"] for row in all_rows}
    replies = [row for row in all_rows if row["parent_id"] is not None]
    assert replies
    assert all(row["parent_id"] in by_id for row in replies)


def test_write_dev_manifest_has_no_heldout_ids_R3_R9_leak_test(tmp_path):
    """The R3/R9 leak test, named clearly: the frozen dev manifest must never contain a
    held-out figure, even though the same collection pass produces both splits."""
    pytest.importorskip("src.ingest.base")
    rows, _records = build(config=CONFIG_PATH, seed=0, root=str(tmp_path))
    manifest_path = tmp_path / "data" / "processed" / "dev_manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    dev_figure_ids = {r["figure_id"] for r in rows if r["split"] == SPLIT_DEV}
    heldout_figure_ids = {r["figure_id"] for r in rows if r["split"] == SPLIT_HELDOUT}
    manifest_figure_ids = {f["id"] for f in manifest.get("figures", [])}

    assert manifest_figure_ids, "manifest must list at least one figure"
    assert manifest_figure_ids.isdisjoint(heldout_figure_ids), (
        "R3/R9 violation: a held-out figure id reached the frozen dev manifest"
    )
    assert manifest_figure_ids <= dev_figure_ids
    assert manifest["n_comments"] == sum(1 for r in rows if r["split"] == SPLIT_DEV)


def test_build_writes_real_parquet_matching_contract_schema(tmp_path):
    pytest.importorskip("src.ingest.base")
    import pandas as pd

    rows, _records = build(config=CONFIG_PATH, seed=0, root=str(tmp_path))
    raw_path = tmp_path / "data" / "raw" / "comments_raw.parquet"
    assert raw_path.is_file()

    df = pd.read_parquet(raw_path)
    assert set(df.columns) == CONTRACT_COLUMNS
    assert len(df) == len(rows)
    assert 10_000 <= len(df) <= 30_000


def test_build_is_byte_identical_across_runs_same_seed(tmp_path):
    pytest.importorskip("src.ingest.base")
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    build(config=CONFIG_PATH, seed=0, root=str(root_a))
    build(config=CONFIG_PATH, seed=0, root=str(root_b))
    a_bytes = (root_a / "data" / "raw" / "comments_raw.parquet").read_bytes()
    b_bytes = (root_b / "data" / "raw" / "comments_raw.parquet").read_bytes()
    assert a_bytes == b_bytes
