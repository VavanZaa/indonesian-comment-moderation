"""Tests for `src/ingest/base.py` and the three platform connectors.

AGENTS.md R8 (author hashing) and R9 (platform provenance, the figure-level split) are
the two rules this lane exists to enforce. NO NETWORK CALLS: every "success" API path
is exercised through an injected fake client, and every real-credential path is
exercised through the "credentials missing" skip - `googleapiclient`/`requests` are
never imported by these tests, matching build contract §0 (no SDK installed).
"""
from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from src.ingest import base, collect, instagram, tiktok, x, youtube

SALT = "test-salt-2026"


# --------------------------------------------------------------------------- helpers
def _fig(fid="figure_a", split="dev", genre="entertainment", sources=None):
    return {
        "id": fid,
        "genre": genre,
        "split": split,
        "sources": sources or [{"platform": "youtube", "account": "UC_A", "mode": "import"}],
    }


def _cfg(figures, **extra):
    cfg = {"privacy": {"author_hash_salt": SALT}, "figures": figures}
    cfg.update(extra)
    return cfg


def _row(native_id="c1", **overrides):
    defaults = dict(
        platform="youtube",
        figure_id="figure_a",
        source_account="acc",
        post_id="v1",
        post_url="https://www.youtube.com/watch?v=v1",
        parent_id=None,
        author_id="raw_author_123",
        salt=SALT,
        published_at="2026-01-01T00:00:00Z",
        text_original="hello",
        split="dev",
        genre="entertainment",
        collection_mode="api",
        provenance="YouTube · Figure A · post 1 of 1",
    )
    defaults.update(overrides)
    row = base.build_row(**defaults)
    row["comment_id"] = base.make_comment_id(defaults["platform"], native_id)
    return row


# =============================================================================
# schema (build contract §1)
# =============================================================================
def test_columns_match_build_contract():
    assert base.COLUMNS == (
        "comment_id", "platform", "figure_id", "source_account", "post_id", "post_url",
        "parent_id", "author_hash", "published_at", "text_original", "split", "genre",
        "collection_mode", "provenance", "like_count",
    )


def test_build_row_produces_exactly_the_contract_columns_minus_comment_id():
    row = _row()
    assert set(row) == set(base.COLUMNS)


def test_comment_id_is_platform_prefixed():
    assert base.make_comment_id("youtube", "abc123") == "youtube:abc123"
    assert base.make_comment_id("instagram", "17895") == "instagram:17895"
    assert base.make_comment_id("x", "999") == "x:999"
    row = _row(native_id="native77", platform="youtube")
    assert row["comment_id"] == "youtube:native77"


def test_make_comment_id_rejects_unknown_platform_and_empty_native_id():
    with pytest.raises(ValueError):
        base.make_comment_id("reddit", "1")  # not a supported platform
    with pytest.raises(ValueError):
        base.make_comment_id("youtube", "")
    with pytest.raises(ValueError):
        base.make_comment_id("youtube", None)


def test_tiktok_is_a_recognized_platform():
    """Added 2026-09-08, import-only - see src/ingest/tiktok.py."""
    assert "tiktok" in base.PLATFORMS
    assert base.make_comment_id("tiktok", "789") == "tiktok:789"


def test_validate_frame_rejects_wrong_columns():
    df = pd.DataFrame([{"comment_id": "youtube:1"}])  # every other column missing
    with pytest.raises(ValueError, match="schema violation"):
        base.validate_frame(df)


def test_validate_frame_rejects_unknown_platform_and_split_and_mode():
    row = _row(native_id="c1")
    row["platform"] = "reddit"  # not a supported platform (tiktok is, as of 2026-09-08)
    df = pd.DataFrame([row])
    with pytest.raises(ValueError, match="platform"):
        base.validate_frame(df)


def test_validate_frame_rejects_duplicate_comment_id():
    r1 = _row(native_id="dupe")
    r2 = _row(native_id="dupe")
    df = pd.DataFrame([r1, r2])
    with pytest.raises(ValueError, match="unique"):
        base.validate_frame(df)


def test_validate_frame_accepts_a_well_formed_frame():
    rows = [_row(native_id="c1"), _row(native_id="c2", parent_id="youtube:c1")]
    df = pd.DataFrame(rows).reindex(columns=list(base.COLUMNS))
    base.validate_frame(df)  # must not raise


# =============================================================================
# R8 - author hashing
# =============================================================================
def test_r8_hash_author_is_deterministic_and_salted():
    h1 = base.hash_author("raw_handle_1", SALT)
    h2 = base.hash_author("raw_handle_1", SALT)
    h3 = base.hash_author("raw_handle_2", SALT)
    h4 = base.hash_author("raw_handle_1", "different-salt")
    assert h1 == h2
    assert h1 != h3
    assert h1 != h4
    assert len(h1) == 16


def test_r8_hash_author_requires_a_salt():
    with pytest.raises(ValueError):
        base.hash_author("someone", "")


def test_r8_build_row_never_returns_the_raw_author_id():
    row = base.build_row(
        platform="youtube", figure_id="figure_a", source_account="acc",
        post_id="v1", post_url="https://y/v1", parent_id=None,
        author_id="THE_RAW_HANDLE_MUST_NOT_SURVIVE", salt=SALT,
        published_at="2026-01-01T00:00:00Z", text_original="hi",
        split="dev", genre="entertainment", collection_mode="api",
        provenance="YouTube · Figure A · post 1 of 1",
    )
    assert "author_id" not in row
    assert row["author_hash"] == base.hash_author("THE_RAW_HANDLE_MUST_NOT_SURVIVE", SALT)
    assert "THE_RAW_HANDLE_MUST_NOT_SURVIVE" not in json.dumps(row, default=str)


def test_r8_no_raw_handle_reaches_the_written_parquet(tmp_path):
    row = _row(native_id="c1", author_id="RAW_HANDLE_XYZ")
    out = tmp_path / "comments_raw.parquet"
    base.write_corpus([row], out)
    df = pd.read_parquet(out)
    blob = df.to_csv(index=False)
    assert "RAW_HANDLE_XYZ" not in blob
    assert df.loc[0, "author_hash"] == base.hash_author("RAW_HANDLE_XYZ", SALT)


# =============================================================================
# R9 - split is per FIGURE, and provenance is pseudonymized
# =============================================================================
def test_r9_load_figures_raises_on_split_leak():
    """The single most important test in this lane (AGENTS.md R9 / D35): one
    figure_id must never be declared under both splits, because holding out one
    platform while training on the same person's other platform leaks the held-out
    set through the figure, invisibly to the freeze."""
    figures = [
        {"id": "figure_x", "genre": "entertainment", "split": "dev",
         "sources": [{"platform": "youtube", "account": "chan1", "mode": "import"}]},
        {"id": "figure_x", "genre": "entertainment", "split": "heldout",
         "sources": [{"platform": "instagram", "account": "acct1", "mode": "import"}]},
    ]
    with pytest.raises(base.SplitLeakError):
        base.load_figures(_cfg(figures))


def test_r9_load_figures_merges_same_figure_same_split():
    figures = [
        {"id": "figure_x", "genre": "entertainment", "split": "dev",
         "sources": [{"platform": "youtube", "account": "chan1", "mode": "import"}]},
        {"id": "figure_x", "genre": "entertainment", "split": "dev",
         "sources": [{"platform": "instagram", "account": "acct1", "mode": "import"}]},
    ]
    out = base.load_figures(_cfg(figures))
    assert len(out) == 1
    assert {s["platform"] for s in out[0]["sources"]} == {"youtube", "instagram"}


def test_r9_validate_frame_catches_split_leak_in_assembled_corpus():
    """Defense in depth: even if a leak slipped past config parsing, the assembled
    frame is checked again before anything is written to disk."""
    r1 = _row(native_id="c1", figure_id="figure_x", split="dev")
    r2 = _row(native_id="c2", figure_id="figure_x", split="heldout")
    df = pd.DataFrame([r1, r2]).reindex(columns=list(base.COLUMNS))
    with pytest.raises(base.SplitLeakError):
        base.validate_frame(df)


def test_r9_load_figures_requires_the_figures_block():
    with pytest.raises(ValueError, match="figures"):
        base.load_figures({"privacy": {"author_hash_salt": SALT}})


def test_r9_provenance_string_format():
    assert base.provenance_string("youtube", "Figure A", 4, 15) == "YouTube · Figure A · post 4 of 15"
    assert base.provenance_string("instagram", "Figure B", 1, 1) == "Instagram · Figure B · post 1 of 1"
    assert base.provenance_string("tiktok", "Figure C", 1, 2) == "TikTok · Figure C · post 1 of 2"
    assert base.provenance_string("x", "Figure C", 2, 3) == "X · Figure C · post 2 of 3"


def test_r9_provenance_for_posts_numbers_in_order():
    prov = base.provenance_for_posts("youtube", "Figure A", ["v1", "v2", "v3"])
    assert prov["v1"].endswith("post 1 of 3")
    assert prov["v2"].endswith("post 2 of 3")
    assert prov["v3"].endswith("post 3 of 3")


def test_r9_figure_aliases_contain_no_account_names():
    figures = [
        _fig("figure_a", sources=[{"platform": "youtube", "account": "SECRET_CHANNEL", "mode": "import"}]),
        _fig("figure_b", split="heldout",
             sources=[{"platform": "x", "account": "SECRET_HANDLE", "mode": "import"}]),
    ]
    aliases = base.build_figure_aliases(figures)
    assert aliases == {"figure_a": "Figure A", "figure_b": "Figure B"}
    blob = json.dumps(aliases)
    assert "SECRET_CHANNEL" not in blob
    assert "SECRET_HANDLE" not in blob


def test_r9_dev_manifest_excludes_heldout_ids(tmp_path):
    dev_row = _row(native_id="dev1", figure_id="figure_a", split="dev")
    heldout_row = _row(native_id="held1", figure_id="figure_e", split="heldout")
    df = pd.DataFrame([dev_row, heldout_row]).reindex(columns=list(base.COLUMNS))

    figures = [
        _fig("figure_a", split="dev"),
        _fig("figure_e", split="heldout"),
    ]
    manifest_path = tmp_path / "dev_manifest.json"
    base.write_dev_manifest(df, _cfg(figures), manifest_path, collected_at="2026-09-07T00:00:00Z")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["n_comments"] == 1
    blob = json.dumps(payload)
    assert "youtube:held1" not in blob
    assert "figure_e" not in blob  # the held-out figure itself must not appear either
    assert payload["figures"][0]["id"] == "figure_a"


# =============================================================================
# timestamp normalization - three native formats, one UTC ISO-8601 shape
# =============================================================================
def test_normalize_timestamp_youtube_format():
    assert base.normalize_timestamp("2026-05-01T12:34:56Z") == "2026-05-01T12:34:56Z"


def test_normalize_timestamp_instagram_offset_format():
    assert base.normalize_timestamp("2026-05-01T12:34:56+0000") == "2026-05-01T12:34:56Z"
    assert base.normalize_timestamp("2026-05-01T19:34:56+0700") == "2026-05-01T12:34:56Z"


def test_normalize_timestamp_x_millis_format():
    assert base.normalize_timestamp("2026-05-01T12:34:56.000Z") == "2026-05-01T12:34:56Z"


def test_normalize_timestamp_epoch_seconds_and_millis():
    seconds = base.normalize_timestamp(1_777_984_496)
    from_string = base.normalize_timestamp("1777984496")
    millis = base.normalize_timestamp(1_777_984_496_000)
    assert seconds == from_string == millis
    assert seconds.endswith("Z") and len(seconds) == len("2026-05-05T05:34:56Z")


def test_normalize_timestamp_rejects_empty():
    with pytest.raises(ValueError):
        base.normalize_timestamp("")
    with pytest.raises(ValueError):
        base.normalize_timestamp(None)


# =============================================================================
# import mode - one export file per platform, mapped correctly
# =============================================================================
def _write_import(tmp_path, platform, account, rows):
    d = tmp_path / "data" / "raw" / "imports" / platform
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{account}.json").write_text(json.dumps(rows), encoding="utf-8")


def test_import_mode_maps_youtube_export(tmp_path):
    _write_import(tmp_path, "youtube", "UC_A", [
        {"id": "yt1", "post_id": "vid1", "text": "top level comment",
         "author_id": "raw_channel_1", "published_at": "2026-05-01T12:34:56Z"},
        {"id": "yt2", "post_id": "vid1", "parent_id": "yt1", "text": "a reply",
         "author_id": "raw_channel_2", "published_at": "2026-05-01T12:35:10Z"},
    ])
    fig = _fig("figure_a", sources=[{"platform": "youtube", "account": "UC_A", "mode": "import"}])
    aliases = {"figure_a": "Figure A"}
    rows = youtube.collect(_cfg([fig]), figures=[fig], aliases=aliases, root=tmp_path)

    assert len(rows) == 2
    by_id = {r["comment_id"]: r for r in rows}
    top = by_id["youtube:yt1"]
    reply = by_id["youtube:yt2"]
    assert top["parent_id"] is None
    assert reply["parent_id"] == "youtube:yt1"
    assert top["author_hash"] == base.hash_author("raw_channel_1", SALT)
    assert top["published_at"] == "2026-05-01T12:34:56Z"
    assert top["collection_mode"] == "import"
    assert top["split"] == "dev"
    assert "Figure A" in top["provenance"]
    assert "post 1 of 1" in top["provenance"]
    assert "raw_channel_1" not in json.dumps(rows)


def test_import_mode_maps_instagram_export(tmp_path):
    _write_import(tmp_path, "instagram", "acct_a", [
        {"id": "ig1", "post_id": "media1", "text": "nice pic",
         "author_id": "raw_ig_user", "timestamp": "2026-05-01T12:34:56+0000"},
    ])
    fig = _fig("figure_a", sources=[{"platform": "instagram", "account": "acct_a", "mode": "import"}])
    aliases = {"figure_a": "Figure A"}
    rows = instagram.collect(_cfg([fig]), figures=[fig], aliases=aliases, root=tmp_path)

    assert len(rows) == 1
    row = rows[0]
    assert row["comment_id"] == "instagram:ig1"
    assert row["published_at"] == "2026-05-01T12:34:56Z"
    assert row["author_hash"] == base.hash_author("raw_ig_user", SALT)
    assert row["collection_mode"] == "import"
    assert "raw_ig_user" not in json.dumps(rows)


def test_import_mode_maps_x_export(tmp_path):
    _write_import(tmp_path, "x", "handle_b", [
        {"id": "x1", "post_id": "conv1", "text": "a reply tweet",
         "author_id": "raw_x_user", "created_at": "2026-05-01T12:34:56.000Z"},
    ])
    fig = _fig("figure_b", sources=[{"platform": "x", "account": "handle_b", "mode": "import"}])
    aliases = {"figure_b": "Figure A"}
    rows = x.collect(_cfg([fig]), figures=[fig], aliases=aliases, root=tmp_path)

    assert len(rows) == 1
    row = rows[0]
    assert row["comment_id"] == "x:x1"
    assert row["published_at"] == "2026-05-01T12:34:56Z"
    assert row["author_hash"] == base.hash_author("raw_x_user", SALT)
    assert row["collection_mode"] == "import"
    assert "raw_x_user" not in json.dumps(rows)


def test_import_mode_maps_tiktok_export(tmp_path):
    _write_import(tmp_path, "tiktok", "creator_a", [
        {"id": "tt1", "post_id": "vid1", "text": "a comment",
         "author_id": "raw_tt_user", "create_time": "2026-05-01T12:34:56Z"},
        {"id": "tt2", "post_id": "vid1", "reply_id": "tt1", "text": "a reply",
         "author_id": "raw_tt_user_2", "create_time": "2026-05-01T12:35:10Z"},
    ])
    fig = _fig("figure_a", sources=[{"platform": "tiktok", "account": "creator_a", "mode": "import"}])
    aliases = {"figure_a": "Figure A"}
    rows = tiktok.collect(_cfg([fig]), figures=[fig], aliases=aliases, root=tmp_path)

    assert len(rows) == 2
    by_id = {r["comment_id"]: r for r in rows}
    top = by_id["tiktok:tt1"]
    reply = by_id["tiktok:tt2"]
    assert top["parent_id"] is None
    assert reply["parent_id"] == "tiktok:tt1"
    assert top["author_hash"] == base.hash_author("raw_tt_user", SALT)
    assert top["published_at"] == "2026-05-01T12:34:56Z"
    assert top["collection_mode"] == "import"
    assert "raw_tt_user" not in json.dumps(rows)


def test_import_mode_with_no_export_file_skips_quietly(tmp_path, capsys):
    fig = _fig("figure_a", sources=[{"platform": "youtube", "account": "NO_SUCH_ACCOUNT", "mode": "import"}])
    rows = youtube.collect(_cfg([fig]), figures=[fig], aliases={"figure_a": "Figure A"}, root=tmp_path)
    assert rows == []
    assert "no export file found" in capsys.readouterr().err


# =============================================================================
# per-source collection limits (2026-09-09)
# =============================================================================
def test_per_source_max_videos_overrides_the_channel_wide_default():
    """Channels differ enormously in how deep you must go before the comments are
    there - a Shorts-heavy channel needs its whole back catalogue to match what a
    commentary channel yields in 60 uploads. The override is per SOURCE so raising
    it for one figure cannot silently re-expand every other."""
    seen = {}

    class _CountingClient:
        def list_uploaded_video_ids(self, channel_id, max_videos):
            seen[channel_id] = max_videos
            return []

        def list_comment_threads(self, video_id, max_results):
            return iter(())

    figs = [
        _fig("deep", sources=[{"platform": "youtube", "account": "UC_DEEP",
                               "mode": "api", "max_videos": 600}]),
        _fig("shallow", sources=[{"platform": "youtube", "account": "UC_SHALLOW", "mode": "api"}]),
    ]
    cfg = _cfg(figs, youtube={"api_key": "k", "max_videos_per_channel": 60})
    youtube.collect(cfg, figures=base.load_figures(cfg), aliases={}, client=_CountingClient())

    assert seen["UC_DEEP"] == 600, "per-source override must win"
    assert seen["UC_SHALLOW"] == 60, "every other source keeps the channel-wide default"


def test_per_source_limits_survive_load_figures():
    """`load_figures` rebuilds each source dict from an allowlist, so a new key has to
    be threaded through deliberately - it was silently dropped before this."""
    cfg = _cfg([_fig("f", sources=[{"platform": "youtube", "account": "UC_A", "mode": "api",
                                    "max_videos": 600, "max_comments": 1234}])])
    [fig] = base.load_figures(cfg)
    assert fig["sources"][0]["max_videos"] == 600
    assert fig["sources"][0]["max_comments"] == 1234


def test_per_source_limits_reject_nonsense_values():
    cfg = _cfg([_fig("f", sources=[{"platform": "youtube", "account": "UC_A",
                                    "mode": "api", "max_videos": 0}])])
    with pytest.raises(ValueError, match="max_videos"):
        base.load_figures(cfg)


# =============================================================================
# missing credentials - a clear skip, never a crash
# =============================================================================
def test_youtube_missing_api_key_skips_without_crashing(capsys):
    fig = _fig("figure_a", sources=[{"platform": "youtube", "account": "UC_A", "mode": "api"}])
    cfg = _cfg([fig], youtube={"api_key": "PUT_YOUR_KEY_HERE"})
    rows = youtube.collect(cfg, figures=[fig], aliases={"figure_a": "Figure A"})
    assert rows == []
    assert "youtube.api_key" in capsys.readouterr().err


def test_instagram_missing_login_skips_without_crashing(capsys):
    fig = _fig("figure_a", sources=[{"platform": "instagram", "account": "acct_a", "mode": "api"}])
    cfg = _cfg([fig])  # no instagram: block at all
    rows = instagram.collect(cfg, figures=[fig], aliases={"figure_a": "Figure A"})
    assert rows == []
    assert "login_username" in capsys.readouterr().err


class _TikTokStartupFailure:
    async def start(self):
        raise RuntimeError("browser executable is missing")


def test_tiktok_startup_failure_skips_cleanly(capsys):
    fig = _fig("figure_a", sources=[{"platform": "tiktok", "account": "creator_a", "mode": "api"}])
    rows = tiktok.collect(
        _cfg([fig]), figures=[fig], aliases={"figure_a": "Figure A"},
        client=_TikTokStartupFailure(),
    )
    assert rows == []
    assert "startup" in capsys.readouterr().err


def test_x_missing_bearer_token_skips_without_crashing(capsys):
    fig = _fig("figure_a", sources=[{"platform": "x", "account": "handle_a", "mode": "api"}])
    cfg = _cfg([fig], x={"bearer_token": ""})
    rows = x.collect(cfg, figures=[fig], aliases={"figure_a": "Figure A"})
    assert rows == []
    assert "x.bearer_token" in capsys.readouterr().err


# =============================================================================
# api mode, success path - injected fake clients (no network, no SDK)
# =============================================================================
class _FakeYouTubeClient:
    def list_uploaded_video_ids(self, channel_id, max_videos):
        return ["v1"][:max_videos]

    def list_comment_threads(self, video_id, max_results):
        yield {
            "id": "t1",
            "snippet": {"topLevelComment": {"id": "c1", "snippet": {
                "authorChannelId": {"value": "raw_author_x"},
                "publishedAt": "2026-05-01T12:00:00Z",
                "textOriginal": "top level comment",
            }}},
            "replies": {"comments": [{"id": "c2", "snippet": {
                "authorChannelId": {"value": "raw_author_y"},
                "publishedAt": "2026-05-01T12:05:00Z",
                "textOriginal": "a reply",
            }}]},
        }


def test_youtube_api_mode_success_with_fake_client():
    fig = _fig("figure_a", sources=[{"platform": "youtube", "account": "UC_A", "mode": "api"}])
    cfg = _cfg([fig], youtube={"api_key": "real-looking-key"})
    rows = youtube.collect(cfg, figures=[fig], aliases={"figure_a": "Figure A"}, client=_FakeYouTubeClient())

    assert len(rows) == 2
    by_id = {r["comment_id"]: r for r in rows}
    assert by_id["youtube:c1"]["parent_id"] is None
    assert by_id["youtube:c2"]["parent_id"] == "youtube:c1"
    assert by_id["youtube:c1"]["author_hash"] == base.hash_author("raw_author_x", SALT)
    assert "raw_author_x" not in json.dumps(rows)
    assert by_id["youtube:c1"]["collection_mode"] == "api"


class _FakeInstagramClient:
    def list_media(self, account, max_media):
        assert account == "target_creator"
        return [{"id": "991", "permalink": "https://www.instagram.com/p/ABC123/"}][:max_media]

    def list_comments(self, media_id, max_results):
        assert media_id == "991"
        yield {
            "id": "ig_top", "text": "top level", "author_id": "raw_ig_author",
            "timestamp": "2026-05-01T12:00:00+0000", "parent_id": None,
        }
        if max_results > 1:
            yield {
                "id": "ig_reply", "text": "a reply", "author_id": "raw_ig_replier",
                "timestamp": "2026-05-01T12:01:00+0000", "parent_id": "ig_top",
            }


def test_instagram_instaloader_mode_success_with_fake_client():
    fig = _fig(
        "figure_a",
        sources=[{"platform": "instagram", "account": "target_creator", "mode": "api"}],
    )
    cfg = _cfg([fig], instagram={"max_posts_per_account": 3, "max_comments_total": 10})
    rows = instagram.collect(
        cfg, figures=[fig], aliases={"figure_a": "Figure A"},
        client=_FakeInstagramClient(),
    )

    assert len(rows) == 2
    by_id = {row["comment_id"]: row for row in rows}
    assert by_id["instagram:ig_top"]["parent_id"] is None
    assert by_id["instagram:ig_reply"]["parent_id"] == "instagram:ig_top"
    assert by_id["instagram:ig_top"]["post_id"] == "991"
    assert by_id["instagram:ig_top"]["post_url"].endswith("/ABC123/")
    assert by_id["instagram:ig_top"]["author_hash"] == base.hash_author("raw_ig_author", SALT)
    assert by_id["instagram:ig_top"]["collection_mode"] == "scrape"
    assert "raw_ig_author" not in json.dumps(rows)


class _FakeTikTokClient:
    def __init__(self):
        self.started = False

    async def start(self):
        self.started = True

    async def list_videos(self, account, max_videos, video_refs):
        assert self.started
        assert account == "creator_a"
        assert video_refs == []
        yield {"id": "7106686413101468970", "url": "https://www.tiktok.com/@creator_a/video/7106686413101468970"}

    async def list_comments(self, video_id, max_results, include_replies):
        assert video_id == "7106686413101468970"
        assert include_replies is True
        yield {
            "id": "tt_top", "text": "top tiktok comment", "author_id": "raw_tt_author",
            "create_time": 1_777_984_496, "parent_id": None,
        }
        if max_results > 1:
            yield {
                "id": "tt_reply", "text": "reply tiktok comment", "author_id": "raw_tt_replier",
                "create_time": 1_777_984_556, "parent_id": "tt_top",
            }


def test_tiktok_api_mode_success_with_fake_client():
    fig = _fig(
        "figure_a",
        sources=[{"platform": "tiktok", "account": "creator_a", "mode": "api"}],
    )
    cfg = _cfg(
        [fig],
        tiktok={"max_videos_per_account": 3, "max_comments_total": 10, "include_replies": True},
    )
    rows = tiktok.collect(
        cfg, figures=[fig], aliases={"figure_a": "Figure A"}, client=_FakeTikTokClient(),
    )

    assert len(rows) == 2
    by_id = {row["comment_id"]: row for row in rows}
    assert by_id["tiktok:tt_top"]["parent_id"] is None
    assert by_id["tiktok:tt_reply"]["parent_id"] == "tiktok:tt_top"
    assert by_id["tiktok:tt_top"]["post_id"] == "7106686413101468970"
    assert by_id["tiktok:tt_top"]["author_hash"] == base.hash_author("raw_tt_author", SALT)
    assert by_id["tiktok:tt_top"]["collection_mode"] == "scrape"
    assert "raw_tt_author" not in json.dumps(rows)


class _FakeTikTokConfiguredVideos(_FakeTikTokClient):
    async def list_videos(self, account, max_videos, video_refs):
        assert video_refs == ["https://www.tiktok.com/@creator_a/video/7106686413101468970"]
        yield {"id": "7106686413101468970", "url": video_refs[0]}


def test_tiktok_configured_video_urls_bypass_profile_discovery():
    fig = _fig(
        "figure_a",
        sources=[{"platform": "tiktok", "account": "creator_a", "mode": "api"}],
    )
    cfg = _cfg([fig], tiktok={
        "video_ids_by_account": {
            "creator_a": ["https://www.tiktok.com/@creator_a/video/7106686413101468970"]
        },
        "max_comments_total": 1,
    })
    rows = tiktok.collect(
        cfg, figures=[fig], aliases={"figure_a": "Figure A"},
        client=_FakeTikTokConfiguredVideos(),
    )
    assert [row["comment_id"] for row in rows] == ["tiktok:tt_top"]


class _QuotaExceededClient:
    def list_uploaded_video_ids(self, channel_id, max_videos):
        return ["v1", "v2"]

    def list_comment_threads(self, video_id, max_results):
        raise youtube.QuotaExceeded("quotaExceeded")


def test_youtube_quota_exceeded_stops_cleanly_and_keeps_partial_results(capsys):
    fig = _fig("figure_a", sources=[{"platform": "youtube", "account": "UC_A", "mode": "api"}])
    cfg = _cfg([fig], youtube={"api_key": "real-looking-key"})
    rows = youtube.collect(cfg, figures=[fig], aliases={"figure_a": "Figure A"}, client=_QuotaExceededClient())
    assert rows == []  # nothing collected before the quota error, but no crash
    assert "quotaExceeded" in capsys.readouterr().err


class _CommentsDisabledThenOkClient:
    def list_uploaded_video_ids(self, channel_id, max_videos):
        return ["disabled_video", "ok_video"]

    def list_comment_threads(self, video_id, max_results):
        if video_id == "disabled_video":
            raise youtube.CommentsDisabled("commentsDisabled")
        yield {
            "id": "t1",
            "snippet": {"topLevelComment": {"id": "ok1", "snippet": {
                "authorChannelId": {"value": "raw_ok_author"},
                "publishedAt": "2026-05-01T12:00:00Z",
                "textOriginal": "still collected",
            }}},
        }


def test_youtube_comments_disabled_on_one_video_does_not_stop_the_others(capsys):
    fig = _fig("figure_a", sources=[{"platform": "youtube", "account": "UC_A", "mode": "api"}])
    cfg = _cfg([fig], youtube={"api_key": "real-looking-key"})
    rows = youtube.collect(
        cfg, figures=[fig], aliases={"figure_a": "Figure A"}, client=_CommentsDisabledThenOkClient()
    )
    assert len(rows) == 1
    assert rows[0]["comment_id"] == "youtube:ok1"
    assert "comments disabled" in capsys.readouterr().err


# =============================================================================
# end-to-end: collect.py fans out across platforms and splits, in import mode
# =============================================================================
def test_collect_end_to_end_writes_all_three_artefacts(tmp_path):
    _write_import(tmp_path, "youtube", "UC_A", [
        {"id": "yt1", "post_id": "vid1", "text": "top", "author_id": "raw_a",
         "published_at": "2026-05-01T12:00:00Z"},
        {"id": "yt2", "post_id": "vid1", "parent_id": "yt1", "text": "reply", "author_id": "raw_b",
         "published_at": "2026-05-01T12:01:00Z"},
    ])
    _write_import(tmp_path, "instagram", "acct_a", [
        {"id": "ig1", "post_id": "media1", "text": "insta comment", "author_id": "raw_c",
         "timestamp": "2026-05-01T12:02:00+0000"},
    ])
    _write_import(tmp_path, "x", "handle_b", [
        {"id": "x1", "post_id": "conv1", "text": "x reply", "author_id": "raw_d",
         "created_at": "2026-05-01T12:03:00.000Z"},
    ])

    figures = [
        {"id": "figure_a", "genre": "entertainment", "split": "dev", "sources": [
            {"platform": "youtube", "account": "UC_A", "mode": "import"},
            {"platform": "instagram", "account": "acct_a", "mode": "import"},
        ]},
        {"id": "figure_b", "genre": "news", "split": "heldout", "sources": [
            {"platform": "x", "account": "handle_b", "mode": "import"},
        ]},
    ]
    cfg = _cfg(figures)

    rows, out_figures, aliases = collect.collect(cfg, root=tmp_path)
    assert len(rows) == 4  # 2 youtube + 1 instagram (dev) + 1 x (heldout)
    assert aliases == {"figure_a": "Figure A", "figure_b": "Figure B"}

    df = base.write_collection(rows, cfg, tmp_path)
    assert len(df) == 4
    assert list(df.columns) == list(base.COLUMNS)

    raw_path = tmp_path / base.RAW_PATH
    manifest_path = tmp_path / base.DEV_MANIFEST_PATH
    aliases_path = tmp_path / base.FIGURE_ALIASES_PATH
    assert raw_path.is_file() and manifest_path.is_file() and aliases_path.is_file()

    on_disk = pd.read_parquet(raw_path)
    assert len(on_disk) == 4
    assert set(on_disk["split"]) == {"dev", "heldout"}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["n_comments"] == 3  # dev only: yt1, yt2, ig1 - not x1
    manifest_blob = json.dumps(manifest)
    assert "x:x1" not in manifest_blob
    assert "figure_b" not in manifest_blob
    assert manifest["platforms"] == {"instagram": 1, "youtube": 2}

    alias_payload = json.loads(aliases_path.read_text(encoding="utf-8"))
    assert alias_payload == {"figure_a": "Figure A", "figure_b": "Figure B"}
    alias_blob = json.dumps(alias_payload)
    for secret in ("UC_A", "acct_a", "handle_b"):
        assert secret not in alias_blob

    # every provenance string is pseudonymized - never a raw account name
    for prov in on_disk["provenance"]:
        assert "UC_A" not in prov and "acct_a" not in prov and "handle_b" not in prov
        assert prov.startswith(("YouTube", "Instagram", "X"))


def test_collect_rejects_unknown_platform_name():
    fig = _fig("figure_a")
    with pytest.raises(ValueError):
        collect.collect(_cfg([fig]), platforms=["reddit"])  # not a supported connector
