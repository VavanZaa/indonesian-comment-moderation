"""Network-free regression tests for actual adapter behavior and corpus protection."""
import asyncio
from datetime import datetime
from types import SimpleNamespace as NS

import pandas as pd
import pytest

from src.ingest import base, instagram, tiktok
from src.ingest.scraper_utils import write_result
from tests.test_connectors import _cfg, _fig, _FakeInstagramClient, _FakeTikTokClient


def config_for(platform, **settings):
    account = "target_creator" if platform == "instagram" else "creator_a"
    return _cfg([_fig(sources=[{"platform": platform, "account": account, "mode": "scrape"}])],
                **{platform: settings})


@pytest.mark.parametrize("platform,client", [
    ("instagram", _FakeInstagramClient()), ("tiktok", _FakeTikTokClient()),
])
def test_hard_cap_even_if_provider_overdelivers(platform, client):
    module = instagram if platform == "instagram" else tiktok
    rows = module.collect(config_for(platform, max_comments_total=1), client=client)
    assert len(rows) == 1
    assert rows[0]["collection_mode"] == "scrape"
    base.validate_frame(pd.DataFrame(rows).reindex(columns=base.COLUMNS))


@pytest.mark.parametrize("platform,client", [
    ("instagram", _FakeInstagramClient()), ("tiktok", _FakeTikTokClient()),
])
def test_zero_budget_does_not_start_or_fetch(platform, client):
    module = instagram if platform == "instagram" else tiktok
    assert module.collect(config_for(platform, max_comments_total=0), client=NS()) == []


def test_instagram_native_comments_replies_and_no_overfetch():
    client = instagram._RealInstagramClient.__new__(instagram._RealInstagramClient)
    client._config = {"include_replies": True}
    owner = NS(userid=123, username="RAW_USER")
    reply = NS(id=2, text="reply", owner=owner, created_at_utc=datetime(2026, 1, 1))

    class Comment:
        id, text, created_at_utc = 1, "hello", datetime(2026, 1, 1)
        @property
        def owner(self):
            return owner
        @property
        def answers(self):
            accesses.append("answers")
            return iter([reply])

    accesses = []
    client._posts = {"55": NS(get_comments=lambda: iter([Comment()]))}
    one = list(client.list_comments("55", 1))
    assert len(one) == 1 and accesses == []
    both = list(client.list_comments("55", 2))
    assert both[1]["parent_id"] == "1"
    assert both[1]["author_id"] == 123


def test_instagram_partial_error_keeps_rows_and_does_not_log_secrets(capsys):
    class Broken(_FakeInstagramClient):
        def list_comments(self, *args):
            yield from super().list_comments(*args)
            raise RuntimeError("session=SECRET_COOKIE raw_author=PRIVATE")
    rows = instagram.collect(config_for("instagram"), client=Broken())
    assert len(rows) == 2
    log = capsys.readouterr().err
    assert "RuntimeError" in log and "SECRET_COOKIE" not in log and "PRIVATE" not in log


def test_tiktok_reply_pagination_uses_video_id_and_detects_repeated_cursor():
    requests = []
    class API:
        async def make_request(self, **kwargs):
            requests.append(kwargs)
            return {"comments": [{"cid": str(len(requests))}], "has_more": True, "cursor": 0}
    client = tiktok._RealTikTokClient({})
    client._api = API()
    async def run():
        rows = []
        with pytest.raises(RuntimeError, match="repeated"):
            async for row in client._replies("7106686413101468970", "comment1", 50):
                rows.append(row)
        return rows
    assert asyncio.run(run()) == [{"cid": "1"}]
    assert requests[0]["params"]["item_id"] == "7106686413101468970"


def test_tiktok_native_adapter_reply_mapping_and_cap():
    top = NS(id="1", text="top", as_dict={"cid": "1", "text": "top", "user": {"uid": "RAW"},
                "create_time": 1700000000, "reply_comment_total": 3})
    class Video:
        async def comments(self, **kwargs):
            yield top
            raise AssertionError("must not fetch after cap")
    class API:
        def video(self, **kwargs):
            return Video()
        async def make_request(self, **kwargs):
            return {"comments": [{"cid": "2", "text": "reply", "user": {"uid": "RAW2"},
                                  "create_time": 1700000001}], "has_more": False}
    client = tiktok._RealTikTokClient({})
    client._api = API()
    async def run():
        return [item async for item in client.list_comments("7106686413101468970", 2, True)]
    items = asyncio.run(run())
    cfg = config_for("tiktok")
    row = tiktok._row_from_comment(items[1], fig=cfg["figures"][0],
          src=cfg["figures"][0]["sources"][0], video_id="7106686413101468970",
          video_url="https://www.tiktok.com/@creator_a/video/7106686413101468970",
          salt="s", provenance="TikTok · Figure A · post 1 of 1")
    assert row["parent_id"] == "tiktok:1"
    assert row["author_hash"] == base.hash_author("RAW2", "s")


def test_tiktok_malformed_duplicate_and_partial_rows(capsys):
    class Client(_FakeTikTokClient):
        async def list_comments(self, *args):
            yield {"id": "invalid", "text": "x", "timestamp": "bad"}
            async for row in super().list_comments(*args):
                yield row
                yield row
            raise RuntimeError("SECRET_TOKEN")
    rows = tiktok.collect(config_for("tiktok"), client=Client())
    assert len(rows) == 2
    assert "SECRET_TOKEN" not in capsys.readouterr().err


def test_failed_scrape_does_not_overwrite_existing_corpus(tmp_path):
    cfg = config_for("instagram")
    rows = instagram.collect(cfg, client=_FakeInstagramClient())
    assert write_result(rows, cfg, tmp_path) == 0
    path = tmp_path / base.RAW_PATH
    before = path.read_bytes()
    assert write_result([], cfg, tmp_path, overwrite=True) == 1
    with pytest.raises(SystemExit, match="Output already exists"):
        write_result(rows, cfg, tmp_path)
    assert path.read_bytes() == before
    manifest = (tmp_path / base.DEV_MANIFEST_PATH).read_text()
    assert '"collection_mode": "scrape"' in manifest


@pytest.mark.parametrize("value", ["https://evil.test/123456789", "https://www.tiktok.com/@wrong/video/7106686413101468970"])
def test_tiktok_rejects_wrong_host_or_account(value):
    with pytest.raises(ValueError):
        tiktok._video_id(value, "creator_a")


def test_instagram_url_shortcodes():
    assert instagram._shortcode("https://www.instagram.com/reel/ABC-_1/?igsh=foo") == "ABC-_1"
    with pytest.raises(ValueError):
        instagram._shortcode("https://evil.test/p/ABC/")
