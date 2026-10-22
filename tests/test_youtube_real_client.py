"""Tests for `_RealYouTubeClient` itself - the thin wrapper `test_connectors.py`
deliberately never exercises (that file's contract is "no SDK imported, ever";
`_RealYouTubeClient` only imports `googleapiclient` lazily inside its own methods).

A fake `googleapiclient.discovery` module is injected via `sys.modules`, the same
pattern `test_annotate.py` uses for `llama_cpp` - this file stays independent of
whether the real package is installed.

Regression coverage for 2026-09-08: a live 60-video pull of a real, active channel
produced duplicate `comment_id`s - one commentThread and all four of its replies came
back verbatim on two different pages. Root cause: `commentThreads().list` defaults to
`order="relevance"`, a ranking that can shift between two page requests on a video
still receiving comments, so an already-yielded thread can reappear on the next page.
The fix pins `order="time"`, a monotonic cursor. This file asserts that param is sent
on every call, so the fix cannot silently regress.
"""
from __future__ import annotations

import sys
import types

import pytest

from src.ingest.youtube import _RealYouTubeClient


class _FakeCommentThreadsResource:
    """Records every `list(...)` call's kwargs and returns scripted pages."""

    def __init__(self, pages):
        self.calls: list[dict] = []
        self._pages = list(pages)

    def list(self, **kwargs):
        self.calls.append(kwargs)
        page = self._pages.pop(0) if self._pages else {"items": []}
        return _FakeExecutable(page)


class _FakeExecutable:
    def __init__(self, payload):
        self._payload = payload

    def execute(self):
        return self._payload


class _FakeYT:
    def __init__(self, pages):
        self._comment_threads = _FakeCommentThreadsResource(pages)

    def commentThreads(self):
        return self._comment_threads


@pytest.fixture()
def fake_googleapiclient(monkeypatch):
    """Injects a fake `googleapiclient.discovery.build` and `googleapiclient.errors`
    so `_RealYouTubeClient.__init__` (which imports the real package lazily) succeeds
    with no SDK installed, matching build contract §0."""
    discovery_module = types.ModuleType("googleapiclient.discovery")
    errors_module = types.ModuleType("googleapiclient.errors")

    class _HttpError(Exception):
        pass

    errors_module.HttpError = _HttpError

    state = {"yt": None}

    def _build(service, version, developerKey=None, cache_discovery=None):
        return state["yt"]

    discovery_module.build = _build

    pkg = types.ModuleType("googleapiclient")
    pkg.discovery = discovery_module
    pkg.errors = errors_module

    monkeypatch.setitem(sys.modules, "googleapiclient", pkg)
    monkeypatch.setitem(sys.modules, "googleapiclient.discovery", discovery_module)
    monkeypatch.setitem(sys.modules, "googleapiclient.errors", errors_module)
    yield state


def test_list_comment_threads_pins_order_time_on_every_page(fake_googleapiclient):
    """The regression this file exists for: `order="time"` must be sent on the first
    page AND every subsequent page - a page-2-only fix would still duplicate page 1
    against page 2 the same way the real bug did."""
    pages = [
        {"items": [{"id": "t1"}, {"id": "t2"}], "nextPageToken": "page2"},
        {"items": [{"id": "t3"}], "nextPageToken": None},
    ]
    fake_yt = _FakeYT(pages)
    fake_googleapiclient["yt"] = fake_yt

    client = _RealYouTubeClient("fake-key")
    items = list(client.list_comment_threads("video1", max_results=10))

    assert [i["id"] for i in items] == ["t1", "t2", "t3"]
    calls = fake_yt._comment_threads.calls
    assert len(calls) == 2, "must have paginated to the second page"
    for call in calls:
        assert call["order"] == "time", (
            "commentThreads().list must pin order='time' on every page - the default "
            "order='relevance' is not a stable cursor and can duplicate threads across "
            "pages on an actively-commented video"
        )


def test_list_comment_threads_still_paginates_correctly_with_order_pinned(fake_googleapiclient):
    """Pinning order must not break the existing pageToken/maxResults contract."""
    pages = [
        {"items": [{"id": f"t{i}"} for i in range(5)], "nextPageToken": "p2"},
        {"items": [{"id": f"t{i}"} for i in range(5, 8)], "nextPageToken": None},
    ]
    fake_yt = _FakeYT(pages)
    fake_googleapiclient["yt"] = fake_yt

    client = _RealYouTubeClient("fake-key")
    items = list(client.list_comment_threads("video1", max_results=100))

    assert len(items) == 8
    assert len({i["id"] for i in items}) == 8, "no duplicates across the two pages"
    calls = fake_yt._comment_threads.calls
    assert calls[1]["pageToken"] == "p2"


def test_list_comment_threads_drops_a_boundary_duplicate_across_pages(fake_googleapiclient, capsys):
    """Regression for the real 2026-09-08 bug: even with order='time' pinned, a
    same-second publish-time tie can return one thread on both page 1 and page 2.
    The generator must yield it once, and say so on stderr rather than staying silent."""
    pages = [
        {"items": [{"id": "t1"}, {"id": "t2"}], "nextPageToken": "p2"},
        # t2 reappears at the top of page 2 - the exact shape of the real bug.
        {"items": [{"id": "t2"}, {"id": "t3"}], "nextPageToken": None},
    ]
    fake_yt = _FakeYT(pages)
    fake_googleapiclient["yt"] = fake_yt

    client = _RealYouTubeClient("fake-key")
    items = list(client.list_comment_threads("video1", max_results=100))

    assert [i["id"] for i in items] == ["t1", "t2", "t3"], "t2 must be yielded exactly once"
    assert "skipped 1 duplicate" in capsys.readouterr().err


def test_list_comment_threads_replies_go_with_their_deduplicated_thread(fake_googleapiclient):
    """A duplicated thread carries its replies with it (build contract: replies live
    inside the commentThread resource) - deduplicating the thread must not orphan or
    double-count them."""
    dup_thread = {"id": "t1", "replies": {"comments": [{"id": "t1.r1"}, {"id": "t1.r2"}]}}
    pages = [
        {"items": [dup_thread], "nextPageToken": "p2"},
        {"items": [dup_thread, {"id": "t2"}], "nextPageToken": None},
    ]
    fake_yt = _FakeYT(pages)
    fake_googleapiclient["yt"] = fake_yt

    client = _RealYouTubeClient("fake-key")
    items = list(client.list_comment_threads("video1", max_results=100))

    assert [i["id"] for i in items] == ["t1", "t2"]
    assert items[0]["replies"]["comments"] == [{"id": "t1.r1"}, {"id": "t1.r2"}]
