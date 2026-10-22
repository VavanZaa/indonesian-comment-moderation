"""TikTok comment collection with TikTok-Api, plus local export import mode.

    python -m src.ingest.tiktok --config config/local.yaml

Live collection uses David Teather's unofficial, Playwright-backed ``TikTok-Api``.
It first discovers recent videos for each configured public username, then stores
top-level comments and replies with parent links. A ``video_ids_by_account`` config
mapping can bypass profile-video discovery, which is useful when TikTok changes or
blocks the user-video endpoint.

TikTok's private endpoints change frequently and may return empty responses when
automated traffic is detected. A fresh ``msToken`` cookie and sometimes a permitted
proxy are needed. Operators remain responsible for collection permissions and the
platform's terms. Failures are isolated per account/video and partial rows are kept.

The heavy dependencies are imported lazily so import-only runs and unit tests do not
need TikTok-Api or a Playwright browser installed.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import sys
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from . import base
from .scraper_utils import failure, limit, write_result

PLATFORM = "tiktok"
DEFAULT_CONFIG = base.DEFAULT_CONFIG
_VIDEO_ID_RE = re.compile(r"(?:/video/)?(\d{8,})")


class TikTokScraperError(RuntimeError):
    """Live TikTok-Api setup or collection failed."""


def _configured_secret(config: Mapping[str, Any], key: str, env_name: str) -> str:
    value = os.environ.get(env_name)
    if value is None:
        value = (config.get("tiktok") or {}).get(key)
    value = str(value or "").strip()
    if value.upper().startswith("PUT_YOUR"):
        return ""
    return value


def _video_id(value: Any, account: str = "") -> str:
    value = str(value or "").strip()
    if "://" in value:
        url = urlparse(value)
        match = re.fullmatch(r"/@([^/]+)/video/(\d{8,})/?", url.path)
        if url.hostname not in ("tiktok.com", "www.tiktok.com") or not match:
            raise ValueError("expected a full TikTok video URL")
        if account and match[1].casefold() != account.lstrip("@").casefold():
            raise ValueError("video URL does not belong to configured account")
        value = match[2]
    if not re.fullmatch(r"\d{8,}", value):
        raise ValueError("expected a numeric TikTok video ID")
    return value


def _configured_video_refs(config: Mapping[str, Any], account: str) -> list[str]:
    cfg = config.get("tiktok") or {}
    mapping = cfg.get("video_ids_by_account") or cfg.get("videos_by_account") or {}
    if not isinstance(mapping, Mapping):
        raise ValueError("tiktok.video_ids_by_account must be a mapping of account to a list")
    target = str(account).lstrip("@")
    raw = mapping.get(account, mapping.get(target, []))
    if isinstance(raw, (str, int)):
        raw = [raw]
    return [str(item) for item in (raw or [])]


def _comment_mapping(comment: Any, parent_id: str | None = None) -> dict[str, Any]:
    """Reduce a TikTok-Api Comment to fields consumed immediately by ``build_row``."""
    raw = dict(getattr(comment, "as_dict", None) or {})
    user = raw.get("user") or {}
    author = getattr(comment, "author", None)
    return {
        "id": getattr(comment, "id", None) or raw.get("cid") or raw.get("id"),
        "text": getattr(comment, "text", None) if hasattr(comment, "text") else raw.get("text"),
        "author_id": user.get("uid")
        or getattr(author, "user_id", None)
        or user.get("unique_id")
        or getattr(author, "username", ""),
        "create_time": raw.get("create_time") or raw.get("createTime"),
        "parent_id": parent_id,
        "reply_count": raw.get("reply_comment_total") or raw.get("reply_count") or 0,
    }


class _RealTikTokClient:
    """Async adapter around the supported TikTok-Api object model."""

    def __init__(self, config: Mapping[str, Any]):
        self._config = config
        self._api: Any = None

    async def start(self) -> None:
        try:
            from TikTokApi import TikTokApi
        except ImportError as exc:  # pragma: no cover - operator environment
            raise TikTokScraperError(
                "TikTok-Api is not installed; run `pip install -r requirements.txt` and "
                "`python -m playwright install chromium`"
            ) from exc

        cfg = self._config.get("tiktok") or {}
        token = _configured_secret(self._config, "ms_token", "TIKTOK_MS_TOKEN")
        browser = os.environ.get("TIKTOK_BROWSER") or str(cfg.get("browser") or "chromium")
        session_kwargs: dict[str, Any] = {
            "num_sessions": 1,
            "sleep_after": int(cfg.get("sleep_after", 3)),
            "browser": browser,
            "headless": bool(cfg.get("headless", True)),
            "timeout": 30000,
            "enable_session_recovery": False,
        }
        if token:
            session_kwargs["ms_tokens"] = [token]
        if cfg.get("proxies"):
            session_kwargs["proxies"] = list(cfg["proxies"])
        if cfg.get("starting_url"):
            session_kwargs["starting_url"] = str(cfg["starting_url"])

        self._api = TikTokApi(logging_level=logging.CRITICAL + 1)
        await self._api.__aenter__()
        try:
            await self._api.create_sessions(**session_kwargs)
        except BaseException:
            await self.close()
            raise

    async def close(self) -> None:
        if self._api is not None:
            api, self._api = self._api, None
            await api.__aexit__(None, None, None)

    async def list_videos(
        self,
        account: str,
        max_videos: int,
        video_refs: Sequence[str],
    ) -> AsyncIterator[dict[str, str]]:
        target = str(account).lstrip("@")
        if video_refs:
            for ref in video_refs[:max_videos]:
                video_id = _video_id(ref, account)
                yield {
                    "id": video_id,
                    "url": f"https://www.tiktok.com/@{target}/video/{video_id}",
                }
            return

        async for video in self._api.user(username=target).videos(count=max_videos):
            video_id = str(video.id)
            yield {
                "id": video_id,
                "url": getattr(video, "url", None)
                or f"https://www.tiktok.com/@{target}/video/{video_id}",
            }

    async def list_comments(
        self,
        video_id: str,
        max_results: int,
        include_replies: bool,
    ) -> AsyncIterator[dict[str, Any]]:
        video = self._api.video(id=str(video_id))
        emitted = 0
        async for comment in video.comments(count=max_results, retries=1):
            if emitted >= max_results:
                return
            top = _comment_mapping(comment)
            yield top
            emitted += 1
            if emitted >= max_results:
                return

            if not include_replies or not top["reply_count"] or emitted >= max_results:
                continue
            reply_limit = min(int(top["reply_count"]), max_results - emitted)
            try:
                async for reply in self._replies(video_id, str(top["id"]), reply_limit):
                    if emitted >= max_results:
                        return
                    yield {**reply, "parent_id": str(top["id"])}
                    emitted += 1
                    if emitted >= max_results:
                        return
            except Exception as exc:  # noqa: BLE001
                # Reply endpoints are less stable than top-level comments; retain the
                # parent and continue through the rest of the video.
                failure(PLATFORM, "replies", exc)
                raise

    async def _replies(self, video_id, comment_id, count):
        # TikTokApi 7.3.3 Comment.replies passes author.user_id as item_id.
        # Keep its session/signing transport but supply the actual video ID.
        cursor, emitted = 0, 0
        cursors = set()
        while emitted < count:
            if str(cursor) in cursors:
                raise RuntimeError("repeated reply cursor")
            cursors.add(str(cursor))
            data = await self._api.make_request(
                url="https://www.tiktok.com/api/comment/list/reply/",
                params={"item_id": video_id, "comment_id": comment_id, "cursor": cursor,
                        "count": min(20, count - emitted)}, retries=1,
            )
            if not isinstance(data, dict) or data.get("status_code", 0) != 0:
                raise RuntimeError("invalid reply response")
            replies = data.get("comments") or []
            for reply in replies:
                yield reply
                emitted += 1
                if emitted >= count:
                    return
            if not data.get("has_more") or not replies:
                return
            cursor = data.get("cursor")
            if cursor is None:
                raise RuntimeError("missing reply cursor")


def _row_from_comment(
    item: Mapping[str, Any],
    *,
    fig: Mapping[str, Any],
    src: Mapping[str, Any],
    video_id: str,
    video_url: str,
    salt: str,
    provenance: str,
) -> dict[str, Any] | None:
    native_id = item.get("id") or item.get("cid")
    text = item.get("text")
    timestamp = item.get("create_time") or item.get("published_at") or item.get("timestamp")
    if not native_id or text is None or timestamp is None:
        return None
    user = item.get("user") or {}
    author_id = (
        item.get("author_id")
        or user.get("uid")
        or user.get("unique_id")
        or item.get("username")
        or ""
    )
    parent_native = item.get("parent_id") or None
    row = base.build_row(
        platform=PLATFORM,
        figure_id=fig["id"],
        source_account=src["account"],
        post_id=video_id,
        post_url=video_url,
        parent_id=(base.make_comment_id(PLATFORM, parent_native) if parent_native else None),
        author_id=author_id,
        salt=salt,
        published_at=base.normalize_timestamp(timestamp),
        text_original=text,
        split=fig["split"],
        genre=fig["genre"],
        collection_mode="scrape",
        provenance=provenance,
    )
    row["comment_id"] = base.make_comment_id(PLATFORM, native_id)
    return row


async def _collect_api_async(
    fig: Mapping[str, Any],
    src: Mapping[str, Any],
    config: Mapping[str, Any],
    aliases: Mapping[str, str],
    salt: str,
    *,
    client: Any = None,
) -> list[dict[str, Any]]:
    account = src["account"]
    cfg = config.get("tiktok") or {}
    max_videos = limit(cfg, "max_videos_per_account", 15)
    max_comments = limit(cfg, "max_comments_total", 25000)
    if not max_videos or not max_comments:
        return []
    include_replies = bool(cfg.get("include_replies", True))

    try:
        video_refs = _configured_video_refs(config, account)
    except ValueError as exc:
        print(f"[tiktok] invalid configuration for {account!r}: {exc}. Skipping.", file=sys.stderr)
        return []

    if client is None:
        raise RuntimeError("_collect_api_async requires a started client")

    try:
        videos = []
        async with asyncio.timeout(120):
            async for video in client.list_videos(account, max_videos, video_refs):
                videos.append(video)
                if len(videos) >= max_videos:
                    break
    except Exception as exc:  # noqa: BLE001
        hint = (
            " Configure tiktok.video_ids_by_account to bypass profile discovery."
            if not video_refs
            else ""
        )
        failure(PLATFORM, "video discovery", exc)
        print(hint, file=sys.stderr)
        return []

    if not videos:
        print(f"[tiktok] account {account!r} returned no videos - skipping.", file=sys.stderr)
        return []

    video_ids = [str(video["id"]) for video in videos]
    alias = aliases.get(fig["id"], fig["id"])
    prov = base.provenance_for_posts(PLATFORM, alias, video_ids)

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    remaining = max_comments
    for video in videos:
        if remaining <= 0:
            print("[tiktok] max_comments_total reached - stopping early.", file=sys.stderr)
            break
        video_id = str(video["id"])
        video_url = video.get("url") or (
            f"https://www.tiktok.com/@{str(account).lstrip('@')}/video/{video_id}"
        )
        try:
            # Bound each next-page/next-item wait, including repeated-cursor failures.
            items = client.list_comments(video_id, remaining, include_replies).__aiter__()
            while remaining > 0:
                try:
                    item = await asyncio.wait_for(anext(items), timeout=60)
                except StopAsyncIteration:
                    break
                try:
                    row = _row_from_comment(
                    item,
                    fig=fig,
                    src=src,
                    video_id=video_id,
                    video_url=video_url,
                    salt=salt,
                    provenance=prov[video_id],
                    )
                except (ValueError, TypeError, OverflowError):
                    continue
                if row is not None and row["comment_id"] not in seen:
                    seen.add(row["comment_id"])
                    rows.append(row)
                    remaining -= 1
                    if remaining <= 0:
                        break
        except Exception as exc:  # noqa: BLE001
            failure(PLATFORM, "comments", exc)
            break
    return rows


def _collect_import(
    fig: Mapping[str, Any],
    src: Mapping[str, Any],
    aliases: Mapping[str, str],
    salt: str,
    *,
    root: str | Path = ".",
) -> list[dict[str, Any]]:
    account = src["account"]
    files = base.import_files_for_account(root, PLATFORM, account)
    if not files:
        print(
            f"[tiktok] import mode: no export file found for account {account!r} under "
            f"data/raw/imports/{PLATFORM}/ - skipping.",
            file=sys.stderr,
        )
        return []

    raw_rows: list[dict[str, Any]] = []
    for file_path in files:
        try:
            raw_rows.extend(base.read_import_file(file_path))
        except Exception as exc:  # noqa: BLE001
            print(
                f"[tiktok] could not read import file {file_path}: {exc}. Skipping that file.",
                file=sys.stderr,
            )

    by_post: dict[str, list[dict[str, Any]]] = {}
    for item in raw_rows:
        post_id = str(item.get("post_id") or item.get("video_id") or item.get("aweme_id") or "")
        by_post.setdefault(post_id, []).append(item)
    post_ids = sorted(by_post)

    alias = aliases.get(fig["id"], fig["id"])
    prov = base.provenance_for_posts(PLATFORM, alias, post_ids)

    rows: list[dict[str, Any]] = []
    for post_id in post_ids:
        default_url = f"https://www.tiktok.com/@{account}/video/{post_id}" if post_id else ""
        for item in by_post[post_id]:
            native_id = item.get("comment_id") or item.get("id") or item.get("cid")
            text = item.get("text") or item.get("text_original")
            if not native_id or text is None:
                continue
            try:
                published = base.normalize_timestamp(
                    item.get("published_at") or item.get("create_time") or item.get("timestamp")
                )
            except ValueError as exc:
                print(
                    f"[tiktok] import: skipping comment {native_id!r} with unparseable "
                    f"timestamp: {exc}",
                    file=sys.stderr,
                )
                continue
            parent_native = item.get("parent_id") or item.get("reply_id") or None
            row = base.build_row(
                platform=PLATFORM,
                figure_id=fig["id"],
                source_account=account,
                post_id=post_id,
                post_url=item.get("post_url") or default_url,
                parent_id=(base.make_comment_id(PLATFORM, parent_native) if parent_native else None),
                author_id=item.get("author_id")
                or item.get("author")
                or item.get("username")
                or item.get("unique_id")
                or "",
                salt=salt,
                published_at=published,
                text_original=text,
                split=fig["split"],
                genre=fig["genre"],
                collection_mode="import",
                provenance=prov.get(post_id, ""),
            )
            row["comment_id"] = base.make_comment_id(PLATFORM, native_id)
            rows.append(row)
    return rows


async def collect_async(
    config: Mapping[str, Any],
    figures: Sequence[Mapping[str, Any]] | None = None,
    aliases: Mapping[str, str] | None = None,
    *,
    client: Any = None,
    root: str | Path = ".",
) -> list[dict[str, Any]]:
    """Async collector for services/notebooks already running an event loop."""
    figures = list(figures) if figures is not None else base.load_figures(config)
    aliases = dict(aliases) if aliases is not None else base.build_figure_aliases(figures)
    salt = base.author_hash_salt(config)
    remaining = limit(config.get("tiktok") or {}, "max_comments_total", 25000)

    rows: list[dict[str, Any]] = []
    live_sources: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for fig in figures:
        for src in fig["sources"]:
            if src["platform"] != PLATFORM:
                continue
            if src["mode"] == "import":
                rows.extend(_collect_import(fig, src, aliases, salt, root=root))
            else:
                live_sources.append((fig, src))

    if not live_sources or not remaining or not limit(config.get("tiktok") or {}, "max_videos_per_account", 15):
        return rows

    owns_client = client is None
    api_client = client if client is not None else _RealTikTokClient(config)
    accounts = ", ".join(repr(src["account"]) for _fig, src in live_sources)
    try:
        if hasattr(api_client, "start"):
            await asyncio.wait_for(api_client.start(), timeout=120)
    except Exception as exc:  # noqa: BLE001
        failure(PLATFORM, "startup", exc)
        return rows

    try:
        for fig, src in live_sources:
            scoped = {**config, "tiktok": {**(config.get("tiktok") or {}), "max_comments_total": remaining}}
            found = await _collect_api_async(fig, src, scoped, aliases, salt, client=api_client)
            rows.extend(found)
            remaining -= len(found)
    finally:
        if owns_client and hasattr(api_client, "close"):
            try:
                await api_client.close()
            except Exception as exc:  # noqa: BLE001
                failure(PLATFORM, "session cleanup", exc)
    return rows


def collect(
    config: Mapping[str, Any],
    figures: Sequence[Mapping[str, Any]] | None = None,
    aliases: Mapping[str, str] | None = None,
    *,
    client: Any = None,
    root: str | Path = ".",
) -> list[dict[str, Any]]:
    """Synchronous adapter used by ``src.ingest.collect`` and the CLI."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            collect_async(config, figures, aliases, client=client, root=root)
        )
    raise RuntimeError(
        "tiktok.collect() cannot run inside an active event loop; await "
        "tiktok.collect_async() instead"
    )


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m src.ingest.tiktok",
        description="TikTok comment scraping with TikTok-Api, or local export import.",
    )
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--root", default=".")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    config = base.load_config(args.config)
    rows = collect(config, root=args.root)
    return write_result(rows, config, args.root, args.overwrite)


if __name__ == "__main__":
    raise SystemExit(main())
