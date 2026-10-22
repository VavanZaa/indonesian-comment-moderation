"""YouTube connector - `commentThreads.list`, 1 quota unit per call, <=100 comments.

    python -m src.ingest.youtube --config config/local.yaml

Works for any public channel with an API key (build contract §1). Every source with
`mode: api` under a figure's YouTube entry is collected through the real Data API v3;
every source with `mode: import` is read from a local export instead (no key needed).

`collect()` never raises for a reachable-but-uncooperative source - a missing key,
`quotaExceeded`, `commentsDisabled`, or any other per-video failure is logged to
stderr and that source/video is skipped, so a half-finished run still writes what it
has (build contract §1). Configuration errors (`figures:` missing a required field,
an R9 split leak) are real bugs and are allowed to raise.

`googleapiclient` is imported lazily, only inside `_RealYouTubeClient`, which is only
constructed when `client=None` and real credentials are present - so this module
imports cleanly, and `collect()` makes no network call, with no SDK installed
(build contract §0). Tests inject a fake client instead.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import base

PLATFORM = "youtube"
DEFAULT_CONFIG = base.DEFAULT_CONFIG


class QuotaExceeded(RuntimeError):
    """The API key's daily quota is exhausted."""


class CommentsDisabled(RuntimeError):
    """The video has comments turned off."""


# --------------------------------------------------------------------------- real client
class _RealYouTubeClient:
    """Thin wrapper over `googleapiclient`. Only instantiated for a real `mode: api`
    collection with a configured key - never in a test."""

    def __init__(self, api_key: str):
        from googleapiclient.discovery import build  # lazy: build contract §0

        self._yt = build("youtube", "v3", developerKey=api_key, cache_discovery=False)

    @staticmethod
    def _translate(exc: Exception) -> Exception:
        text = str(exc)
        reason = ""
        details = getattr(exc, "error_details", None) or []
        if details:
            reason = details[0].get("reason", "") if isinstance(details[0], dict) else ""
        if reason == "quotaExceeded" or "quotaExceeded" in text:
            return QuotaExceeded(text)
        if reason == "commentsDisabled" or "commentsDisabled" in text:
            return CommentsDisabled(text)
        return RuntimeError(text)

    def list_uploaded_video_ids(self, channel_id: str, max_videos: int) -> list[str]:
        from googleapiclient.errors import HttpError

        try:
            resp = self._yt.channels().list(part="contentDetails", id=channel_id).execute()
        except HttpError as exc:
            raise self._translate(exc) from exc
        items = resp.get("items") or []
        if not items:
            return []
        playlist_id = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

        ids: list[str] = []
        page_token = None
        while len(ids) < max_videos:
            try:
                resp = self._yt.playlistItems().list(
                    part="contentDetails",
                    playlistId=playlist_id,
                    maxResults=min(50, max_videos - len(ids)),
                    pageToken=page_token,
                ).execute()
            except HttpError as exc:
                raise self._translate(exc) from exc
            ids.extend(it["contentDetails"]["videoId"] for it in resp.get("items", []))
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
        return ids[:max_videos]

    def list_comment_threads(self, video_id: str, max_results: int):
        """Yield raw `commentThread` resources, paginating until `max_results` threads
        have been seen or the API runs out of pages.

        Two things were needed to make this safe, both found on a live 60-video pull
        of an active channel (2026-09-08):

        1. `order="time"` is pinned explicitly. The API's default is `order="relevance"`,
           a ranking rather than a stable cursor - it can shift between two page
           requests seconds apart on a video still receiving comments.
        2. Even with `order="time"`, one commentThread and all four of its replies came
           back as an exact duplicate block on a second page - a boundary artefact of
           time-based pagination: when multiple comments share the same publish
           second, the cursor has no tiebreaker and can return the boundary comment on
           both the page before and the page after it. This is a real API limitation,
           not something an ordering choice fixes, so this method also deduplicates by
           thread id as it yields. A handful of skipped duplicates (out of thousands of
           threads) is expected and logged, not an error.
        """
        from googleapiclient.errors import HttpError

        fetched = 0
        page_token = None
        seen_ids: set[str] = set()
        n_dupes = 0
        while fetched < max_results:
            try:
                resp = self._yt.commentThreads().list(
                    part="snippet,replies",
                    videoId=video_id,
                    maxResults=min(100, max_results - fetched),
                    pageToken=page_token,
                    order="time",
                    textFormat="plainText",
                ).execute()
            except HttpError as exc:
                raise self._translate(exc) from exc
            items = resp.get("items", [])
            for item in items:
                thread_id = item.get("id")
                if thread_id is not None and thread_id in seen_ids:
                    n_dupes += 1
                    continue
                if thread_id is not None:
                    seen_ids.add(thread_id)
                yield item
            fetched += len(items)
            page_token = resp.get("nextPageToken")
            if not page_token or not items:
                break
        if n_dupes:
            print(
                f"[youtube] video {video_id!r}: skipped {n_dupes} duplicate "
                "commentThread(s) returned across a page boundary (same-second "
                "publish-time tie, a known limitation of time-ordered pagination).",
                file=sys.stderr,
            )


# --------------------------------------------------------------------------- row building
def _row_from_comment(
    native_id: str,
    snippet: Mapping[str, Any],
    parent_native_id: str | None,
    *,
    fig: Mapping[str, Any],
    src: Mapping[str, Any],
    video_id: str,
    video_url: str,
    salt: str,
    provenance: str,
) -> dict[str, Any] | None:
    text = snippet.get("textOriginal")
    if text is None:
        text = snippet.get("textDisplay")
    if not native_id or text is None:
        return None
    author_id = (snippet.get("authorChannelId") or {}).get("value") or snippet.get("authorDisplayName") or ""
    published_raw = snippet.get("publishedAt")
    if not published_raw:
        return None
    row = base.build_row(
        platform=PLATFORM,
        figure_id=fig["id"],
        source_account=src["account"],
        post_id=video_id,
        post_url=video_url,
        parent_id=(base.make_comment_id(PLATFORM, parent_native_id) if parent_native_id else None),
        author_id=author_id,
        salt=salt,
        published_at=base.normalize_timestamp(published_raw),
        text_original=text,
        split=fig["split"],
        genre=fig["genre"],
        collection_mode="api",
        provenance=provenance,
        like_count=int(snippet.get("likeCount", 0)),
    )
    row["comment_id"] = base.make_comment_id(PLATFORM, native_id)
    return row


def _rows_from_thread(
    item: Mapping[str, Any],
    *,
    fig: Mapping[str, Any],
    src: Mapping[str, Any],
    video_id: str,
    video_url: str,
    salt: str,
    provenance: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    top = (item.get("snippet") or {}).get("topLevelComment") or {}
    top_native_id = top.get("id") or item.get("id")
    top_snippet = top.get("snippet") or {}
    row = _row_from_comment(
        top_native_id, top_snippet, None,
        fig=fig, src=src, video_id=video_id, video_url=video_url, salt=salt, provenance=provenance,
    )
    if row is not None:
        out.append(row)

    for reply in (item.get("replies") or {}).get("comments", []):
        r_row = _row_from_comment(
            reply.get("id"), reply.get("snippet") or {}, top_native_id,
            fig=fig, src=src, video_id=video_id, video_url=video_url, salt=salt, provenance=provenance,
        )
        if r_row is not None:
            out.append(r_row)
    return out


# --------------------------------------------------------------------------- api mode
def _collect_api(
    fig: Mapping[str, Any],
    src: Mapping[str, Any],
    config: Mapping[str, Any],
    aliases: Mapping[str, str],
    salt: str,
    *,
    client: Any = None,
) -> list[dict[str, Any]]:
    account = src["account"]
    yt_cfg = config.get("youtube") or {}
    api_key = str(yt_cfg.get("api_key") or "").strip()

    if client is None:
        if not api_key or api_key.upper().startswith("PUT_YOUR"):
            print(
                f"[youtube] missing youtube.api_key - skipping API collection for channel "
                f"{account!r} (figure {fig['id']}). YouTube's Data API returns public "
                f"comments for any public channel once a key is set.",
                file=sys.stderr,
            )
            return []
        client = _RealYouTubeClient(api_key)

    # Per-source override beats the channel-wide default (2026-09-09). Channels differ
    # enormously in how deep you need to go: a commentary channel's 60 most recent
    # uploads can hold 100k+ comments, while a channel whose recent output is mostly
    # Shorts needs its whole back catalogue to reach the same volume. Raising the
    # global default instead would silently re-expand every other figure too, and the
    # cost of that lands on the 2.5h annotation pass, not here.
    max_videos = int(src.get("max_videos") or yt_cfg.get("max_videos_per_channel", 15))
    max_comments = int(src.get("max_comments") or yt_cfg.get("max_comments_total", 25000))

    try:
        video_ids_map = yt_cfg.get("video_ids_by_account") or {}
        if account in video_ids_map:
            video_ids = video_ids_map[account]
            if isinstance(video_ids, str):
                video_ids = [video_ids]
            # Take only up to max_videos if specified
            video_ids = video_ids[:max_videos]
        else:
            video_ids = client.list_uploaded_video_ids(account, max_videos)
    except QuotaExceeded as exc:
        print(f"[youtube] quotaExceeded listing videos for {account!r}: {exc}. Skipping.", file=sys.stderr)
        return []
    except Exception as exc:  # noqa: BLE001 - any client/network failure is a skip, not a crash
        print(f"[youtube] could not list videos for channel {account!r}: {exc}. Skipping.", file=sys.stderr)
        return []

    if not video_ids:
        print(f"[youtube] channel {account!r} returned no videos - skipping.", file=sys.stderr)
        return []

    alias = aliases.get(fig["id"], fig["id"])
    prov = base.provenance_for_posts(PLATFORM, alias, video_ids)

    rows: list[dict[str, Any]] = []
    remaining = max_comments
    for video_id in video_ids:
        if remaining <= 0:
            print("[youtube] max_comments_total reached - stopping early.", file=sys.stderr)
            break
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        try:
            for item in client.list_comment_threads(video_id, remaining):
                new_rows = _rows_from_thread(
                    item, fig=fig, src=src, video_id=video_id, video_url=video_url,
                    salt=salt, provenance=prov[video_id],
                )
                rows.extend(new_rows)
                remaining -= 1  # one commentThread ~ one quota-relevant unit
        except CommentsDisabled:
            print(f"[youtube] comments disabled on video {video_id!r} - skipping.", file=sys.stderr)
            continue
        except QuotaExceeded:
            print(
                "[youtube] quotaExceeded - stopping YouTube collection early, writing "
                "partial results.", file=sys.stderr,
            )
            break
        except Exception as exc:  # noqa: BLE001
            print(f"[youtube] error collecting video {video_id!r}: {exc}. Skipping.", file=sys.stderr)
            continue
    return rows


# --------------------------------------------------------------------------- import mode
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
            f"[youtube] import mode: no export file found for channel {account!r} under "
            f"data/raw/imports/{PLATFORM}/ - skipping.",
            file=sys.stderr,
        )
        return []

    raw_rows: list[dict[str, Any]] = []
    for f in files:
        try:
            raw_rows.extend(base.read_import_file(f))
        except Exception as exc:  # noqa: BLE001
            print(f"[youtube] could not read import file {f}: {exc}. Skipping that file.", file=sys.stderr)

    by_post: dict[str, list[dict[str, Any]]] = {}
    for r in raw_rows:
        post_id = str(r.get("post_id") or r.get("video_id") or "")
        by_post.setdefault(post_id, []).append(r)
    post_ids = sorted(by_post)

    alias = aliases.get(fig["id"], fig["id"])
    prov = base.provenance_for_posts(PLATFORM, alias, post_ids)

    rows: list[dict[str, Any]] = []
    for post_id in post_ids:
        video_url = f"https://www.youtube.com/watch?v={post_id}" if post_id else ""
        for r in by_post[post_id]:
            native_id = r.get("comment_id") or r.get("id")
            text = r.get("text") or r.get("text_original")
            if not native_id or text is None:
                continue
            try:
                published = base.normalize_timestamp(
                    r.get("published_at") or r.get("timestamp") or r.get("created_at")
                )
            except ValueError as exc:
                print(
                    f"[youtube] import: skipping comment {native_id!r} with unparseable "
                    f"timestamp: {exc}",
                    file=sys.stderr,
                )
                continue
            parent_native = r.get("parent_id") or None
            row = base.build_row(
                platform=PLATFORM,
                figure_id=fig["id"],
                source_account=account,
                post_id=post_id,
                post_url=r.get("post_url") or video_url,
                parent_id=(base.make_comment_id(PLATFORM, parent_native) if parent_native else None),
                author_id=r.get("author_id") or r.get("author") or r.get("username") or "",
                salt=salt,
                published_at=published,
                text_original=text,
                split=fig["split"],
                genre=fig["genre"],
                collection_mode="import",
                provenance=prov.get(post_id, ""),
                like_count=int(r.get("like_count") or r.get("likes") or 0),
            )
            row["comment_id"] = base.make_comment_id(PLATFORM, native_id)
            rows.append(row)
    return rows


# --------------------------------------------------------------------------- public API
def collect(
    config: Mapping[str, Any],
    figures: Sequence[Mapping[str, Any]] | None = None,
    aliases: Mapping[str, str] | None = None,
    *,
    client: Any = None,
    root: str | Path = ".",
) -> list[dict[str, Any]]:
    """Collect every YouTube source across `figures` (default: every figure in the
    config). Never raises for a per-source failure; see the module docstring."""
    figures = list(figures) if figures is not None else base.load_figures(config)
    aliases = dict(aliases) if aliases is not None else base.build_figure_aliases(figures)
    salt = base.author_hash_salt(config)

    rows: list[dict[str, Any]] = []
    for fig in figures:
        for src in fig["sources"]:
            if src["platform"] != PLATFORM:
                continue
            if src["mode"] == "import":
                rows.extend(_collect_import(fig, src, aliases, salt, root=root))
            else:
                rows.extend(_collect_api(fig, src, config, aliases, salt, client=client))
    return rows


# --------------------------------------------------------------------------- CLI
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m src.ingest.youtube",
        description="YouTube comment collection (commentThreads.list) or import (build contract §1).",
    )
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--root", default=".")
    args = ap.parse_args(argv)

    config = base.load_config(args.config)
    rows = collect(config, root=args.root)
    base.write_collection(rows, config, args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
