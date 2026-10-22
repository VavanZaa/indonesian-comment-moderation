"""X connector - `tweets/search/recent` with `conversation_id:`, a paid API tier.

    python -m src.ingest.x --config config/local.yaml

There is no free reply search on X (build contract §1): `tweets/search/recent`
requires a paid Basic-tier-or-higher bearer token, and even then only covers the last
7 days of tweets. Most X sources in practice will use `mode: import` instead.

Reply-tree reconstruction is simplified: `tweets/search/recent?query=conversation_id:`
returns every tweet in a conversation but not the `referenced_tweets` reply-to edges
without a further expansion. Every reply found this way is recorded with `parent_id`
pointing at the conversation's root tweet, not necessarily its immediate parent - a
documented limitation, not a fabrication (every row is still a real reply *somewhere*
in that conversation).

`requests` is imported lazily, only inside `_RealXClient`, which is only constructed
when `client=None` and a real bearer token is configured - this module imports
cleanly, and `collect()` makes no network call, with no HTTP SDK installed (build
contract §0). Tests inject a fake client instead.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import base

PLATFORM = "x"
DEFAULT_CONFIG = base.DEFAULT_CONFIG


class XAPIError(RuntimeError):
    """Any X API error not otherwise classified."""


class RateLimited(XAPIError):
    """HTTP 429 - the endpoint's rate limit window is exhausted."""


class PermissionDenied(XAPIError):
    """HTTP 401/403 - the token's tier does not include `tweets/search/recent`, which
    is the expected outcome without a paid plan."""


# --------------------------------------------------------------------------- real client
class _RealXClient:
    """Thin wrapper over the X API v2. Only instantiated for a real `mode: api`
    collection with a configured bearer token - never in a test."""

    BASE_URL = "https://api.twitter.com/2"

    def __init__(self, bearer_token: str):
        self._token = bearer_token

    def _get(self, path: str, params: Mapping[str, Any]) -> dict[str, Any]:
        import requests  # lazy: build contract §0

        headers = {"Authorization": f"Bearer {self._token}"}
        resp = requests.get(f"{self.BASE_URL}{path}", params=dict(params), headers=headers, timeout=30)
        try:
            data = resp.json()
        except ValueError:
            data = {}
        if resp.status_code == 429:
            raise RateLimited(data.get("title") or "rate limited")
        if resp.status_code in (401, 403):
            raise PermissionDenied(
                data.get("title") or "forbidden - tweets/search/recent sits behind a paid tier"
            )
        if resp.status_code >= 400:
            raise XAPIError(data.get("title") or f"HTTP {resp.status_code}")
        return data

    def resolve_user_id(self, account: str) -> str:
        """`account` may already be a numeric user id, or an @handle to resolve."""
        if str(account).lstrip("@").isdigit():
            return str(account).lstrip("@")
        data = self._get(f"/users/by/username/{str(account).lstrip('@')}", {})
        uid = (data.get("data") or {}).get("id")
        if not uid:
            raise XAPIError(f"could not resolve X account {account!r} to a user id")
        return uid

    def list_conversation_ids(self, user_id: str, max_posts: int) -> list[str]:
        data = self._get(
            f"/users/{user_id}/tweets",
            {"max_results": min(100, max(5, max_posts)), "tweet.fields": "conversation_id"},
        )
        ids: list[str] = []
        for t in data.get("data", []):
            cid = t.get("conversation_id") or t.get("id")
            if cid and cid not in ids:
                ids.append(cid)
        return ids[:max_posts]

    def list_replies(self, conversation_id: str, max_results: int):
        """Yield raw tweet resources replying within `conversation_id`, paginated via
        `meta.next_token`."""
        next_token = None
        fetched = 0
        while fetched < max_results:
            params: dict[str, Any] = {
                "query": f"conversation_id:{conversation_id}",
                "max_results": min(100, max(10, max_results - fetched)),
                "tweet.fields": "created_at,author_id,conversation_id",
            }
            if next_token:
                params["next_token"] = next_token
            data = self._get("/tweets/search/recent", params)
            items = data.get("data", [])
            for item in items:
                yield item
            fetched += len(items)
            next_token = (data.get("meta") or {}).get("next_token")
            if not next_token or not items:
                break


# --------------------------------------------------------------------------- row building
def _row_from_reply(
    item: Mapping[str, Any],
    *,
    fig: Mapping[str, Any],
    src: Mapping[str, Any],
    conversation_id: str,
    post_url: str,
    salt: str,
    provenance: str,
) -> dict[str, Any] | None:
    native_id = item.get("id")
    text = item.get("text")
    created_at = item.get("created_at")
    if not native_id or text is None or not created_at:
        return None
    author_id = item.get("author_id") or ""
    is_root = str(native_id) == str(conversation_id)
    row = base.build_row(
        platform=PLATFORM,
        figure_id=fig["id"],
        source_account=src["account"],
        post_id=conversation_id,
        post_url=post_url,
        parent_id=(None if is_root else base.make_comment_id(PLATFORM, conversation_id)),
        author_id=author_id,
        salt=salt,
        published_at=base.normalize_timestamp(created_at),
        text_original=text,
        split=fig["split"],
        genre=fig["genre"],
        collection_mode="api",
        provenance=provenance,
    )
    row["comment_id"] = base.make_comment_id(PLATFORM, native_id)
    return row


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
    x_cfg = config.get("x") or {}
    token = str(x_cfg.get("bearer_token") or "").strip()

    if client is None:
        if not token or token.upper().startswith("PUT_YOUR"):
            print(
                f"[x] missing x.bearer_token - skipping API collection for account "
                f"{account!r} (figure {fig['id']}). tweets/search/recent (conversation_id:) "
                f"sits behind a paid API tier; there is no free reply search. Use mode: "
                f"import without one.",
                file=sys.stderr,
            )
            return []
        client = _RealXClient(token)

    max_posts = int(x_cfg.get("max_conversations_per_account", 15))
    max_comments = int(x_cfg.get("max_comments_total", 25000))

    try:
        user_id = client.resolve_user_id(account)
        conversation_ids = client.list_conversation_ids(user_id, max_posts)
    except PermissionDenied as exc:
        print(
            f"[x] permission denied for account {account!r}: {exc}. This is the expected "
            f"outcome without a paid Basic-tier-or-higher token. Skipping.", file=sys.stderr,
        )
        return []
    except RateLimited as exc:
        print(f"[x] rate limited resolving account {account!r}: {exc}. Skipping.", file=sys.stderr)
        return []
    except Exception as exc:  # noqa: BLE001
        print(f"[x] could not list conversations for account {account!r}: {exc}. Skipping.", file=sys.stderr)
        return []

    if not conversation_ids:
        print(f"[x] account {account!r} returned no recent conversations - skipping.", file=sys.stderr)
        return []

    alias = aliases.get(fig["id"], fig["id"])
    prov = base.provenance_for_posts(PLATFORM, alias, conversation_ids)

    rows: list[dict[str, Any]] = []
    remaining = max_comments
    for conversation_id in conversation_ids:
        if remaining <= 0:
            print("[x] max_comments_total reached - stopping early.", file=sys.stderr)
            break
        post_url = f"https://x.com/i/status/{conversation_id}"
        try:
            for item in client.list_replies(conversation_id, remaining):
                row = _row_from_reply(
                    item, fig=fig, src=src, conversation_id=conversation_id, post_url=post_url,
                    salt=salt, provenance=prov[conversation_id],
                )
                if row is not None:
                    rows.append(row)
                    remaining -= 1
        except RateLimited as exc:
            print(
                f"[x] rate limited on conversation {conversation_id!r}: {exc}. Stopping "
                f"early, writing partial results.", file=sys.stderr,
            )
            break
        except PermissionDenied as exc:
            print(f"[x] permission denied on conversation {conversation_id!r}: {exc}. Skipping.", file=sys.stderr)
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"[x] error collecting conversation {conversation_id!r}: {exc}. Skipping.", file=sys.stderr)
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
            f"[x] import mode: no export file found for account {account!r} under "
            f"data/raw/imports/{PLATFORM}/ - skipping.",
            file=sys.stderr,
        )
        return []

    raw_rows: list[dict[str, Any]] = []
    for f in files:
        try:
            raw_rows.extend(base.read_import_file(f))
        except Exception as exc:  # noqa: BLE001
            print(f"[x] could not read import file {f}: {exc}. Skipping that file.", file=sys.stderr)

    by_post: dict[str, list[dict[str, Any]]] = {}
    for r in raw_rows:
        post_id = str(r.get("post_id") or r.get("conversation_id") or "")
        by_post.setdefault(post_id, []).append(r)
    post_ids = sorted(by_post)

    alias = aliases.get(fig["id"], fig["id"])
    prov = base.provenance_for_posts(PLATFORM, alias, post_ids)

    rows: list[dict[str, Any]] = []
    for post_id in post_ids:
        default_url = f"https://x.com/i/status/{post_id}" if post_id else ""
        for r in by_post[post_id]:
            native_id = r.get("comment_id") or r.get("id")
            text = r.get("text") or r.get("text_original")
            if not native_id or text is None:
                continue
            try:
                published = base.normalize_timestamp(
                    r.get("published_at") or r.get("created_at") or r.get("timestamp")
                )
            except ValueError as exc:
                print(
                    f"[x] import: skipping comment {native_id!r} with unparseable "
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
                post_url=r.get("post_url") or default_url,
                parent_id=(base.make_comment_id(PLATFORM, parent_native) if parent_native else None),
                author_id=r.get("author_id") or r.get("author") or r.get("username") or "",
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


# --------------------------------------------------------------------------- public API
def collect(
    config: Mapping[str, Any],
    figures: Sequence[Mapping[str, Any]] | None = None,
    aliases: Mapping[str, str] | None = None,
    *,
    client: Any = None,
    root: str | Path = ".",
) -> list[dict[str, Any]]:
    """Collect every X source across `figures`. Never raises for a per-source
    failure; see the module docstring."""
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
        prog="python -m src.ingest.x",
        description="X reply collection (tweets/search/recent) or import (build contract §1).",
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
