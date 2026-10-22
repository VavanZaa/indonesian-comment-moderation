"""Instagram comment collection with Instaloader, plus local export import mode.

    python -m src.ingest.instagram --config config/local.yaml

Live collection uses the third-party ``instaloader`` package. Instagram requires a
logged-in session before Instaloader can iterate post comments, so configure a login
username and either a reusable session file or password (environment variables are
preferred). Public target accounts are independent from the login account.

Instaloader relies on Instagram's private web endpoints. They can change or throttle
without notice, and operators remain responsible for collection permissions and the
platform's terms. The connector fails per source and keeps already-collected rows.

The dependency is imported lazily. Tests and import-only runs therefore do not need
Instaloader installed, and tests inject a small client implementing ``list_media`` and
``list_comments``.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping, Sequence
from itertools import islice
from pathlib import Path
from typing import Any
import re
from urllib.parse import urlparse

from . import base
from .scraper_utils import failure, limit, selected_posts, write_result

PLATFORM = "instagram"
DEFAULT_CONFIG = base.DEFAULT_CONFIG


class InstagramScraperError(RuntimeError):
    """Base error for live Instaloader setup or collection."""


class AuthenticationRequired(InstagramScraperError):
    """No usable logged-in Instaloader session was configured."""


def _configured_secret(config: Mapping[str, Any], key: str, env_name: str) -> str:
    """Return an environment override or non-placeholder config value."""
    value = os.environ.get(env_name)
    if value is None:
        value = (config.get("instagram") or {}).get(key)
    value = str(value or "").strip()
    if value.upper().startswith("PUT_YOUR"):
        return ""
    return value


class _RealInstagramClient:
    """Small synchronous adapter around Instaloader's public Python API."""

    def __init__(self, config: Mapping[str, Any], *, root: str | Path = "."):
        ig_cfg = config.get("instagram") or {}
        username = _configured_secret(config, "login_username", "INSTAGRAM_USERNAME").lstrip("@")
        password = _configured_secret(config, "password", "INSTAGRAM_PASSWORD")
        session_value = _configured_secret(config, "session_file", "INSTAGRAM_SESSION_FILE")
        if not username:
            raise AuthenticationRequired(
                "set instagram.login_username or INSTAGRAM_USERNAME; Instaloader "
                "requires login to read comments"
            )

        try:
            import instaloader
        except ImportError as exc:  # pragma: no cover - depends on operator environment
            raise InstagramScraperError(
                "Instaloader is not installed; run `pip install -r requirements.txt`."
            ) from exc

        self._instaloader = instaloader
        self._loader = instaloader.Instaloader(
            quiet=True, max_connection_attempts=1, request_timeout=30,
            fatal_status_codes=[401, 403, 429],
        )
        # Instaloader may otherwise print response bodies or credential-bearing URLs.
        self._loader.context.error = lambda *args, **kwargs: None
        self._config = ig_cfg
        self._posts: dict[str, Any] = {}

        session_path: Path | None = None
        if session_value:
            session_path = Path(session_value)
            if not session_path.is_absolute():
                session_path = Path.cwd() / session_path

        loaded = False
        try:
            if session_path is not None:
                self._loader.load_session_from_file(username, filename=str(session_path))
            else:
                # Reuse Instaloader's normal per-user session location when present.
                self._loader.load_session_from_file(username)
            loaded = True
        except FileNotFoundError:
            pass
        except Exception as exc:  # noqa: BLE001 - surface provider-specific failures clearly
            if not password:
                self.close()
                raise AuthenticationRequired("could not load Instagram session") from exc

        if not loaded:
            if not password:
                self.close()
                where = str(session_path) if session_path else "Instaloader's default session directory"
                raise AuthenticationRequired(
                    f"no session found in {where}; set INSTAGRAM_PASSWORD once to log in"
                )
            try:
                self._loader.login(username, password)
            except Exception as exc:  # noqa: BLE001
                self.close()
                raise AuthenticationRequired("Instagram login failed; create a session with the Instaloader CLI (including 2FA if needed)") from exc

            if bool(ig_cfg.get("save_session", True)):
                try:
                    if session_path is not None:
                        session_path.parent.mkdir(parents=True, exist_ok=True)
                    self._loader.save_session_to_file(
                        filename=str(session_path) if session_path is not None else None
                    )
                except Exception:
                    self.close()
                    raise

    def list_media(self, account: str, max_media: int) -> list[dict[str, Any]]:
        target = str(account).lstrip("@")
        chosen = selected_posts(self._config, account)
        if chosen is not None:
            posts = (self._instaloader.Post.from_shortcode(self._loader.context, _shortcode(p))
                     for p in chosen[:max_media])
        else:
            profile = self._instaloader.Profile.from_username(self._loader.context, target)
            posts = islice(profile.get_posts(), max_media)
        media: list[dict[str, Any]] = []
        for post in posts:
            if chosen is not None and post.owner_username.casefold() != target.casefold():
                raise ValueError("selected Instagram post does not belong to source account")
            media_id = str(post.mediaid)
            self._posts[media_id] = post
            media.append(
                {
                    "id": media_id,
                    "permalink": f"https://www.instagram.com/p/{post.shortcode}/",
                }
            )
        return media

    def list_comments(self, media_id: str, max_results: int):
        """Yield top-level comments and their replies as plain mappings."""
        post = self._posts[str(media_id)]
        emitted = 0
        for comment in post.get_comments():
            if emitted >= max_results:
                return
            owner = comment.owner
            yield {
                "id": str(comment.id),
                "text": comment.text,
                "author_id": getattr(owner, "userid", None) or getattr(owner, "username", ""),
                "timestamp": comment.created_at_utc,
                "parent_id": None,
            }
            emitted += 1
            if emitted >= max_results:
                return
            if not self._config.get("include_replies", True):
                continue
            for answer in comment.answers:
                if emitted >= max_results:
                    return
                answer_owner = answer.owner
                yield {
                    "id": str(answer.id),
                    "text": answer.text,
                    "author_id": getattr(answer_owner, "userid", None)
                    or getattr(answer_owner, "username", ""),
                    "timestamp": answer.created_at_utc,
                    "parent_id": str(comment.id),
                }
                emitted += 1
                if emitted >= max_results:
                    return

    def close(self):
        self._loader.close()


def _shortcode(value):
    value = str(value).strip()
    if "://" in value:
        url = urlparse(value)
        if url.hostname not in ("instagram.com", "www.instagram.com"):
            raise ValueError("expected an Instagram post URL")
        match = re.fullmatch(r"/(?:p|reel|tv)/([\w-]+)/?", url.path)
        if not match:
            raise ValueError("expected an Instagram post/reel URL")
        value = match[1]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("invalid Instagram shortcode")
    return value


def _row_from_comment(
    item: Mapping[str, Any],
    *,
    fig: Mapping[str, Any],
    src: Mapping[str, Any],
    media_id: str,
    media_url: str,
    salt: str,
    provenance: str,
) -> dict[str, Any] | None:
    native_id = item.get("id")
    text = item.get("text")
    timestamp = item.get("timestamp") or item.get("created_at")
    if not native_id or text is None or timestamp is None:
        return None
    author_id = (
        item.get("author_id")
        or item.get("username")
        or (item.get("from") or {}).get("id")
        or ""
    )
    parent_native = item.get("parent_id")
    row = base.build_row(
        platform=PLATFORM,
        figure_id=fig["id"],
        source_account=src["account"],
        post_id=media_id,
        post_url=media_url,
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


def _collect_api(
    fig: Mapping[str, Any],
    src: Mapping[str, Any],
    config: Mapping[str, Any],
    aliases: Mapping[str, str],
    salt: str,
    *,
    client: Any = None,
    root: str | Path = ".",
) -> list[dict[str, Any]]:
    account = src["account"]
    ig_cfg = config.get("instagram") or {}

    max_media = limit(ig_cfg, "max_posts_per_account", ig_cfg.get("max_media_per_account", 15))
    max_comments = limit(ig_cfg, "max_comments_total", 25000)
    if not max_media or not max_comments:
        return []
    owned = client is None
    if client is None:
        try:
            client = _RealInstagramClient(config, root=root)
        except AuthenticationRequired as exc:
            print(f"[instagram] {exc}. Skipping.", file=sys.stderr)
            return []
        except Exception as exc:
            failure(PLATFORM, "setup", exc)
            return []

    try:
        media = client.list_media(account, max_media)[:max_media]
    except Exception as exc:  # noqa: BLE001
        failure(PLATFORM, "post discovery", exc)
        if owned:
            client.close()
        return []

    if not media:
        if owned:
            client.close()
        print(f"[instagram] account {account!r} returned no posts - skipping.", file=sys.stderr)
        return []

    media_ids = [str(m["id"]) for m in media]
    alias = aliases.get(fig["id"], fig["id"])
    prov = base.provenance_for_posts(PLATFORM, alias, media_ids)

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    remaining = max_comments
    for media_item in media:
        if remaining <= 0:
            print("[instagram] max_comments_total reached - stopping early.", file=sys.stderr)
            break
        media_id = str(media_item["id"])
        media_url = media_item.get("permalink") or f"https://www.instagram.com/p/{media_id}/"
        try:
            for item in client.list_comments(media_id, remaining):
                try:
                    row = _row_from_comment(
                        item, fig=fig, src=src, media_id=media_id,
                        media_url=media_url, salt=salt, provenance=prov[media_id],
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
            failure(PLATFORM, "comment collection", exc)
            break
    if owned:
        client.close()
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
            f"[instagram] import mode: no export file found for account {account!r} under "
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
                f"[instagram] could not read import file {file_path}: {exc}. Skipping that file.",
                file=sys.stderr,
            )

    by_post: dict[str, list[dict[str, Any]]] = {}
    for item in raw_rows:
        post_id = str(item.get("post_id") or item.get("media_id") or "")
        by_post.setdefault(post_id, []).append(item)
    post_ids = sorted(by_post)

    alias = aliases.get(fig["id"], fig["id"])
    prov = base.provenance_for_posts(PLATFORM, alias, post_ids)

    rows: list[dict[str, Any]] = []
    for post_id in post_ids:
        default_url = f"https://www.instagram.com/p/{post_id}/" if post_id else ""
        for item in by_post[post_id]:
            native_id = item.get("comment_id") or item.get("id")
            text = item.get("text") or item.get("text_original")
            if not native_id or text is None:
                continue
            try:
                published = base.normalize_timestamp(
                    item.get("published_at") or item.get("timestamp") or item.get("created_at")
                )
            except ValueError as exc:
                print(
                    f"[instagram] import: skipping comment {native_id!r} with unparseable "
                    f"timestamp: {exc}",
                    file=sys.stderr,
                )
                continue
            parent_native = item.get("parent_id") or None
            row = base.build_row(
                platform=PLATFORM,
                figure_id=fig["id"],
                source_account=account,
                post_id=post_id,
                post_url=item.get("post_url") or default_url,
                parent_id=(base.make_comment_id(PLATFORM, parent_native) if parent_native else None),
                author_id=item.get("author_id") or item.get("author") or item.get("username") or "",
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


def collect(
    config: Mapping[str, Any],
    figures: Sequence[Mapping[str, Any]] | None = None,
    aliases: Mapping[str, str] | None = None,
    *,
    client: Any = None,
    root: str | Path = ".",
) -> list[dict[str, Any]]:
    """Collect every Instagram source, isolating failures to each source."""
    figures = list(figures) if figures is not None else base.load_figures(config)
    aliases = dict(aliases) if aliases is not None else base.build_figure_aliases(figures)
    salt = base.author_hash_salt(config)
    remaining = limit(config.get("instagram") or {}, "max_comments_total", 25000)

    rows: list[dict[str, Any]] = []
    for fig in figures:
        for src in fig["sources"]:
            if src["platform"] != PLATFORM:
                continue
            if src["mode"] == "import":
                rows.extend(_collect_import(fig, src, aliases, salt, root=root))
            else:
                scoped = {**config, "instagram": {**(config.get("instagram") or {}),
                                                   "max_comments_total": remaining}}
                found = _collect_api(fig, src, scoped, aliases, salt, client=client, root=root)
                rows.extend(found)
                remaining -= len(found)
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m src.ingest.instagram",
        description="Instagram comment scraping with Instaloader, or local export import.",
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
