"""Small shared helpers for the opt-in live scrapers."""
from collections.abc import Mapping
from pathlib import Path
import sys

from . import base


def limit(config, key, default):
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return value


def selected_posts(config, account):
    selected = config.get("posts_by_account", {})
    if not isinstance(selected, Mapping):
        raise ValueError("posts_by_account must map accounts to lists of post IDs/URLs")
    values = selected.get(account, selected.get(account.lstrip("@")))
    if values is not None and not isinstance(values, list):
        raise ValueError("posts_by_account entries must be lists")
    return values


def failure(platform, operation, exc):
    # Provider exceptions may contain signed URLs, cookies, or raw response bodies.
    print(f"[{platform}] {operation} failed ({type(exc).__name__}); keeping partial "
          "results. Check session/access, limits, or upstream availability before retrying.",
          file=sys.stderr)


def write_result(rows, config, root, overwrite=False):
    """A failed/isolated scrape must never erase the existing real corpus."""
    if not rows:
        print("No comments collected; no output files changed.", file=sys.stderr)
        return 1
    existing = [Path(root) / p for p in
                (base.RAW_PATH, base.DEV_MANIFEST_PATH, base.FIGURE_ALIASES_PATH)]
    if not overwrite and any(p.exists() for p in existing):
        raise SystemExit("Output already exists. Choose a fresh --root or explicitly use --overwrite.")
    base.write_collection(rows, config, root)
    return 0
