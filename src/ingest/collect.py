"""Fan out to the four platform connectors and write the shared corpus.

    python -m src.ingest.collect --config config/local.yaml [--platform youtube,instagram,x,tiktok]

Concatenates every connector's rows, validates the assembled frame (schema +
the R9 split-leak guard), and writes the three shared ingest artefacts
(build contract §1):

    data/raw/comments_raw.parquet
    data/processed/dev_manifest.json       (dev rows only - R3)
    data/processed/figure_aliases.json     (pseudonymized - R9)

`--platform` restricts collection to a subset, e.g. for a partial re-run after fixing
one source's credentials. Default is all four.
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import base
from . import instagram as instagram_mod
from . import tiktok as tiktok_mod
from . import x as x_mod
from . import youtube as youtube_mod
from .scraper_utils import write_result

CONNECTORS: dict[str, Any] = {
    "youtube": youtube_mod,
    "instagram": instagram_mod,
    "x": x_mod,
    "tiktok": tiktok_mod,
}
DEFAULT_CONFIG = base.DEFAULT_CONFIG


def collect(
    config: dict[str, Any],
    platforms: Sequence[str] | None = None,
    *,
    root: str | Path = ".",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    """Fan out to the requested connectors. Returns `(rows, figures, aliases)`."""
    names = list(platforms) if platforms else list(CONNECTORS)
    unknown = [p for p in names if p not in CONNECTORS]
    if unknown:
        raise ValueError(f"unknown platform(s) {unknown}, expected any of {sorted(CONNECTORS)}")

    figures = base.load_figures(config)
    aliases = base.build_figure_aliases(figures)

    rows: list[dict[str, Any]] = []
    for name in names:
        platform_rows = CONNECTORS[name].collect(config, figures, aliases, root=root)
        print(f"[collect] {name}: {len(platform_rows)} comments")
        rows.extend(platform_rows)
    return rows, figures, aliases


def run(
    config_path: str | Path = DEFAULT_CONFIG,
    platforms: Sequence[str] | None = None,
    root: str | Path = ".",
    overwrite: bool = False,
) -> int:
    config = base.load_config(config_path)
    rows, _figures, _aliases = collect(config, platforms, root=root)
    return write_result(rows, config, root, overwrite)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m src.ingest.collect",
        description="Fan out to the YouTube/Instagram/X/TikTok connectors and write the shared "
                    "corpus (build contract §1).",
    )
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument(
        "--platform", default=None,
        help="comma-separated subset of youtube,instagram,x,tiktok (default: all four)",
    )
    ap.add_argument("--root", default=".")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    platforms = [p.strip() for p in args.platform.split(",") if p.strip()] if args.platform else None
    return run(args.config, platforms, args.root, args.overwrite)


if __name__ == "__main__":
    raise SystemExit(main())
