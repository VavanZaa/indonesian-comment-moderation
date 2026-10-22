"""Multi-platform comment collection (build contract §1, AGENTS.md R9 / D35).

    python -m src.ingest.collect   --config config/local.yaml [--platform youtube,instagram,x,tiktok]
    python -m src.ingest.youtube   --config config/local.yaml
    python -m src.ingest.instagram --config config/local.yaml
    python -m src.ingest.x         --config config/local.yaml
    python -m src.ingest.tiktok    --config config/local.yaml
    python -m src.ingest.fixture   --config config/local.yaml [--seed 0]

The shared schema helpers (`hash_author`, `make_comment_id`, `build_row`,
`provenance_string`, `load_figures`, `validate_frame`, `write_corpus`,
`write_dev_manifest`, `COLUMNS`) live in `src.ingest.base`, owned by a different lane of
this project. This package's `__init__` intentionally imports nothing from it and does no
work itself, so `import src.ingest.<anything>` never fails just because a sibling module
in this package is still being written elsewhere in the repo.
"""
