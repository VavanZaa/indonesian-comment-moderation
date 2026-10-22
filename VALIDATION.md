# Validation of this source snapshot

Prepared on 2026-10-22. Checks ran on Windows using existing local toolchains and installed dependencies, with imports directed at this exported copy. This is not a clean-room dependency-install test.

| Check | Result |
| --- | --- |
| offline-tests | passed |

All 397 offline tests passed. Small reference lexicons and synthetic fixture generation are included; real comment corpora, account credentials and named-person examples are excluded. Existing like_count schema declarations were synchronized in stale tests and build_row defaults missing synthetic counts to zero. Annotation thresholds and filter logic were preserved.

The repository contains selected main source, required small configuration/reference inputs and documentation. Installed dependencies, model weights, caches, private records and bulk research outputs are excluded. Reproduce the commands in README.md after installing the declared dependencies.
