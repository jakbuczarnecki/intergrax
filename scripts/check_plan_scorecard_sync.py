#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-32.2 — scorecard auto-sync on AUDIT-IDEAL plan row change."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

from intergrax.runtime.architecture.plan_scorecard_sync import (  # noqa: E402
    load_scorecard_sync,
    write_scorecard_sync_artifact,
)


def main() -> int:
    sync = load_scorecard_sync(REPO_ROOT)
    if not sync.in_sync:
        print(
            "AUDIT-IDEAL register status line out of sync with parsed row counts",
            file=sys.stderr,
        )
        return 1
    if sync.harness_l3_layers != 32:
        print("harness scorecard must remain 32/32 L3", file=sys.stderr)
        return 1
    artifact = write_scorecard_sync_artifact(REPO_ROOT, sync)
    print(
        f"OK: plan scorecard sync "
        f"({sync.done_count} done, {sync.deferred_count} deferred, {sync.planned_count} planned) "
        f"-> {artifact.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
