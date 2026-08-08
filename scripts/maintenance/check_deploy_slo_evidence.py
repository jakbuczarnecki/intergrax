#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-30.2 — deploy SLO window evidence gate (W_OPS_RELEASE_CYCLES >= 2)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

from intergrax.runtime.architecture.release_cycle_tracker import (  # noqa: E402
    build_harness_baseline_release_tracker,
    resolve_release_cycle_count,
)


def main() -> int:
    slo_doc = REPO_ROOT / "docs" / "project" / "technical" / "guides" / "HARNESS_ENVIRONMENT.md"
    if not slo_doc.is_file() or "Harness SLO catalog" not in slo_doc.read_text(encoding="utf-8"):
        print("missing Harness SLO catalog in guides/HARNESS_ENVIRONMENT.md", file=sys.stderr)
        return 1

    cycles = resolve_release_cycle_count(repo_root=REPO_ROOT)
    if cycles < 2:
        print(f"release_cycles={cycles} (need >= 2 for deploy SLO evidence)", file=sys.stderr)
        return 1

    baseline = build_harness_baseline_release_tracker()
    if baseline.completed_count < 2:
        print("harness baseline release tracker must include >=2 cycles", file=sys.stderr)
        return 1

    print(f"OK: deploy SLO evidence (release_cycles={cycles})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
