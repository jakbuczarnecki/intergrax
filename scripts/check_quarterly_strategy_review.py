#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-1.1 — quarterly strategy review process gate."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

from intergrax.applications._shared.strategy_review_wiring import resolve_strategy_review_wiring  # noqa: E402
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile  # noqa: E402


def main() -> int:
    wiring = resolve_strategy_review_wiring(
        ApplicationEnvironmentProfile.product_defaults(),
        repo_root=REPO_ROOT,
    )
    if not wiring.enabled:
        print("product host must enable quarterly strategy review", file=sys.stderr)
        return 1
    if wiring.report is None or not wiring.report.ready:
        print("quarterly strategy review documents missing", file=sys.stderr)
        return 1
    print(f"OK: quarterly strategy review ({len(wiring.report.documents)} documents)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
