#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""CI gate — STRICT product EnvironmentHealthScore smoke (APP-OPS-3)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "applications", REPO_ROOT / "agents"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from intergrax.applications._shared.health_score_wiring import (  # noqa: E402
    check_strict_product_health_scores,
)


def main() -> int:
    violations = check_strict_product_health_scores(REPO_ROOT)

    if violations:
        print("application health score gate: FAILED")
        for item in sorted(violations):
            print(f"  - {item}")
        return 1

    print("application health score gate: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
