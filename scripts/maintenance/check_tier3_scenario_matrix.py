#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""CI gate — Tier-3 scenario matrix minimum per reference host posture (APP-CON-7)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "applications", REPO_ROOT / "agents"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from intergrax.applications._shared.tier3_scenario_matrix_wiring import (  # noqa: E402
    check_tier3_scenario_matrix,
)


def main() -> int:
    violations = check_tier3_scenario_matrix(REPO_ROOT)
    if violations:
        print("tier3 scenario matrix gate: FAILED")
        for item in sorted(violations):
            print(f"  - {item}")
        return 1

    print("tier3 scenario matrix gate: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
