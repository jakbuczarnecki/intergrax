#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""ACP-CLOSE CI aggregate — fleet migration gate + scoreboard blockers (CI-1 · CI-3)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def _run(script: str, *args: str) -> int:
    path = REPO_ROOT / "scripts" / script
    for cmd in (
        ["uv", "run", "python", str(path), *args],
        [PYTHON, str(path), *args],
    ):
        completed = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
        if completed.returncode == 0:
            return 0
    return 1


def main() -> int:
    violations: list[str] = []

    if _run("audit_agent_fleet_legacy.py") != 0:
        violations.append("audit_agent_fleet_legacy.py failed")

    if _run("check_agent_fleet_migration.py") != 0:
        violations.append("check_agent_fleet_migration.py failed (ACP-CLOSE-CI-1)")

    if _run("check_agent_creation_guide_acp_canon.py") != 0:
        violations.append(
            "check_agent_creation_guide_acp_canon.py failed (ACP-CLOSE-LEG-4)"
        )

    if _run("check_agent_acp_ap02_tool_loop_boundary.py") != 0:
        violations.append(
            "check_agent_acp_ap02_tool_loop_boundary.py failed (ACP-CLOSE-CI-2)"
        )

    if (
        _run(
            "check_agent_production_readiness.py",
            "--regenerate",
            "--fail-on-blockers",
            "--require-fleet-migration-closure",
            "--require-mutating-checkpoint-idempotency-100",
        )
        != 0
    ):
        violations.append(
            "check_agent_production_readiness.py failed "
            "(ACP-CLOSE-CI-3: --fail-on-blockers + fleet closure + mutating 100%)"
        )

    if violations:
        print("ACP-CLOSE CI gate violations:")
        print("\n".join(violations))
        return 1

    print("ACP-CLOSE CI gate: OK (fleet migration + production readiness blockers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
