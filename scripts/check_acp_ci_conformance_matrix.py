#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""ACP CI conformance matrix aggregate (architecture §40.10 · ACP-PROD-10)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

MATRIX: tuple[tuple[str, str, str | None], ...] = (
    ("CI-01", "check_agents_lifecycle_metadata.py", None),
    ("CI-02", "check_agents_vendor_imports.py", None),
    ("CI-04", "check_agent_fleet_migration.py", "tests/unit/scripts/test_check_agent_acp_close_ci.py"),
    ("CI-05", "check_agent_pattern_conformance.py", None),
    ("CI-06", "check_capability_routing.py", "tests/unit/runtime/registry/test_capability_routing_acp_con6.py"),
    ("CI-07", "check_agent_typed_state.py", None),
    ("CI-08", "check_agent_threat_model.py", "tests/unit/agents/test_org_policy_merge_acp_org.py"),
    ("CI-09", "check_agent_step_security.py", None),
    ("CI-12", "check_observability_gates.py", None),
    ("CI-13", "check_agent_step_security.py", "tests/unit/agents/persistence/test_acp_prod_persistence.py"),
    ("CI-14", "check_agent_production_readiness.py", "tests/unit/agents/persistence/test_acp_prod_persistence.py"),
    ("CI-15", "report_agent_production_readiness.py", None),
    ("CI-16", "check_agent_acp_close_ci.py", "tests/unit/scripts/test_check_agent_acp_close_ci.py"),
)


def _run_script(script: str) -> int:
    path = REPO_ROOT / "scripts" / script
    for cmd in (
        ["uv", "run", "python", str(path)],
        [PYTHON, str(path)],
    ):
        completed = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
        if completed.returncode == 0:
            return 0
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="ACP CI conformance matrix")
    parser.add_argument(
        "--scripts-only",
        action="store_true",
        help="Verify scripts/tests exist without executing gates",
    )
    args = parser.parse_args()

    violations: list[str] = []
    scripts_dir = REPO_ROOT / "scripts"
    for row_id, script, test_path in MATRIX:
        if not (scripts_dir / script).is_file():
            violations.append(f"{row_id}: missing script {script}")
        if test_path and not (REPO_ROOT / test_path).is_file():
            violations.append(f"{row_id}: missing test {test_path}")
        if not args.scripts_only and not violations:
            if _run_script(script) != 0:
                violations.append(f"{row_id}: gate failed for {script}")

    if violations:
        print("ACP CI conformance matrix violations:")
        print("\n".join(violations))
        return 1

    print("ACP CI conformance matrix: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
