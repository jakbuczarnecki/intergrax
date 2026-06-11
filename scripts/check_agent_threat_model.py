#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""CI gate — agent layer threat matrix coverage (architecture §40.7 · ACP-PROD-7)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

THREAT_GATES: tuple[tuple[str, str], ...] = (
    ("SDK bypass", "check_agents_vendor_imports.py"),
    ("Gateway-only step security", "check_agent_step_security.py"),
    ("Capability routing", "check_capability_routing.py"),
    ("Tier-3 import boundary", "check_agents_no_tier3_imports.py"),
    ("Fleet migration", "check_agent_fleet_migration.py"),
    ("Production readiness", "check_agent_production_readiness.py"),
)

REQUIRED_TESTS: tuple[str, ...] = (
    "tests/unit/runtime/registry/test_capability_routing_acp_con6.py",
    "tests/unit/agents/test_org_policy_merge_acp_org.py",
    "tests/unit/agents/persistence/test_acp_prod_persistence.py",
)


def main() -> int:
    violations: list[str] = []
    scripts = REPO_ROOT / "scripts"
    for label, script_name in THREAT_GATES:
        if not (scripts / script_name).is_file():
            violations.append(f"missing threat gate script for {label}: {script_name}")
    for rel_test in REQUIRED_TESTS:
        if not (REPO_ROOT / rel_test).is_file():
            violations.append(f"missing threat verification test: {rel_test}")
    if violations:
        print("Agent threat model gate violations:")
        print("\n".join(violations))
        return 1
    print("Agent threat model gate: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
