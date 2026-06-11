#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Block regression on migrated fleet agents (ACP-MIG-6)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = REPO_ROOT / "build" / "agent_fleet_inventory.json"

FORBIDDEN_IN_MIGRATED_AGENT: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bRuntimeEngine\s*\("), "RuntimeEngine()"),
    (re.compile(r"\brun_pipeline_step\s*\("), "run_pipeline_step()"),
    (re.compile(r"class\s+\w+\s*\(\s*HarnessReferenceAgent\b"), "HarnessReferenceAgent subclass"),
)


def _migrated_agent_ids() -> frozenset[str]:
    if not INVENTORY_PATH.is_file():
        return frozenset({"echo", "signoff_probe", "research"})
    payload = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    agents = payload.get("agents", [])
    return frozenset(
        row["agent_id"]
        for row in agents
        if row.get("migration_status") == "migrated"
    )


def main() -> int:
    migrated = _migrated_agent_ids()
    violations: list[str] = []
    for agent_id in sorted(migrated):
        agent_py = REPO_ROOT / "agents" / agent_id / f"{agent_id}_agent.py"
        if not agent_py.is_file():
            violations.append(f"missing migrated agent module: {agent_py.relative_to(REPO_ROOT)}")
            continue
        for line_no, line in enumerate(agent_py.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            for pattern, label in FORBIDDEN_IN_MIGRATED_AGENT:
                if pattern.search(line):
                    rel = agent_py.relative_to(REPO_ROOT).as_posix()
                    violations.append(f"{rel}:{line_no}: forbidden {label} in migrated agent")

    if violations:
        print("Fleet migration regression violations:")
        print("\n".join(violations))
        return 1

    print(f"Fleet migration gate: OK ({len(migrated)} migrated agents)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
