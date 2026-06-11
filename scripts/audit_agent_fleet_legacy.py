#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Inventory Tier-2 agent fleet legacy vs ACP surfaces (ACP-MIG-1)."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENTS_ROOT = REPO_ROOT / "agents"
OUTPUT_PATH = REPO_ROOT / "build" / "agent_fleet_inventory.json"

MIGRATION_TIERS: dict[str, str] = {
    "echo": "T0",
    "signoff_probe": "T0",
    "research": "T1",
    "summary": "T1",
    "local_search": "T1",
    "legal": "T2",
    "local_indexer": "T2",
    "local_synthesizer": "T2",
    "dispute_intake": "T2",
    "dispute_analyst": "T2",
    "dispute_strategist": "T2",
    "dispute_scenario": "T2",
    "organization_worker": "T4",
    "intergrax_assistant": "T4",
    "problem_radar": "T4",
    "vendor_discovery": "T4",
}

MIGRATED_AGENTS: frozenset[str] = frozenset({"echo", "signoff_probe", "research"})

LEGACY_MARKERS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bRuntimeEngine\b"), "runtime_engine"),
    (re.compile(r"\brun_pipeline_step\b"), "uaep_pipeline"),
    (re.compile(r"\bHarnessReferenceAgent\b"), "harness_reference_only"),
)

ACP_MARKERS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bCognitiveAgent\b|\bReflexAgent\b|\bReactAgent\b"), "cognitive_pattern_base"),
    (re.compile(r"\bon_next_step\b"), "on_next_step"),
    (re.compile(r"\bcognitive_pattern\s*="), "cognitive_pattern_declared"),
)


@dataclass(frozen=True)
class AgentInventoryRow:
    agent_id: str
    tier: str
    agent_module: str
    migration_status: str
    legacy_flags: list[str]
    acp_flags: list[str]
    uses_runtime_engine_in_agent: bool
    uses_uaep_pipeline_in_agent: bool
    typed_acp: bool


def _agent_packages() -> list[Path]:
    packages: list[Path] = []
    for path in sorted(AGENTS_ROOT.iterdir()):
        if not path.is_dir():
            continue
        if path.name.startswith("_") or path.name == "lab":
            continue
        agent_py = path / f"{path.name}_agent.py"
        if agent_py.exists():
            packages.append(path)
    return packages


def _scan_file(path: Path, markers: tuple[tuple[re.Pattern[str], str], ...]) -> list[str]:
    text = path.read_text(encoding="utf-8")
    found: list[str] = []
    for pattern, label in markers:
        if pattern.search(text):
            found.append(label)
    return found


def build_inventory() -> dict[str, object]:
    rows: list[AgentInventoryRow] = []
    for package in _agent_packages():
        agent_id = package.name
        agent_py = package / f"{agent_id}_agent.py"
        legacy_flags = _scan_file(agent_py, LEGACY_MARKERS)
        acp_flags = _scan_file(agent_py, ACP_MARKERS)
        uses_runtime = "runtime_engine" in legacy_flags
        uses_pipeline = "uaep_pipeline" in legacy_flags
        typed_acp = "cognitive_pattern_base" in acp_flags and not uses_runtime
        if agent_id in MIGRATED_AGENTS:
            migration_status = "migrated"
        elif typed_acp:
            migration_status = "typed_acp"
        else:
            migration_status = "legacy"
        rows.append(
            AgentInventoryRow(
                agent_id=agent_id,
                tier=MIGRATION_TIERS.get(agent_id, "T3"),
                agent_module=f"agents.{agent_id}.{agent_id}_agent",
                migration_status=migration_status,
                legacy_flags=legacy_flags,
                acp_flags=acp_flags,
                uses_runtime_engine_in_agent=uses_runtime,
                uses_uaep_pipeline_in_agent=uses_pipeline,
                typed_acp=typed_acp,
            )
        )
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "agent_count": len(rows),
        "migrated_count": sum(1 for row in rows if row.migration_status == "migrated"),
        "legacy_count": sum(1 for row in rows if row.migration_status == "legacy"),
        "agents": [asdict(row) for row in rows],
    }


def main() -> int:
    inventory = build_inventory()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH.relative_to(REPO_ROOT)} ({inventory['agent_count']} agents)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
