#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""CI gate for agent production readiness scoreboard (ACP-PROD-12 · ACP-LEG-2)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT, REPO_ROOT / "agents", REPO_ROOT / "applications"):
    _entry = str(_path)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)
_CI_DIR = REPO_ROOT / "scripts" / "ci"
if str(_CI_DIR) not in sys.path:
    sys.path.insert(0, str(_CI_DIR))
from script_paths import resolve_script  # noqa: E402

REPORT_PATH = REPO_ROOT / "build" / "agent_production_readiness.json"
INVENTORY_PATH = REPO_ROOT / "build" / "agent_fleet_inventory.json"
TYPED_STATE_ALLOWLIST = resolve_script("check_agent_typed_state.py")

from intergrax.agents.readiness.scoreboard import (  # noqa: E402
    build_roster_readiness_report,
    load_fleet_inventory,
    mutating_checkpoint_idempotency_at_100,
)
from intergrax.contracts.agent_readiness import AgentReadinessDimension  # noqa: E402


def _typed_state_allowlist_empty() -> bool:
    text = TYPED_STATE_ALLOWLIST.read_text(encoding="utf-8")
    return "ALLOWLIST_RELATIVE: frozenset[str] = frozenset()" in text or "frozenset({})" in text


def main() -> int:
    parser = argparse.ArgumentParser(description="Check agent production readiness thresholds")
    parser.add_argument("--min-overall", type=float, default=0.0, help="Minimum roster mean overall %")
    parser.add_argument("--min-runtime", type=float, default=100.0, help="Minimum per-agent Runtime %")
    parser.add_argument(
        "--require-fleet-migration-closure",
        action="store_true",
        help="ACP-LEG-2: legacy_count=0 and Runtime 100%% roster-wide",
    )
    parser.add_argument("--fail-on-blockers", action="store_true", help="Fail when any dimension has blockers")
    parser.add_argument(
        "--require-mutating-checkpoint-idempotency-100",
        action="store_true",
        help="ACP-CLOSE-PROD-8: mutating agents must score 100%% on checkpointing and idempotency",
    )
    parser.add_argument("--regenerate", action="store_true", help="Regenerate scoreboard before check")
    args = parser.parse_args()

    violations: list[str] = []

    if args.require_fleet_migration_closure or args.min_runtime >= 100.0:
        if not INVENTORY_PATH.is_file():
            violations.append("missing build/agent_fleet_inventory.json — run audit_agent_fleet_legacy.py")
        else:
            inventory = load_fleet_inventory()
            if int(inventory.get("legacy_count", 0)) != 0:
                violations.append(f"fleet legacy_count={inventory.get('legacy_count')} (expected 0)")
            if not _typed_state_allowlist_empty():
                violations.append("check_agent_typed_state ALLOWLIST must be empty for ACP-LEG-2")

    try:
        if args.regenerate or not REPORT_PATH.is_file():
            roster = build_roster_readiness_report()
            REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
            REPORT_PATH.write_text(json.dumps(roster.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
        else:
            from intergrax.contracts.agent_readiness import AgentProductionReadinessRosterReport

            roster = AgentProductionReadinessRosterReport.model_validate_json(
                REPORT_PATH.read_text(encoding="utf-8")
            )
    except Exception as exc:  # noqa: BLE001
        violations.append(f"scoreboard load/generation failed: {exc}")
        print("\n".join(violations))
        return 1

    if roster.roster_mean_overall_pct < args.min_overall:
        violations.append(
            f"roster mean overall {roster.roster_mean_overall_pct}% < {args.min_overall}%"
        )

    if args.require_fleet_migration_closure and not roster.fleet_migration_complete:
        violations.append("fleet_migration_complete=false in scoreboard")

    if args.require_mutating_checkpoint_idempotency_100 and not mutating_checkpoint_idempotency_at_100(
        roster
    ):
        violations.append(
            "mutating agents must score 100% on checkpointing and idempotency (ACP-CLOSE-PROD-8)"
        )

    for agent_report in roster.agents:
        runtime = agent_report.dimension_score(AgentReadinessDimension.RUNTIME)
        runtime_pct = runtime.pct if runtime else 0.0
        if runtime_pct < args.min_runtime:
            violations.append(f"{agent_report.agent_id}: runtime {runtime_pct}% < {args.min_runtime}%")
        if args.fail_on_blockers:
            for dim in agent_report.dimensions:
                if dim.blockers:
                    violations.append(
                        f"{agent_report.agent_id}/{dim.dimension.value}: blockers={'; '.join(dim.blockers)}"
                    )

    if violations:
        print("Agent production readiness violations:")
        print("\n".join(violations))
        return 1

    print(
        "Agent production readiness gate: OK "
        f"({roster.agent_count} agents, runtime mean {roster.runtime_dimension_mean_pct}%, "
        f"fleet_migration_complete={roster.fleet_migration_complete})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
