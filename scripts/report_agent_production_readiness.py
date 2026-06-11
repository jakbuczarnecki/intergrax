#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Generate Agent Production Readiness scoreboard (ACP-PROD-12)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT, REPO_ROOT / "agents", REPO_ROOT / "applications"):
    _entry = str(_path)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)
DEFAULT_OUTPUT = REPO_ROOT / "build" / "agent_production_readiness.json"
DEFAULT_MARKDOWN = REPO_ROOT / "build" / "agent_production_readiness.md"

from intergrax.agents.readiness.scoreboard import (  # noqa: E402
    build_agent_readiness_report,
    build_roster_readiness_report,
    load_fleet_inventory,
    roster_to_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Agent production readiness scoreboard")
    parser.add_argument("--agent", help="Single agent_id from fleet inventory")
    parser.add_argument("--roster", action="store_true", help="Score full roster")
    parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="json",
        help="Output format for roster report",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: build/agent_production_readiness.json|.md)",
    )
    args = parser.parse_args()

    if not args.agent and not args.roster:
        args.roster = True

    try:
        inventory = load_fleet_inventory()
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        print("Run: uv run python scripts/audit_agent_fleet_legacy.py", file=sys.stderr)
        return 1

    if args.agent:
        row = next((item for item in inventory.get("agents", []) if item.get("agent_id") == args.agent), None)
        if row is None:
            print(f"Unknown agent_id: {args.agent}", file=sys.stderr)
            return 1
        module = str(row["agent_module"])
        rel = module.replace(".", "/") + ".py"
        agent_py = REPO_ROOT / rel
        report = build_agent_readiness_report(
            agent_id=args.agent,
            agent_module=module,
            agent_py=agent_py,
            inventory=inventory,
        )
        print(json.dumps(report.model_dump(mode="json"), indent=2))
        return 0

    roster = build_roster_readiness_report(inventory=inventory)
    if args.format == "markdown":
        output = args.output or DEFAULT_MARKDOWN
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(roster_to_markdown(roster), encoding="utf-8")
        print(f"Wrote {output.relative_to(REPO_ROOT)}")
        return 0

    output = args.output or DEFAULT_OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(roster.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {output.relative_to(REPO_ROOT)} "
        f"({roster.agent_count} agents, runtime mean {roster.runtime_dimension_mean_pct}%)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
