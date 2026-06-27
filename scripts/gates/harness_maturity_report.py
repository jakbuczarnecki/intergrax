#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Emit 32-layer harness maturity scorecard (IDEAL-1.1 / IDEAL-32.3)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Post IDEAL-L3 W1 scorecard (harness infra; product rows unchanged)
LAYER_SCORECARD: list[dict[str, str | int]] = [
    {"layer": 1, "name": "Strategic Harness Model", "score": "L3"},
    {"layer": 2, "name": "Tier Model and Dependency Boundaries", "score": "L3"},
    {"layer": 3, "name": "Interface and Task Intake", "score": "L3"},
    {"layer": 4, "name": "Identity, Trust and Tenancy", "score": "L3"},
    {"layer": 5, "name": "Policy and Governance", "score": "L3"},
    {"layer": 6, "name": "LLM and Model Adapter Layer", "score": "L3"},
    {"layer": 7, "name": "Reasoning, Planning and Cognition", "score": "L3"},
    {"layer": 8, "name": "Execution Runtime and Agent OS", "score": "L3"},
    {"layer": 9, "name": "Orchestration, Scheduler and Execution Graph", "score": "L3"},
    {"layer": 10, "name": "Subagents and Multi-Agent Coordination", "score": "L3"},
    {"layer": 11, "name": "Tool Layer", "score": "L3"},
    {"layer": 12, "name": "Skill Layer", "score": "L3"},
    {"layer": 13, "name": "Integration Layer", "score": "L3"},
    {"layer": 14, "name": "RAG and Retrieval Layer", "score": "L3"},
    {"layer": 15, "name": "Memory Layer", "score": "L3"},
    {"layer": 16, "name": "Context Engineering Layer", "score": "L3"},
    {"layer": 17, "name": "Prompt Engineering and Prompt Registry", "score": "L3"},
    {"layer": 18, "name": "Agent Assembly and Agent Contracts", "score": "L3"},
    {"layer": 19, "name": "Registry Architecture", "score": "L3"},
    {"layer": 20, "name": "Capability Graph Architecture", "score": "L3"},
    {"layer": 21, "name": "Observability and Telemetry", "score": "L3"},
    {"layer": 22, "name": "Error Handling and Reliability", "score": "L3"},
    {"layer": 23, "name": "Security and Data Governance", "score": "L3"},
    {"layer": 24, "name": "Cost and Resource Governance", "score": "L3"},
    {"layer": 25, "name": "Evaluation and Benchmarking", "score": "L3"},
    {"layer": 26, "name": "Testing, CI and Architecture Gates", "score": "L3"},
    {"layer": 27, "name": "Developer Experience, Scaffold and Lab", "score": "L3"},
    {"layer": 28, "name": "Product Environment and Tier-3 Applications", "score": "L3"},
    {"layer": 29, "name": "Modality, Vision, Audio and Dedicated ML", "score": "L3"},
    {"layer": 30, "name": "Operational Excellence and SLOs", "score": "L3"},
    {"layer": 31, "name": "Agent Lifecycle Governance", "score": "L3"},
    {"layer": 32, "name": "Architecture Governance and Documentation Loop", "score": "L3"},
]


def _l3_count() -> int:
    return sum(1 for row in LAYER_SCORECARD if row["score"] == "L3")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print JSON report")
    parser.add_argument(
        "--enforce-l3-critical",
        action="store_true",
        help="Fail when critical layers (policy/reliability/observability) below L3",
    )
    args = parser.parse_args()

    report = {
        "layers": LAYER_SCORECARD,
        "l3_layers": _l3_count(),
        "total_layers": len(LAYER_SCORECARD),
        "ideal_l3_phase": "IDEAL-L3-W2",
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Harness maturity: {_l3_count()}/{len(LAYER_SCORECARD)} layers at L3+")
        for row in LAYER_SCORECARD:
            print(f"  [{row['score']}] {row['layer']:>2} {row['name']}")

    if args.enforce_l3_critical:
        critical = {5, 21, 22}
        for row in LAYER_SCORECARD:
            if row["layer"] in critical and row["score"] != "L3":
                print(
                    f"CRITICAL layer {row['layer']} below L3: {row['score']}",
                    file=sys.stderr,
                )
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
