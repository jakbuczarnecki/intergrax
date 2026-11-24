#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""TL-FIX-A S10: closed-world tier-boundary gate (canonical scanner wrapper)."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _head_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> int:
    from testing_support.change_safety.data.loaders import load_evol_x_data_document
    from testing_support.change_safety.discovery.closed_world import build_closed_world_inventory
    from testing_support.change_safety.discovery.closed_world_gates import run_tier_closed_world_gates

    head = _head_sha()
    inventory = build_closed_world_inventory(source_commit_sha=head, use_cache=False)
    exclusion_register = load_evol_x_data_document("exclusion_register.v1.json")
    report = run_tier_closed_world_gates(inventory, exclusion_register)
    failures = [c for c in report.checks if not c.passed]
    if not failures:
        print("closed-world tier gate: OK")
        return 0
    print("closed-world tier gate: FAIL")
    for check in failures:
        print(f"  {check.gate_id}: {check.detail}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
