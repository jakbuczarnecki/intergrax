#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""IDEAL-19.3 — registry snapshot baseline diff gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    baseline_path = REPO_ROOT / "tests" / "fixtures" / "registry_snapshot" / "baseline.json"
    if not baseline_path.is_file():
        print(f"missing baseline: {baseline_path}", file=sys.stderr)
        return 1
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    required_keys = {
        "tool_ids",
        "skill_ids",
        "prompt_ids",
        "agent_contract_ids",
        "evaluation_registry_ids",
        "prompt_bindings",
    }
    if set(baseline.keys()) != required_keys:
        print("baseline schema mismatch", file=sys.stderr)
        return 1
    print("OK: registry snapshot baseline schema valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
