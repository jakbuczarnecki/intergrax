#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""M-LLM-X.12.2 / M-LLM-X.13.1 — forbid Tier-0/1 routing imports from applications/."""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TIER0_ROUTING = REPO_ROOT / "intergrax" / "llm_adapters" / "routing"
TIER1_RUNTIME_STATE = REPO_ROOT / "intergrax" / "runtime" / "nexus" / "engine" / "runtime_state.py"
FORBIDDEN = re.compile(r"\b(from|import)\s+intergrax\.applications\b")
FORBIDDEN_EVALUATING_ADAPTER = re.compile(
    r"\b(from|import)\s+intergrax\.applications\._shared\.routing_evaluating_adapter\b",
)


def main() -> int:
    errors: list[str] = []
    for path in sorted(TIER0_ROUTING.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for match in FORBIDDEN.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            errors.append(
                f"{path.relative_to(REPO_ROOT)}:{line}: forbidden applications import in Tier-0 routing",
            )
    if TIER1_RUNTIME_STATE.is_file():
        text = TIER1_RUNTIME_STATE.read_text(encoding="utf-8")
        for match in FORBIDDEN_EVALUATING_ADAPTER.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            errors.append(
                f"{TIER1_RUNTIME_STATE.relative_to(REPO_ROOT)}:{line}: "
                "forbidden Tier-3 routing_evaluating_adapter import in runtime_state",
            )
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("check_llm_routing_tier_boundary: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
