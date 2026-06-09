# © Artur Czarnecki. All rights reserved.

"""CI gate — no inline Nexus planner prompts in hot path (COG-OBS.2)."""

from __future__ import annotations

import sys
from pathlib import Path

FORBIDDEN_SNIPPET = "You are a Nexus task planner. Return JSON only"
TARGET = Path("intergrax/runtime/nexus/planning/nexus_llm_plan_builder.py")


def main() -> int:
    if not TARGET.exists():
        print(f"check_reasoning_gates: missing {TARGET}")
        return 1
    text = TARGET.read_text(encoding="utf-8")
    if FORBIDDEN_SNIPPET in text:
        print("check_reasoning_gates: inline planner prompt detected — use nexus_planner_prompts")
        return 1
    print("check_reasoning_gates: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
