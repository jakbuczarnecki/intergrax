# © Artur Czarnecki. All rights reserved.

"""CI gate — no inline Nexus planner prompts in hot path (COG-OBS.2)."""

from __future__ import annotations

import sys
from pathlib import Path

FORBIDDEN_SNIPPET = "You are a Nexus task planner. Return JSON only"
HOT_PATH_FILES = (
    Path("intergrax/runtime/nexus/planning/nexus_llm_plan_builder.py"),
    Path("intergrax/runtime/nexus/planning/nexus_planner_prompts.py"),
)


def main() -> int:
    for target in HOT_PATH_FILES:
        if not target.exists():
            print(f"check_reasoning_gates: missing {target}")
            return 1
        text = target.read_text(encoding="utf-8")
        if FORBIDDEN_SNIPPET in text:
            print(
                f"check_reasoning_gates: inline planner prompt in {target} — use Prompt Registry"
            )
            return 1
    prompt_yaml = Path("prompts/nexus_task_planner/1.yaml")
    if not prompt_yaml.exists():
        print("check_reasoning_gates: missing nexus_task_planner prompt asset")
        return 1
    if "user_template: null" in prompt_yaml.read_text(encoding="utf-8"):
        print("check_reasoning_gates: nexus_task_planner user_template must be set")
        return 1
    print("check_reasoning_gates: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
