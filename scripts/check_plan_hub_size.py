# © Artur Czarnecki. All rights reserved.
"""CI gate: plan hubs must stay token-efficient; forbid cross-plan register duplication."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs" / "plan"

MAX_HUB_LINES = 1500
MAX_HUB_TOKENS = 25_000

# ORCH master register belongs in ORCHESTRATION / platform satellites only.
ORCH_REGISTER_MARKER = "### ORCH — Master register"
ORCH_REGISTER_ALLOWED = {
    PLAN / "ORCHESTRATION.md",
    PLAN / "plan" / "PLATFORM_FOUNDATION_master_registers.md",
    PLAN / "plan" / "ORCHESTRATION_master_registers.md",
    PLAN / "plan" / "UNIFIED_EXECUTION_RUNTIME_master_registers.md",
}


def main() -> int:
    errors: list[str] = []

    for path in sorted(PLAN.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        tokens = len(text) // 4
        if lines and lines[0].startswith("# "):
            # hub plan file (not AUDIT_IDEAL etc. if they're small — still check size)
            if len(lines) > MAX_HUB_LINES:
                errors.append(
                    f"{path.relative_to(ROOT)}: {len(lines)} lines exceeds hub max {MAX_HUB_LINES}"
                )
            if tokens > MAX_HUB_TOKENS:
                errors.append(
                    f"{path.relative_to(ROOT)}: ~{tokens} tokens exceeds hub max {MAX_HUB_TOKENS}"
                )

        if ORCH_REGISTER_MARKER in text and path.resolve() not in {p.resolve() for p in ORCH_REGISTER_ALLOWED}:
            errors.append(
                f"{path.relative_to(ROOT)}: contains {ORCH_REGISTER_MARKER!r} — use canonical ORCHESTRATION/plan/plan/ source"
            )

    if errors:
        print("check_plan_hub_size: FAILED", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print("check_plan_hub_size: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
