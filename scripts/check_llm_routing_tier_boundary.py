#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""M-LLM-X.12.2 — Tier-0 llm_adapters must not import applications/."""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TIER0_ROUTING = REPO_ROOT / "intergrax" / "llm_adapters" / "routing"
FORBIDDEN = re.compile(r"\b(from|import)\s+intergrax\.applications\b")


def main() -> int:
    errors: list[str] = []
    for path in sorted(TIER0_ROUTING.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for match in FORBIDDEN.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            errors.append(
                f"{path.relative_to(REPO_ROOT)}:{line}: forbidden applications import in Tier-0 routing",
            )
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("check_llm_routing_tier_boundary: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
