#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Block new production imports of deprecated tools_agent (Phase Q+-L.1)."""

from __future__ import annotations

import sys
from pathlib import Path

SCAN_ROOTS = (
    "applications",
    "intergrax/runtime",
    "intergrax/agents",
    "intergrax/applications",
    "intergrax/scaffold",
)

FORBIDDEN = (
    "from intergrax.tools.tools_agent import",
    "import intergrax.tools.tools_agent",
)

# Allowed until Legal migration completes (Q+-L.2).
GRANDFATHER_PREFIXES = (
    "agents/legal/",
    "intergrax/tools/tools_agent.py",
    "intergrax/runtime/nexus/config.py",
    "intergrax/runtime/nexus/config_sections.py",
    "intergrax/runtime/nexus/runtime_steps/tools_step.py",
    "intergrax/runtime/nexus/runtime_steps/core_llm_step.py",
    "intergrax/runtime/nexus/planning/step_planner.py",
    "tests/",
)


def _grandfathered(rel: str) -> bool:
    return any(rel.startswith(p) or p in rel for p in GRANDFATHER_PREFIXES)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    violations: list[str] = []
    for root_name in SCAN_ROOTS:
        root = repo_root / root_name
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            rel = path.relative_to(repo_root).as_posix()
            if _grandfathered(rel):
                continue
            for line_no, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                for forbidden in FORBIDDEN:
                    if forbidden in stripped:
                        violations.append(f"{rel}:{line_no}: {stripped}")
    if violations:
        print("tools_agent imports outside grandfather list:")
        print("\n".join(violations))
        return 1
    print("tools_agent import audit: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
