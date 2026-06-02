#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Block new ToolsAgent.run() calls outside tests and tools_agent module (Phase U-Leg.1)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

SCAN_ROOTS = (
    "applications",
    "intergrax/runtime",
    "intergrax/agents",
    "intergrax/applications",
    "agents",
)

PATTERN = re.compile(r"\bToolsAgent\s*\([^)]*\)\s*\.\s*run\s*\(")

GRANDFATHER_PREFIXES = (
    "intergrax/tools/tools_agent.py",
    "tests/unit/tools/test_tools_agent.py",
    "notebooks/",
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
            text = path.read_text(encoding="utf-8")
            for line_no, line in enumerate(text.splitlines(), start=1):
                if PATTERN.search(line):
                    violations.append(f"{rel}:{line_no}: {line.strip()}")
    if violations:
        print("ToolsAgent(...).run( outside grandfather list:")
        print("\n".join(violations))
        return 1
    print("tools_agent run audit: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
