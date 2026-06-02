#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Discourage new ToolInvocationPlan.from_legacy() in production paths (Phase U-Leg.3)."""

from __future__ import annotations

import sys
from pathlib import Path

SCAN_ROOTS = (
    "intergrax/runtime/nexus",
    "intergrax/agents",
    "agents",
    "applications",
)

FORBIDDEN = "ToolInvocationPlan.from_legacy("

GRANDFATHER = (
    "intergrax/runtime/nexus/tools/tool_runtime.py",
    "intergrax/runtime/nexus/tools/tool_gateway.py",
    "tests/",
)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    violations: list[str] = []
    for root_name in SCAN_ROOTS:
        root = repo_root / root_name
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            rel = path.relative_to(repo_root).as_posix()
            if any(rel.startswith(p) or p in rel for p in GRANDFATHER):
                continue
            for line_no, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                if FORBIDDEN in line and not line.strip().startswith("#"):
                    violations.append(f"{rel}:{line_no}: {line.strip()}")
    if violations:
        print("ToolInvocationPlan.from_legacy( in production paths:")
        print("\n".join(violations))
        return 1
    print("legacy tool plan boolean audit: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
