#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Discourage legacy planner booleans and ToolInvocationPlan boolean shims (PF-MAINT-LEG-02)."""

from __future__ import annotations

import sys
from pathlib import Path

SCAN_ROOTS = (
    "intergrax/runtime/nexus",
    "intergrax/agents",
    "agents",
    "applications",
    "prompts",
)

FORBIDDEN = (
    "ToolInvocationPlan.from_legacy(",
    "uses_legacy_booleans_only(",
    "ToolInvocationPlan(use_rag",
    "ToolInvocationPlan(use_websearch",
    '"use_rag"',
    '"use_websearch"',
)

GRANDFATHER = (
    "intergrax/runtime/nexus/tracing/plan/",
    "intergrax/runtime/nexus/context/context_builder.py",
    "intergrax/runtime/nexus/tools/tool_runtime.py",
    "tests/",
)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    violations: list[str] = []
    for root_name in SCAN_ROOTS:
        root = repo_root / root_name
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.suffix not in {".py", ".yaml", ".yml"}:
                continue
            rel = path.relative_to(repo_root).as_posix()
            if any(rel.startswith(p) or p in rel for p in GRANDFATHER):
                continue
            for line_no, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                for token in FORBIDDEN:
                    if token in line and "planner_default" not in rel:
                        if token in ('"use_rag"', '"use_websearch"') and "planner_" not in rel:
                            continue
                        violations.append(f"{rel}:{line_no}: {stripped}")
    if violations:
        print("legacy tool plan boolean audit:")
        print("\n".join(violations))
        return 1
    print("legacy tool plan boolean audit: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
