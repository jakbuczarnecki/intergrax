#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""CI gate — ACP-AP-02: Nexus graph orchestration must not schedule tool iterations (ACP-CLOSE-CI-2)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Plane 1 — graph execution / orchestration must not own tool ReAct loops.
GRAPH_ORCHESTRATION_PREFIXES: tuple[str, ...] = (
    "intergrax/runtime/nexus/execution/",
    "intergrax/runtime/nexus/orchestration/",
)

FORBIDDEN_TOOL_LOOP_TOKENS: tuple[str, ...] = (
    "run_bounded_tool_loop",
    "tool_loop_step",
    "plan_native_round",
    "ToolPlanningService",
)

ALLOWED_RUN_BOUNDED_TOOL_LOOP_IMPORTERS: frozenset[str] = frozenset(
    {
        "intergrax/runtime/nexus/tools/plan_context_invocation.py",
    }
)

def _rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _scan_forbidden_tokens(rel: str, text: str) -> list[str]:
    hits: list[str] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        for token in FORBIDDEN_TOOL_LOOP_TOKENS:
            if token in line:
                hits.append(f"{rel}:{line_no}: forbidden token `{token}`")
    return hits


def main() -> int:
    violations: list[str] = []

    for prefix in GRAPH_ORCHESTRATION_PREFIXES:
        root = REPO_ROOT / prefix
        if not root.is_dir():
            violations.append(f"missing graph orchestration tree: {prefix}")
            continue
        for path in sorted(root.rglob("*.py")):
            rel = _rel(path)
            violations.extend(_scan_forbidden_tokens(rel, path.read_text(encoding="utf-8")))

    agents_root = REPO_ROOT / "agents"
    for path in sorted(agents_root.rglob("*.py")):
        rel = _rel(path)
        violations.extend(_scan_forbidden_tokens(rel, path.read_text(encoding="utf-8")))

    nexus_root = REPO_ROOT / "intergrax" / "runtime" / "nexus"
    for path in sorted(nexus_root.rglob("*.py")):
        rel = _rel(path)
        if rel == "intergrax/runtime/nexus/tools/tool_loop.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "run_bounded_tool_loop" not in text:
            continue
        if rel not in ALLOWED_RUN_BOUNDED_TOOL_LOOP_IMPORTERS:
            violations.append(
                f"{rel}: run_bounded_tool_loop only allowed in "
                f"{sorted(ALLOWED_RUN_BOUNDED_TOOL_LOOP_IMPORTERS)} and tool_loop_step.py"
            )

    if violations:
        print("ACP-AP-02 tool loop boundary violations:")
        print("\n".join(sorted(set(violations))))
        return 1

    print("ACP-AP-02 tool loop boundary: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
