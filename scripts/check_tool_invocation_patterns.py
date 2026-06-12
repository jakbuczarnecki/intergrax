#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""TOOL-ENG-28 — shipped ToolInvocationPattern registry and factory gate."""

from __future__ import annotations

import sys
from pathlib import Path

from intergrax.runtime.nexus.config_types import ToolInvocationMode
from intergrax.runtime.nexus.tools import patterns as shipped_patterns
from intergrax.runtime.nexus.tools.tool_invocation_pattern import pattern_for_mode
from intergrax.runtime.nexus.tools.tool_invocation_registry import shipped_pattern_ids
from intergrax.runtime.nexus.tools.tool_loop import resolve_tool_invocation_pattern


def main() -> int:
    errors: list[str] = []

    expected_classes = {
      ToolInvocationMode.SINGLE_PASS: shipped_patterns.SinglePassPattern,
      ToolInvocationMode.BOUNDED_REACT: shipped_patterns.BoundedReactPattern,
      ToolInvocationMode.PARALLEL_BATCH: shipped_patterns.ParallelBatchPattern,
      ToolInvocationMode.DETERMINISTIC_CHAIN: shipped_patterns.DeterministicChainPattern,
      ToolInvocationMode.PARALLEL_SEMANTIC_BATCH: shipped_patterns.ParallelSemanticBatchPattern,
  }

    for mode, expected_cls in expected_classes.items():
        try:
            resolved = pattern_for_mode(mode)
        except NotImplementedError as exc:
            errors.append(f"pattern_for_mode({mode.value}) not shipped: {exc}")
            continue
        if not isinstance(resolved, expected_cls):
            errors.append(
                f"pattern_for_mode({mode.value}) expected {expected_cls.__name__}, "
                f"got {type(resolved).__name__}"
            )
        if resolved.pattern_id != mode.value:
            errors.append(
                f"pattern_id mismatch for {mode.value}: {resolved.pattern_id!r}"
            )

    if shipped_pattern_ids() != frozenset(mode.value for mode in ToolInvocationMode):
        errors.append("shipped_pattern_ids() must cover every ToolInvocationMode enum value")

    tool_loop_source = Path("intergrax/runtime/nexus/tools/tool_loop.py").read_text(encoding="utf-8")
    if "resolve_invocation_pattern" not in tool_loop_source:
        errors.append("tool_loop.py must delegate to resolve_invocation_pattern")

    plan_source = Path("intergrax/runtime/nexus/tools/plan_context_invocation.py").read_text(
        encoding="utf-8"
    )
    if "run_bounded_tool_loop" not in plan_source:
        errors.append("plan_context_invocation.py must call run_bounded_tool_loop")

    resolved_default = resolve_tool_invocation_pattern(
        invocation_mode=None,
        max_iterations=1,
    )
    if resolved_default.pattern_id != "single_pass":
        errors.append("default resolve_tool_invocation_pattern must be single_pass")

    if errors:
        print("check_tool_invocation_patterns: FAIL", file=sys.stderr)
        for item in errors:
            print(f"  - {item}", file=sys.stderr)
        return 1

    print("OK: tool invocation patterns gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
