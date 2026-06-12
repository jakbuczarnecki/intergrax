# © Artur Czarnecki. All rights reserved.

"""Batch tool result merge contract (TOOL-ENG-29)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from intergrax.runtime.nexus.engine.runtime_state import ToolCallTrace


@dataclass(frozen=True, slots=True)
class ToolInvocationAggregate:
    """Canonical merge of N tool traces before LLM context injection."""

    traces: tuple[ToolCallTrace, ...]
    combined_context: str
    success_count: int
    failure_count: int

    @classmethod
    def from_traces(cls, traces: Sequence[ToolCallTrace]) -> ToolInvocationAggregate:
        ordered = tuple(traces)
        success_count = sum(1 for trace in ordered if trace.success)
        failure_count = len(ordered) - success_count
        return cls(
            traces=ordered,
            combined_context=_format_traces_context(ordered),
            success_count=success_count,
            failure_count=failure_count,
        )


def _format_traces_context(traces: Sequence[ToolCallTrace]) -> str:
    tool_lines: list[str] = []
    for trace in traces:
        tool_lines.append(f"Tool '{trace.tool_name}' was called.")
        if trace.arguments:
            try:
                args_str = json.dumps(trace.arguments, ensure_ascii=False)
            except Exception:
                args_str = str(trace.arguments)
            tool_lines.append(f"Arguments: {args_str}")
        if trace.output_preview:
            tool_lines.append("Output:")
            tool_lines.append(trace.output_preview)
        if trace.error_message:
            tool_lines.append("Error:")
            tool_lines.append(trace.error_message)
        tool_lines.append("")
    return "\n".join(tool_lines).strip()
