# © Artur Czarnecki. All rights reserved.

"""Bounded multi-iteration tool loop (TOOL-ENG-6 · TOOL-ENG-22)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.tool_call import LLMToolCall
from intergrax.runtime.nexus.budget.budget_ticks import enforce_tool_call_budget
from intergrax.runtime.nexus.config_types import ToolInvocationMode
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState, ToolCallTrace
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.tool_invocation_pattern import (
    ToolInvocationPattern,
    ToolInvocationResult,
    ToolInvocationStopReason,
    pattern_for_mode,
)
from intergrax.runtime.nexus.tools.tool_planner_protocol import ToolPlannerProtocol
from intergrax.tools.core.tool_plan import PlannedToolCall
from intergrax.tools.execution_models import ToolExecutionRequest

if TYPE_CHECKING:
    pass

# Backward-compatible aliases (TOOL-ENG-6 consumers).
BoundedToolLoopResult = ToolInvocationResult
ToolLoopStopReason = ToolInvocationStopReason


def _coerce_messages(planner_input: str | list[ChatMessage]) -> list[ChatMessage]:
    if isinstance(planner_input, list):
        return list(planner_input)
    return [ChatMessage(role="user", content=planner_input)]


def _tool_call_openai_dict(tool_call: LLMToolCall) -> dict[str, object]:
    return {
        "id": tool_call.id,
        "type": "function",
        "function": {
            "name": tool_call.name,
            "arguments": tool_call.arguments_json,
        },
    }


def execute_planned_tool_calls(
    *,
    state: RuntimeState,
    invoker: RuntimeToolInvoker,
    calls: Sequence[PlannedToolCall],
    idempotency_prefix: str,
) -> list[ToolCallTrace]:
    traces: list[ToolCallTrace] = []
    for index, call in enumerate(calls):
        req = ToolExecutionRequest(
            run_id=state.run_id,
            step_id=call.step_id or f"tool-{index}",
            tool_id=call.tool_id,
            input=call.input,
            idempotency_key=f"{idempotency_prefix}:{call.tool_id}:{index}",
        )
        result = invoker.invoke(state=state, request=req, agent_id=state.request.agent_id)
        if result.success:
            output_preview = result.output.model_dump_json()[:400]
            error_msg = None
        else:
            output_preview = None
            error_msg = result.error.error_message

        traces.append(
            ToolCallTrace(
                tool_name=call.tool_id,
                arguments=call.input.model_dump(),
                output_preview=output_preview,
                success=result.success,
                error_message=error_msg,
                raw_trace={},
            )
        )
        enforce_tool_call_budget(state)
    return traces


def append_native_tool_messages(
    messages: list[ChatMessage],
    *,
    assistant_content: str,
    tool_calls: Sequence[LLMToolCall],
    traces: Sequence[ToolCallTrace],
) -> None:
    if not tool_calls:
        return
    messages.append(
        ChatMessage(
            role="assistant",
            content=assistant_content,
            tool_calls=[_tool_call_openai_dict(tc) for tc in tool_calls],
        )
    )
    for tool_call, trace in zip(tool_calls, traces, strict=False):
        body = trace.output_preview or trace.error_message or ""
        messages.append(
            ChatMessage(
                role="tool",
                content=body,
                tool_call_id=tool_call.id,
                name=tool_call.name,
            )
        )


def resolve_tool_invocation_pattern(
    *,
    invocation_mode: ToolInvocationMode | None,
    max_iterations: int,
    pattern: ToolInvocationPattern | None = None,
) -> ToolInvocationPattern:
    if pattern is not None:
        return pattern
    if invocation_mode is not None:
        return pattern_for_mode(invocation_mode)
    if max_iterations > 1:
        from intergrax.runtime.nexus.tools.patterns.bounded_react import BoundedReactPattern

        return BoundedReactPattern()
    return pattern_for_mode(ToolInvocationMode.SINGLE_PASS)


def run_bounded_tool_loop(
    *,
    state: RuntimeState,
    invoker: RuntimeToolInvoker,
    tool_planner: ToolPlannerProtocol,
    planner_input: str | list[ChatMessage],
    allowed_tool_ids: Sequence[str] | None,
    max_iterations: int,
    invocation_mode: ToolInvocationMode | None = None,
    pattern: ToolInvocationPattern | None = None,
) -> ToolInvocationResult:
    """
    Plan → invoke → observe via injected ``ToolInvocationPattern``.

    ``max_iterations > 1`` without explicit mode preserves TOOL-ENG-6 bounded ReAct.
    """
    resolved = resolve_tool_invocation_pattern(
        invocation_mode=invocation_mode,
        max_iterations=max_iterations,
        pattern=pattern,
    )
    return resolved.execute(
        state=state,
        invoker=invoker,
        planner=tool_planner,
        plan=None,
        allowed_tool_ids=allowed_tool_ids,
        max_iterations=max_iterations,
        planner_input=planner_input,
    )


def inject_tool_traces_system_context(
    state: RuntimeState,
    traces: Sequence[ToolCallTrace],
    *,
    runtime_context_prompt: str,
) -> None:
    if not traces:
        return
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

    tools_context_for_llm = "\n".join(tool_lines).strip()
    if not tools_context_for_llm:
        return
    insert_at = len(state.messages_for_llm) - 1
    runtime_prompt = runtime_context_prompt.format(context=tools_context_for_llm)
    state.messages_for_llm.insert(
        insert_at,
        ChatMessage(role="system", content=runtime_prompt),
    )
