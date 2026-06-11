# © Artur Czarnecki. All rights reserved.

"""Bounded multi-iteration tool loop (TOOL-ENG-6 · ACP-CLOSE-PAT-1)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal, Sequence

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.tool_call import LLMToolCall
from intergrax.runtime.nexus.budget.budget_ticks import enforce_tool_call_budget
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState, ToolCallTrace
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.tool_planning_service import ToolPlanningService
from intergrax.runtime.nexus.tools.tool_planner_protocol import ToolPlannerProtocol
from intergrax.tools.core.tool_plan import PlannedToolCall, ToolCallPlan
from intergrax.tools.execution_models import ToolExecutionRequest

ToolLoopStopReason = Literal[
    "empty_tool_calls",
    "max_iterations",
    "budget_exceeded",
    "planner_final_answer",
    "legacy_single_pass",
]


@dataclass(slots=True)
class BoundedToolLoopResult:
    tool_traces: list[ToolCallTrace] = field(default_factory=list)
    loop_iterations: int = 0
    stop_reason: ToolLoopStopReason = "legacy_single_pass"
    appended_messages: list[ChatMessage] = field(default_factory=list)
    used_native_tool_messages: bool = False


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


def run_bounded_tool_loop(
    *,
    state: RuntimeState,
    invoker: RuntimeToolInvoker,
    tool_planner: ToolPlannerProtocol,
    planner_input: str | list[ChatMessage],
    allowed_tool_ids: Sequence[str] | None,
    max_iterations: int,
) -> BoundedToolLoopResult:
    """
    Plan → invoke → observe loop.

    ``max_iterations == 1`` preserves legacy single-pass semantics.
    Multi-iteration loops require native ``ToolPlanningService`` with ``role=tool`` messages.
    """
    max_iters = max(1, int(max_iterations))
    if max_iters == 1 or not isinstance(tool_planner, ToolPlanningService):
        decision = tool_planner.plan_tools(
            input_data=planner_input,
            context=None,
            run_id=state.run_id,
            allowed_tool_ids=allowed_tool_ids,
        )
        tool_plan = decision.tool_plan
        if tool_plan is None or not tool_plan.calls:
            return BoundedToolLoopResult(stop_reason="empty_tool_calls")
        traces = execute_planned_tool_calls(
            state=state,
            invoker=invoker,
            calls=tool_plan.calls,
            idempotency_prefix=state.run_id,
        )
        return BoundedToolLoopResult(
            tool_traces=traces,
            loop_iterations=1,
            stop_reason="legacy_single_pass",
        )

    messages = _coerce_messages(planner_input)
    appended: list[ChatMessage] = []
    all_traces: list[ToolCallTrace] = []
    iterations = 0
    stop_reason: ToolLoopStopReason = "max_iterations"

    while iterations < max_iters:
        iterations += 1
        try:
            llm_result, tool_plan = tool_planner.plan_native_round(
                messages,
                allowed_tool_ids=allowed_tool_ids,
                run_id=state.run_id,
            )
        except Exception:
            break

        if llm_result.content and not tool_plan.calls:
            stop_reason = "planner_final_answer"
            break

        if not tool_plan.calls:
            stop_reason = "empty_tool_calls"
            break

        round_traces = execute_planned_tool_calls(
            state=state,
            invoker=invoker,
            calls=tool_plan.calls,
            idempotency_prefix=f"{state.run_id}:loop{iterations}",
        )
        all_traces.extend(round_traces)
        before = len(messages)
        append_native_tool_messages(
            messages,
            assistant_content=llm_result.content,
            tool_calls=llm_result.tool_calls,
            traces=round_traces,
        )
        appended.extend(messages[before:])

    return BoundedToolLoopResult(
        tool_traces=all_traces,
        loop_iterations=iterations,
        stop_reason=stop_reason,
        appended_messages=appended,
        used_native_tool_messages=True,
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
