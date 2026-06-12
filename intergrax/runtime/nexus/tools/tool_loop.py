# © Artur Czarnecki. All rights reserved.

"""Bounded multi-iteration tool loop (TOOL-ENG-6 · TOOL-ENG-22)."""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from typing import TYPE_CHECKING

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.tool_call import LLMToolCall
from intergrax.runtime.nexus.budget.budget_ticks import enforce_tool_call_budget
from intergrax.runtime.nexus.config_types import ToolInvocationMode
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState, ToolCallTrace
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.tool_invocation_aggregate import ToolInvocationAggregate
from intergrax.runtime.nexus.tools.tool_verify_hooks import emit_high_risk_tool_verify_signal
from intergrax.runtime.nexus.tools.tool_invocation_pattern import (
    ToolInvocationPattern,
    ToolInvocationResult,
    ToolInvocationStopReason,
    resolve_invocation_pattern,
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


def _invoke_planned_call(
    *,
    state: RuntimeState,
    invoker: RuntimeToolInvoker,
    call: PlannedToolCall,
    index: int,
    idempotency_prefix: str,
    invoke_lock: threading.Lock | None = None,
) -> ToolCallTrace:
    req = ToolExecutionRequest(
        run_id=state.run_id,
        step_id=call.step_id or f"tool-{index}",
        tool_id=call.tool_id,
        input=call.input,
        idempotency_key=f"{idempotency_prefix}:{call.tool_id}:{index}",
    )
    result = invoker.invoke(state=state, request=req, agent_id=state.request.agent_id)
    trace = _trace_from_result(call, result)
    emit_high_risk_tool_verify_signal(state=state, invoker=invoker, trace=trace)
    if invoke_lock is not None:
        with invoke_lock:
            enforce_tool_call_budget(state)
    else:
        enforce_tool_call_budget(state)
    return trace


def _trace_from_result(call: PlannedToolCall, result: object) -> ToolCallTrace:
    if result.success:
        output_preview = result.output.model_dump_json()[:400]
        error_msg = None
    else:
        output_preview = None
        error_msg = result.error.error_message
    return ToolCallTrace(
        tool_name=call.tool_id,
        arguments=call.input.model_dump(),
        output_preview=output_preview,
        success=result.success,
        error_message=error_msg,
        raw_trace={},
    )


def _call_has_side_effects(invoker: RuntimeToolInvoker, call: PlannedToolCall) -> bool:
    return invoker.registry.get(call.tool_id).contract.side_effects


def execute_planned_tool_calls(
    *,
    state: RuntimeState,
    invoker: RuntimeToolInvoker,
    calls: Sequence[PlannedToolCall],
    idempotency_prefix: str,
    max_parallel_read_only: int = 1,
) -> list[ToolCallTrace]:
    if not calls:
        return []
    if max_parallel_read_only <= 1:
        return [
            _invoke_planned_call(
                state=state,
                invoker=invoker,
                call=call,
                index=index,
                idempotency_prefix=idempotency_prefix,
            )
            for index, call in enumerate(calls)
        ]

    indexed = list(enumerate(calls))
    results: dict[int, ToolCallTrace] = {}
    read_only = [(index, call) for index, call in indexed if not _call_has_side_effects(invoker, call)]
    mutating = [(index, call) for index, call in indexed if _call_has_side_effects(invoker, call)]

    if read_only:
        invoke_lock = threading.Lock()
        workers = min(max_parallel_read_only, len(read_only))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _invoke_planned_call,
                    state=state,
                    invoker=invoker,
                    call=call,
                    index=index,
                    idempotency_prefix=idempotency_prefix,
                    invoke_lock=invoke_lock,
                ): index
                for index, call in read_only
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()

    for index, call in mutating:
        results[index] = _invoke_planned_call(
            state=state,
            invoker=invoker,
            call=call,
            index=index,
            idempotency_prefix=idempotency_prefix,
        )

    return [results[index] for index in range(len(calls))]


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
    entry_point_pattern_id: str | None = None,
) -> ToolInvocationPattern:
    return resolve_invocation_pattern(
        mode=invocation_mode,
        max_iterations=max_iterations,
        pattern_override=pattern,
        entry_point_pattern_id=entry_point_pattern_id,
    )


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
        pattern=pattern or state.context.config.tool_invocation_pattern,
        entry_point_pattern_id=state.context.config.tool_invocation_pattern_id,
    )
    result = resolved.execute(
        state=state,
        invoker=invoker,
        planner=tool_planner,
        plan=None,
        allowed_tool_ids=allowed_tool_ids,
        max_iterations=max_iterations,
        planner_input=planner_input,
    )
    pattern_id = resolved.pattern_id
    if not result.pattern_id:
        return replace(result, pattern_id=pattern_id)
    return result


def inject_tool_traces_system_context(
    state: RuntimeState,
    traces: Sequence[ToolCallTrace],
    *,
    runtime_context_prompt: str,
    aggregate: ToolInvocationAggregate | None = None,
) -> None:
    if aggregate is not None:
        tools_context_for_llm = aggregate.combined_context
    elif traces:
        tools_context_for_llm = ToolInvocationAggregate.from_traces(traces).combined_context
    else:
        return
    if not tools_context_for_llm:
        return
    insert_at = len(state.messages_for_llm) - 1
    runtime_prompt = runtime_context_prompt.format(context=tools_context_for_llm)
    state.messages_for_llm.insert(
        insert_at,
        ChatMessage(role="system", content=runtime_prompt),
    )
