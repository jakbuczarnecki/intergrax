# © Artur Czarnecki. All rights reserved.

"""Bounded multi-iteration tool loop (TOOL-ENG-6 · TOOL-ENG-22)."""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace

from pydantic import BaseModel

from pydantic import BaseModel

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.tool_call import LLMToolCall
from intergrax.runtime.nexus.budget.budget_ticks import enforce_tool_call_budget
from intergrax.runtime.nexus.config_types import ToolInvocationMode
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState, ToolCallTrace
from intergrax.runtime.nexus.errors.declarative_policy_violation_error import (
    DeclarativePolicyHitlRequiredError,
)
from intergrax.runtime.nexus.tools.declarative_policy_hitl_bridge import (
    DeclarativeHitlCandidateStatus,
    DeclarativeHitlGrantCandidateMismatch,
    DeclarativeHitlScopeAssignmentState,
    UniqueDeclarativeHitlCandidate,
    maybe_assign_declarative_hitl_scope,
    raise_hitl_pause_from_tool_invocation,
    resolve_grant_scope_candidate,
    unique_candidate_from_resolution,
)
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.tool_invocation_aggregate import ToolInvocationAggregate
from intergrax.runtime.nexus.tools.tool_verify_hooks import run_post_tool_verify
from intergrax.runtime.nexus.tools.tool_invocation_pattern import (
    ToolInvocationPattern,
    ToolInvocationResult,
    ToolInvocationStopReason,
    resolve_invocation_pattern,
)
from intergrax.runtime.nexus.tools.tool_planner_protocol import ToolPlannerProtocol
from intergrax.tools.core.tool_plan import PlannedToolCall
from intergrax.tools.execution_models import ToolExecutionRequest, ToolExecutionResult, ToolModelObservation

_TRACE_OUTPUT_PREVIEW_LIMIT = 400


@dataclass(frozen=True, slots=True)
class PlannedToolCallOutcome:
    """Diagnostic trace plus model-facing observation from one planned invoke."""

    trace: ToolCallTrace
    model_observation: ToolModelObservation


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


def _build_planned_request(
    *,
    state: RuntimeState,
    call: PlannedToolCall,
    index: int,
    idempotency_prefix: str,
) -> ToolExecutionRequest[object]:
    return ToolExecutionRequest(
        run_id=state.run_id,
        step_id=call.step_id or f"tool-{index}",
        tool_id=call.tool_id,
        input=call.input,
        idempotency_key=f"{idempotency_prefix}:{call.tool_id}:{index}",
    )


def _invoke_planned_call(
    *,
    state: RuntimeState,
    invoker: RuntimeToolInvoker,
    call: PlannedToolCall,
    index: int,
    idempotency_prefix: str,
    invoke_lock: threading.Lock | None = None,
    assignment_state: DeclarativeHitlScopeAssignmentState | None = None,
    unique_candidate: UniqueDeclarativeHitlCandidate | None = None,
) -> PlannedToolCallOutcome:
    req = _build_planned_request(
        state=state,
        call=call,
        index=index,
        idempotency_prefix=idempotency_prefix,
    )
    req = maybe_assign_declarative_hitl_scope(
        req,
        state=state,
        assignment_state=assignment_state,
        unique_candidate=unique_candidate,
        request_index=index,
    )
    try:
        result = invoker.invoke(state=state, request=req, agent_id=state.request.agent_id)
    except DeclarativePolicyHitlRequiredError as exc:
        raise_hitl_pause_from_tool_invocation(
            exc,
            state=state,
            request=req,
            agent_id=state.request.agent_id,
        )
    trace = _trace_from_result(call, result)
    run_post_tool_verify(state=state, invoker=invoker, trace=trace)
    if invoke_lock is not None:
        with invoke_lock:
            enforce_tool_call_budget(state)
    else:
        enforce_tool_call_budget(state)
    return PlannedToolCallOutcome(
        trace=trace,
        model_observation=ToolModelObservation.from_execution_result(result),
    )


def _trace_from_result(
    call: PlannedToolCall,
    result: ToolExecutionResult[BaseModel],
) -> ToolCallTrace:
    if result.success:
        assert result.output is not None
        output_preview = result.output.model_dump_json()[:_TRACE_OUTPUT_PREVIEW_LIMIT]
        error_msg = None
    else:
        output_preview = None
        assert result.error is not None
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
) -> list[PlannedToolCallOutcome]:
    if not calls:
        return []
    assignment_state = (
        DeclarativeHitlScopeAssignmentState()
        if state.declarative_hitl_grant is not None
        else None
    )
    unique_candidate = None
    if state.declarative_hitl_grant is not None:
        built_requests = [
            _build_planned_request(
                state=state,
                call=call,
                index=index,
                idempotency_prefix=idempotency_prefix,
            )
            for index, call in enumerate(calls)
        ]
        resolution = resolve_grant_scope_candidate(
            built_requests,
            grant=state.declarative_hitl_grant,
            task_id=state.task_id,
        )
        if resolution.status in (
            DeclarativeHitlCandidateStatus.NO_MATCH,
            DeclarativeHitlCandidateStatus.AMBIGUOUS,
        ):
            raise DeclarativeHitlGrantCandidateMismatch(
                status=resolution.status,
                task_id=state.task_id,
            )
        unique_candidate = unique_candidate_from_resolution(resolution)
    invoke_kwargs = {
        "state": state,
        "invoker": invoker,
        "idempotency_prefix": idempotency_prefix,
        "assignment_state": assignment_state,
        "unique_candidate": unique_candidate,
    }
    if max_parallel_read_only <= 1:
        return [
            _invoke_planned_call(
                call=call,
                index=index,
                **invoke_kwargs,
            )
            for index, call in enumerate(calls)
        ]

    indexed = list(enumerate(calls))
    results: dict[int, PlannedToolCallOutcome] = {}
    read_only = [(index, call) for index, call in indexed if not _call_has_side_effects(invoker, call)]
    mutating = [(index, call) for index, call in indexed if _call_has_side_effects(invoker, call)]

    if read_only:
        invoke_lock = threading.Lock()
        workers = min(max_parallel_read_only, len(read_only))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _invoke_planned_call,
                    call=call,
                    index=index,
                    invoke_lock=invoke_lock,
                    **invoke_kwargs,
                ): index
                for index, call in read_only
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()

    for index, call in mutating:
        results[index] = _invoke_planned_call(
            call=call,
            index=index,
            **invoke_kwargs,
        )

    return [results[index] for index in range(len(calls))]


def append_native_tool_messages(
    messages: list[ChatMessage],
    *,
    assistant_content: str,
    tool_calls: Sequence[LLMToolCall],
    outcomes: Sequence[PlannedToolCallOutcome],
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
    for tool_call, outcome in zip(tool_calls, outcomes, strict=False):
        messages.append(
            ChatMessage(
                role="tool",
                content=outcome.model_observation.content,
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
