# © Artur Czarnecki. All rights reserved.

"""Bounded multi-iteration tool loop (TOOL-ENG-6 · TOOL-ENG-22)."""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from dataclasses import dataclass, replace

from pydantic import BaseModel

from intergrax.context.contracts import IterativeToolOutputBlock
from intergrax.contracts.execution_identity import (
    require_active_execution_id,
    require_active_execution_identity,
)
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.tool_call import LLMToolCall
from intergrax.runtime.nexus.budget.budget_ticks import (
    enforce_tool_call_budget,
    record_tool_call_and_enforce,
)
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
from intergrax.runtime.nexus.tools.tool_planner_protocol import (
    IterativeToolPlannerProtocol,
    ToolPlannerProtocol,
)
from intergrax.runtime.nexus.tools.investigation_proof import (
    mint_runtime_observation_evidence_reference,
)
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


def _require_canonical_tool_execution_scope(state: RuntimeState) -> None:
    active_run_id, active_attempt_id = require_active_execution_identity()
    active_execution_id = require_active_execution_id()
    del active_attempt_id, active_execution_id
    if state.run_id != active_run_id:
        raise RuntimeError(
            "tool execution run_id does not match active execution"
        )


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
    _require_canonical_tool_execution_scope(state)
    record_tool_call_and_enforce(state)
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
            state.tool_traces.append(trace)
            enforce_tool_call_budget(state)
    else:
        state.tool_traces.append(trace)
        enforce_tool_call_budget(state)
    step_id = call.step_id or f"tool-{index}"
    return PlannedToolCallOutcome(
        trace=trace,
        model_observation=_model_observation_with_evidence_reference(
            result,
            tool_id=call.tool_id,
            step_id=step_id,
        ),
    )


def _model_observation_with_evidence_reference(
    result: ToolExecutionResult[BaseModel],
    *,
    tool_id: str,
    step_id: str,
) -> ToolModelObservation:
    """Attach a stable model-visible evidence reference when the tool output lacks one."""
    if not result.success or result.output is None:
        return ToolModelObservation.from_execution_result(result)
    payload = result.output.model_dump(mode="json")
    if not isinstance(payload, dict):
        return ToolModelObservation.from_execution_result(result)
    for key in ("evidence_id", "evidence_reference", "observation_reference"):
        existing = payload.get(key)
        if isinstance(existing, str) and existing.strip():
            return ToolModelObservation.from_execution_result(result)
    reference = mint_runtime_observation_evidence_reference(
        tool_id=tool_id,
        step_id=step_id,
    )
    enriched = dict(payload)
    enriched["evidence_reference"] = reference
    return ToolModelObservation(
        content=json.dumps(enriched, ensure_ascii=False, separators=(",", ":"))
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
        error_msg = result.error.error_message[:_TRACE_OUTPUT_PREVIEW_LIMIT]
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


def planned_tool_call_fingerprint(call: PlannedToolCall) -> str:
    """Deterministic fingerprint from tool_id and validated Pydantic input."""
    payload = call.input.model_dump(mode="json")
    canonical_input = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return f"{call.tool_id}\x1e{canonical_input}"


def validate_round_tool_call_count(state: RuntimeState, planned_count: int) -> None:
    """Reject a planner round atomically before any tool side effects."""
    limit = state.context.config.max_tool_calls_per_round
    if planned_count <= limit:
        return
    raise ValueError(
        f"Planned tool calls ({planned_count}) exceed max_tool_calls_per_round ({limit})"
    )


def validate_identical_tool_call_repeats(
    calls: Sequence[PlannedToolCall],
    *,
    fingerprint_counts: dict[str, int],
    max_repeats: int | None,
) -> None:
    """
    Reject a round when executing it would exceed ``max_identical_tool_call_repeats``.

    Counts are per (tool_id, validated input) fingerprint within one bounded loop.
    ``max_repeats=2`` allows two executions; the third identical call is rejected
    before invocation.
    """
    if max_repeats is None or not calls:
        return

    pending: dict[str, int] = {}
    sample_tool_id: dict[str, str] = {}
    for call in calls:
        fingerprint = planned_tool_call_fingerprint(call)
        pending[fingerprint] = pending.get(fingerprint, 0) + 1
        sample_tool_id[fingerprint] = call.tool_id

    for fingerprint, delta in pending.items():
        projected = fingerprint_counts.get(fingerprint, 0) + delta
        if projected > max_repeats:
            tool_id = sample_tool_id[fingerprint]
            raise RuntimeError(
                "Identical tool call repeat limit exceeded for "
                f"{tool_id}: projected executions {projected} > "
                f"max_identical_tool_call_repeats ({max_repeats})"
            )


def record_identical_tool_call_fingerprints(
    calls: Sequence[PlannedToolCall],
    fingerprint_counts: dict[str, int],
) -> None:
    for call in calls:
        fingerprint = planned_tool_call_fingerprint(call)
        fingerprint_counts[fingerprint] = fingerprint_counts.get(fingerprint, 0) + 1


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
    validate_round_tool_call_count(state, len(calls))
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
            futures = {}
            for index, call in read_only:
                worker_context = copy_context()
                futures[
                    pool.submit(
                        worker_context.run,
                        _invoke_planned_call,
                        call=call,
                        index=index,
                        invoke_lock=invoke_lock,
                        **invoke_kwargs,
                    )
                ] = index
            for future in as_completed(futures):
                results[futures[future]] = future.result()

    for index, call in mutating:
        results[index] = _invoke_planned_call(
            call=call,
            index=index,
            **invoke_kwargs,
        )

    return [results[index] for index in range(len(calls))]


def append_assistant_tool_call_message(
    messages: list[ChatMessage],
    *,
    assistant_content: str,
    tool_calls: Sequence[LLMToolCall],
) -> None:
    """Append the assistant turn that requested native tool calls (not tool results)."""
    if not tool_calls:
        return
    messages.append(
        ChatMessage(
            role="assistant",
            content=assistant_content,
            tool_calls=[_tool_call_openai_dict(tc) for tc in tool_calls],
        )
    )


def tool_output_blocks_from_native_round(
    tool_calls: Sequence[LLMToolCall],
    planned_calls: Sequence[PlannedToolCall],
    outcomes: Sequence[PlannedToolCallOutcome],
) -> list[IterativeToolOutputBlock]:
    """Build typed tool-output blocks for ``TOOL_OUTPUT_BLOCKS_HANDLE`` collection."""
    blocks: list[IterativeToolOutputBlock] = []
    for tool_call, planned_call, outcome in zip(tool_calls, planned_calls, outcomes, strict=False):
        blocks.append(
            IterativeToolOutputBlock(
                content=outcome.model_observation.content,
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                step_id=planned_call.step_id,
            )
        )
    return blocks


def append_native_tool_messages(
    messages: list[ChatMessage],
    *,
    assistant_content: str,
    tool_calls: Sequence[LLMToolCall],
    outcomes: Sequence[PlannedToolCallOutcome],
) -> None:
    append_assistant_tool_call_message(
        messages,
        assistant_content=assistant_content,
        tool_calls=tool_calls,
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
    Iterative CE routing is handled by :func:`run_bounded_tool_loop_async`.
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


async def run_bounded_tool_loop_async(
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
    """Async bounded tool loop — routes iterative feedback through CE when wired."""
    max_iters = max(1, int(max_iterations))
    engine = state.context.config.context_engine
    if max_iters > 1 and engine is not None:
        if not isinstance(tool_planner, IterativeToolPlannerProtocol):
            raise TypeError(
                "Bounded iterative tool invocation (max_iterations > 1) requires "
                "a planner implementing IterativeToolPlannerProtocol"
            )
        from intergrax.runtime.nexus.context.iterative_tool_context_assembly import (
            run_ce_bounded_tool_loop,
        )

        return await run_ce_bounded_tool_loop(
            state=state,
            invoker=invoker,
            tool_planner=tool_planner,
            planner_input=planner_input,
            allowed_tool_ids=allowed_tool_ids,
            max_iterations=max_iters,
        )
    # TRANSITIONAL (UE-9D): sync fallback via BoundedReactPattern → append_native_tool_messages
    # when no context_engine is wired. Owner of removal: UE-9D.
    return run_bounded_tool_loop(
        state=state,
        invoker=invoker,
        tool_planner=tool_planner,
        planner_input=planner_input,
        allowed_tool_ids=allowed_tool_ids,
        max_iterations=max_iterations,
        invocation_mode=invocation_mode,
        pattern=pattern,
    )


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
