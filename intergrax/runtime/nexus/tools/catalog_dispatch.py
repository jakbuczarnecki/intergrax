# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Direct catalog tool dispatch for plans and §42.12 gateway (TOOL-ENG-1, TOOL-ENG-2)."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

from intergrax.contracts.tool_request import ToolRequest, ToolResponse, ToolResponseStatus
from intergrax.runtime.nexus.errors.declarative_policy_violation_error import (
    DeclarativePolicyHitlRequiredError,
)
from intergrax.runtime.nexus.tools.declarative_policy_hitl_bridge import (
    DeclarativeHitlCandidateStatus,
    DeclarativeHitlGrantCandidateMismatch,
    DeclarativeHitlScopeAssignmentState,
    maybe_assign_declarative_hitl_scope,
    raise_hitl_pause_from_tool_invocation,
    resolve_grant_scope_candidate,
    unique_candidate_from_resolution,
)
from intergrax.runtime.nexus.tools.tool_invoker_protocol import ToolInvokerProtocol
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.unified.constants import RAG_RETRIEVE_TOOL_ID, WEBSEARCH_QUERY_TOOL_ID

if TYPE_CHECKING:
    from intergrax.runtime.nexus.engine.runtime_state import RuntimeState

_PIPELINE_SHIM_TOOL_IDS = frozenset({RAG_RETRIEVE_TOOL_ID, WEBSEARCH_QUERY_TOOL_ID})


def pipeline_shim_tool_ids() -> frozenset[str]:
    return _PIPELINE_SHIM_TOOL_IDS


def catalog_tool_ids(tool_ids: Sequence[str]) -> tuple[str, ...]:
    """Return plan tool ids that are not RAG/websearch pipeline shims."""
    return tuple(
        tool_id
        for tool_id in tool_ids
        if tool_id and tool_id not in _PIPELINE_SHIM_TOOL_IDS
    )


def resolve_tool_registry(invoker: object | None) -> ToolRegistry | None:
    if invoker is None:
        return None
    if isinstance(invoker, ToolInvokerProtocol):
        return invoker.registry
    return None


def is_registered_catalog_tool(registry: ToolRegistry, tool_id: str) -> bool:
    try:
        registry.get(tool_id)
    except KeyError:
        return False
    return True


def coerce_tool_input(
    registry: ToolRegistry,
    tool_id: str,
    raw: Mapping[str, Any] | None,
) -> BaseModel:
    reg = registry.get(tool_id)
    return reg.contract.input_schema.model_validate(dict(raw or {}))


def _resolve_raw_input(
    tool_id: str,
    tool_inputs: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    payload = tool_inputs.get(tool_id)
    if isinstance(payload, Mapping):
        return payload
    return {}


def invoke_catalog_tool_ids(
    *,
    state: "RuntimeState",
    tool_ids: Sequence[str],
    tool_inputs: Mapping[str, Mapping[str, Any]] | None = None,
    trace_step: str = "CatalogDispatch",
) -> int:
    """
    Invoke catalog tools via configured ``ToolInvokerProtocol``.

    Returns the number of successfully attempted invocations (including failures).
    """
    from intergrax.runtime.nexus.budget.budget_ticks import (
        enforce_tool_call_budget,
        record_tool_call_and_enforce,
    )
    from intergrax.runtime.nexus.engine.runtime_state import ToolCallTrace

    invoker = state.context.config.tool_invoker
    registry = resolve_tool_registry(invoker)
    if invoker is None or registry is None:
        state.trace_event(
            component=TraceComponent.PIPELINE,
            step=trace_step,
            message="Catalog dispatch skipped: tool_invoker not configured.",
            level=TraceLevel.WARNING,
        )
        return 0

    inputs_map = tool_inputs or {}
    dispatched = 0
    agent_id = state.request.agent_id

    if state.tool_traces is None:
        state.tool_traces = []

    for index, tool_id in enumerate(catalog_tool_ids(tool_ids)):
        if not is_registered_catalog_tool(registry, tool_id):
            state.trace_event(
                component=TraceComponent.PIPELINE,
                step=trace_step,
                message=f"Catalog tool {tool_id!r} not registered; skipping.",
                level=TraceLevel.WARNING,
            )
            continue

        step_id = f"{trace_step}:{tool_id}:{index}"
        try:
            validated = coerce_tool_input(
                registry,
                tool_id,
                _resolve_raw_input(tool_id, inputs_map),
            )
        except ValidationError as exc:
            state.trace_event(
                component=TraceComponent.TOOLS,
                step="catalog_dispatch_validation_error",
                message=f"Invalid input for {tool_id!r}.",
                level=TraceLevel.ERROR,
            )
            state.tool_traces.append(
                ToolCallTrace(
                    tool_name=tool_id,
                    arguments=dict(_resolve_raw_input(tool_id, inputs_map)),
                    output_preview=None,
                    success=False,
                    error_message=str(exc),
                    raw_trace={},
                )
            )
            dispatched += 1
            continue

        exec_request = ToolExecutionRequest(
            run_id=state.run_id,
            step_id=step_id,
            tool_id=tool_id,
            input=validated,
            idempotency_key=f"{state.run_id}:{step_id}",
        )
        assignment_state = (
            DeclarativeHitlScopeAssignmentState()
            if state.declarative_hitl_grant is not None
            else None
        )
        unique_candidate = None
        if state.declarative_hitl_grant is not None:
            resolution = resolve_grant_scope_candidate(
                [exec_request],
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
        exec_request = maybe_assign_declarative_hitl_scope(
            exec_request,
            state=state,
            assignment_state=assignment_state,
            unique_candidate=unique_candidate,
            request_index=0,
        )

        record_tool_call_and_enforce(state)
        try:
            result = invoker.invoke(state=state, agent_id=agent_id, request=exec_request)
        except DeclarativePolicyHitlRequiredError as exc:
            raise_hitl_pause_from_tool_invocation(
                exc,
                state=state,
                request=exec_request,
                agent_id=agent_id,
            )
        state.used_tools = True
        dispatched += 1

        if result.success and result.output is not None:
            output_preview = result.output.model_dump_json()[:400]
            error_msg = None
        else:
            output_preview = None
            error_msg = result.error.error_message if result.error else "tool_failed"

        state.tool_traces.append(
            ToolCallTrace(
                tool_name=tool_id,
                arguments=validated.model_dump(),
                output_preview=output_preview,
                success=result.success,
                error_message=error_msg,
                raw_trace={},
            )
        )
        enforce_tool_call_budget(state)

    return dispatched


def invoke_catalog_tool_request(
    *,
    state: "RuntimeState",
    request: ToolRequest,
    trace_step: str = "CatalogGateway",
) -> ToolResponse:
    """§42.12 direct catalog ``ToolRequest`` → configured ``ToolInvokerProtocol``."""
    from intergrax.runtime.nexus.budget.budget_ticks import (
        enforce_tool_call_budget,
        record_tool_call_and_enforce,
    )
    from intergrax.runtime.nexus.engine.runtime_state import ToolCallTrace

    started = time.perf_counter()
    invoker = state.context.config.tool_invoker
    registry = resolve_tool_registry(invoker)
    tool_id = request.tool_name

    if invoker is None or registry is None:
        return ToolResponse(
            request_id=request.request_id,
            status=ToolResponseStatus.FAILED,
            error="tool_invoker_not_configured",
            duration_ms=int((time.perf_counter() - started) * 1000),
        )

    if not is_registered_catalog_tool(registry, tool_id):
        return ToolResponse(
            request_id=request.request_id,
            status=ToolResponseStatus.FAILED,
            error=f"unknown_catalog_tool:{tool_id}",
            duration_ms=int((time.perf_counter() - started) * 1000),
        )

    try:
        validated = coerce_tool_input(registry, tool_id, request.input)
    except ValidationError as exc:
        return ToolResponse(
            request_id=request.request_id,
            status=ToolResponseStatus.FAILED,
            error=f"validation_error:{exc}",
            duration_ms=int((time.perf_counter() - started) * 1000),
        )

    exec_request = ToolExecutionRequest(
        run_id=state.run_id,
        step_id=request.step_id or trace_step,
        tool_id=tool_id,
        input=validated,
        idempotency_key=request.idempotency_key or f"{state.run_id}:{request.request_id}",
    )
    assignment_state = (
        DeclarativeHitlScopeAssignmentState()
        if state.declarative_hitl_grant is not None
        else None
    )
    unique_candidate = None
    if state.declarative_hitl_grant is not None:
        resolution = resolve_grant_scope_candidate(
            [exec_request],
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
    exec_request = maybe_assign_declarative_hitl_scope(
        exec_request,
        state=state,
        assignment_state=assignment_state,
        unique_candidate=unique_candidate,
        request_index=0,
    )

    record_tool_call_and_enforce(state)
    try:
        result = invoker.invoke(
            state=state,
            agent_id=request.agent_id,
            request=exec_request,
        )
    except DeclarativePolicyHitlRequiredError as exc:
        raise_hitl_pause_from_tool_invocation(
            exc,
            state=state,
            request=exec_request,
            agent_id=request.agent_id,
        )
    except Exception as exc:  # noqa: BLE001 — gateway boundary
        return ToolResponse(
            request_id=request.request_id,
            status=ToolResponseStatus.FAILED,
            error=str(exc),
            duration_ms=int((time.perf_counter() - started) * 1000),
        )

    duration_ms = int((time.perf_counter() - started) * 1000)
    if not result.success:
        error_msg = result.error.error_message if result.error else "tool_failed"
        return ToolResponse(
            request_id=request.request_id,
            status=ToolResponseStatus.FAILED,
            error=error_msg,
            duration_ms=duration_ms,
            trace_ref=state.run_id or "",
        )

    output: dict[str, Any] = {}
    if result.output is not None:
        output = result.output.model_dump()

    if state.tool_traces is not None:
        state.tool_traces.append(
            ToolCallTrace(
                tool_name=tool_id,
                arguments=validated.model_dump(),
                output_preview=output and str(output)[:400] or None,
                success=True,
                error_message=None,
                raw_trace={},
            )
        )
    state.used_tools = True
    enforce_tool_call_budget(state)

    return ToolResponse(
        request_id=request.request_id,
        status=ToolResponseStatus.SUCCESS,
        output=output,
        duration_ms=duration_ms,
        trace_ref=state.run_id or "",
    )
