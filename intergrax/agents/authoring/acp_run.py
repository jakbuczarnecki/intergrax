# © Artur Czarnecki. All rights reserved.

"""Agent.run(AgentRunRequest) session loop (architecture §29.4 · ACP-DX-3)."""

from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from intergrax.agents.authoring.acp_session_host import (
    ACP_HOST_CONTEXT_KEY,
    ACPSessionHostContext,
)
from intergrax.agents.authoring.llm_router import StepLLMRouter
from intergrax.agents.authoring.shared_context_bridge import load_view, persist_view, view_from_task_metadata
from intergrax.agents.authoring.step_loop import AgentRuntime
from intergrax.agents.compliance_summary import build_compliance_summary
from intergrax.agents.run_environment import EffectiveAgentRunEnvironment, merge_environment
from intergrax.contracts.acp_metadata_keys import AcpMetadataKey, AcpRunContextKey, AcpStructuredDataKey
from intergrax.contracts.acp_state import ACP_STATE_KEY
from intergrax.contracts.agent_run import AgentRunError, AgentRunRequest, AgentRunResult
from intergrax.contracts.agent_run_trace import AgentRunTrace
from intergrax.contracts.agent_run_enums import (
    AgentRunErrorCode,
    AgentRunStatus,
    StepNextAction,
    TerminalReason,
)
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.validation import ValidationResult
from intergrax.runtime.kernel.step_kernel import HarnessKernel, StepKernelContext
from intergrax.runtime.policy.policy_engine import PolicyEngine


def _host_context_from_metadata(metadata: dict[str, Any]) -> ACPSessionHostContext | None:
    raw = metadata.get(ACP_HOST_CONTEXT_KEY)
    if isinstance(raw, ACPSessionHostContext):
        return raw
    if isinstance(raw, dict):
        return ACPSessionHostContext.model_validate(raw)
    return None


def _initial_state_root(request: AgentRunRequest) -> dict[str, Any]:
    if request.state:
        if ACP_STATE_KEY in request.state:
            return dict(request.state)
        return {ACP_STATE_KEY: dict(request.state)}
    return {ACP_STATE_KEY: {"schema_version": "acp.state.v1", "_version": 0}}


def _terminal_status(outcome_terminal: bool, next_action: StepNextAction) -> AgentRunStatus:
    if next_action == StepNextAction.PAUSE_HITL:
        return AgentRunStatus.PAUSED
    if outcome_terminal:
        return AgentRunStatus.SUCCEEDED
    return AgentRunStatus.SUCCEEDED


async def run_acp_session(
    agent: object,
    request: AgentRunRequest,
) -> AgentRunResult:
    """Execute typed agent session loop until terminal outcome."""
    started = time.perf_counter()
    contract = agent.get_contract()
    host = _host_context_from_metadata(request.metadata)
    base_merged = merge_environment(
        contract=contract,
        request=request,
        app_profile=host.app_profile if host else None,
        binding=host.binding if host else None,
    )
    overlay = agent.configure_run(base_merged)
    merged = merge_environment(
        contract=contract,
        request=request,
        app_profile=host.app_profile if host else None,
        binding=host.binding if host else None,
        configure_run_overlay=overlay,
    )

    run_id = str(request.metadata.get("run_id") or request.correlation_id or f"run_{uuid4().hex}")
    trace_id = str(request.metadata.get("trace_id") or run_id)

    await agent.on_run_start(merged)

    task_id = str(request.metadata.get("task_id") or run_id)
    kernel_ctx = StepKernelContext(
        agent_id=merged.agent_id,
        run_id=run_id,
        task_id=task_id,
        tenant_id=merged.tenant_id,
        side_effect_mode=merged.side_effect_mode,
        max_steps=merged.max_steps,
        checkpoint_every_step=merged.checkpoint_every_step,
        policy_engine=PolicyEngine(),
        organizational=merged.organizational,
        state_root=_initial_state_root(request),
        run_trace=AgentRunTrace(run_id=run_id),
    )

    llm_router = StepLLMRouter(
        allowed_models=tuple(merged.allowed_llm_models),
        default_model=merged.default_llm_model,
    )
    shared_context = load_view(request.metadata) or view_from_task_metadata(
        request.metadata,
        task_id=task_id,
    )

    step_ctx = AgentStepContext(
        step_index=0,
        run_id=run_id,
        agent_id=merged.agent_id,
        contract_id=merged.contract_id,
        side_effect_mode=merged.side_effect_mode,
        state_snapshot=dict(kernel_ctx.state_root),
        metadata={
            **merged.merged_metadata,
            AcpRunContextKey.RUN_INPUT: request.input,
            "memory_namespace": merged.memory_namespace,
            "memory_scope": merged.memory_scope.value,
            "allowed_tools": list(merged.allowed_tools),
            AcpRunContextKey.ORGANIZATIONAL: (
                merged.organizational.model_dump(mode="json")
                if merged.organizational is not None
                else None
            ),
        },
        llm_router=llm_router,
        shared_context=shared_context,
    )

    max_iterations = merged.max_steps or contract.max_steps or 32
    last_outcome = None
    last_record = None

    for _ in range(max_iterations):
        outcome, record = await AgentRuntime.advance_step(agent, step_ctx, kernel_ctx)
        last_outcome = outcome
        last_record = record
        if record.error_code == AgentRunErrorCode.MAX_STEPS_EXCEEDED:
            break
        if record.error_code is not None and not record.outcome_applied:
            return _failed_result(
                run_id=run_id,
                trace_id=trace_id,
                merged=merged,
                kernel_ctx=kernel_ctx,
                errors=outcome.errors
                or [
                    AgentRunError(
                        code=record.error_code,
                        message=record.error_code.value,
                    )
                ],
                duration_ms=int((time.perf_counter() - started) * 1000),
                terminal_reason=TerminalReason.ERROR,
            )
        if outcome.is_terminal or outcome.next_action == StepNextAction.PAUSE_HITL:
            break
        step_ctx = step_ctx.model_copy(
            update={
                "step_index": step_ctx.step_index + 1,
                "state_snapshot": dict(kernel_ctx.state_root),
            },
        )

    if last_outcome is None or last_record is None:
        return _failed_result(
            run_id=run_id,
            trace_id=trace_id,
            merged=merged,
            kernel_ctx=kernel_ctx,
            errors=[
                AgentRunError(
                    code=AgentRunErrorCode.INTERNAL_ERROR,
                    message="session loop produced no outcome",
                )
            ],
            duration_ms=int((time.perf_counter() - started) * 1000),
            terminal_reason=TerminalReason.ERROR,
        )

    duration_ms = int((time.perf_counter() - started) * 1000)
    status = _terminal_status(last_outcome.is_terminal, last_outcome.next_action)
    terminal_reason = last_outcome.terminal_reason or TerminalReason.GOAL_MET
    if last_record.budget_exceeded:
        status = AgentRunStatus.FAILED
        terminal_reason = TerminalReason.MAX_STEPS_EXCEEDED

    if step_ctx.shared_context is not None:
        persist_view(request.metadata, step_ctx.shared_context)

    result = AgentRunResult(
        status=status,
        output=last_outcome.output or "",
        state=dict(kernel_ctx.state_root),
        errors=list(last_outcome.errors),
        trace_id=trace_id,
        run_id=run_id,
        trace=kernel_ctx.run_trace,
        terminal_reason=terminal_reason,
        duration_ms=duration_ms,
        compliance_summary=build_compliance_summary(kernel_ctx.run_trace),
        structured_data={
            AcpStructuredDataKey.TRACE_SUMMARY: _trace_summary_payload(
                kernel_ctx.run_trace,
                terminal_reason=terminal_reason,
            ),
        },
    )
    validation = agent.validate_output(result)
    if not validation.valid:
        return _failed_result(
            run_id=run_id,
            trace_id=trace_id,
            merged=merged,
            kernel_ctx=kernel_ctx,
            errors=[
                AgentRunError(
                    code=AgentRunErrorCode.VALIDATION_FAILED,
                    message=error,
                )
                for error in validation.errors
            ],
            duration_ms=duration_ms,
            terminal_reason=TerminalReason.VALIDATION_FAILED,
        )

    await agent.on_run_end(result)
    return result


def _trace_summary_payload(
    trace: AgentRunTrace,
    *,
    terminal_reason: TerminalReason,
) -> dict[str, object]:
    return {
        "total_steps": trace.total_steps,
        "total_llm_tokens": trace.total_llm_tokens,
        "total_tool_calls": trace.total_tool_calls,
        "total_rag_calls": trace.total_rag_calls,
        "terminal_reason": terminal_reason.value,
    }


def _failed_result(
    *,
    run_id: str,
    trace_id: str,
    merged: EffectiveAgentRunEnvironment,
    kernel_ctx: StepKernelContext,
    errors: list[AgentRunError],
    duration_ms: int,
    terminal_reason: TerminalReason,
) -> AgentRunResult:
    _ = merged
    return AgentRunResult(
        status=AgentRunStatus.FAILED,
        output="",
        state=dict(kernel_ctx.state_root),
        errors=errors,
        trace_id=trace_id,
        run_id=run_id,
        trace=kernel_ctx.run_trace,
        terminal_reason=terminal_reason,
        duration_ms=duration_ms,
    )
