# © Artur Czarnecki. All rights reserved.

"""Agent.run(AgentRunRequest) session loop (architecture §29.4 · ACP-DX-3)."""

from __future__ import annotations

import time
from typing import Any

from intergrax.agents.authoring.acp_session_host import (
    ACP_HOST_CONTEXT_KEY,
    ACPSessionHostContext,
)
from intergrax.agents.authoring.budget_enforcing_llm_router import wrap_budget_enforcing_router
from intergrax.agents.authoring.llm_router import StepLLMRouter
from intergrax.agents.authoring.shared_context_bridge import load_view, persist_view, view_from_task_metadata
from intergrax.agents.authoring.step_loop import AgentRuntime
from intergrax.runtime.wiring.reliability_runtime_bridge import resolve_reliability_wiring_options
from intergrax.agents.authoring.artifact_refs import artifact_refs_from_payloads
from intergrax.agents.compliance_summary import build_compliance_summary
from intergrax.agents.persistence.catalog_declarative_invoker import (
    CatalogDeclarativeToolInvoker,
)
from intergrax.agents.persistence.session_persistence import (
    make_checkpoint_hook,
    resolve_session_persistence,
)
from intergrax.agents.persistence.compensation_queue_wiring import (
    resolve_compensation_queue_from_metadata,
)
from intergrax.agents.persistence.idempotency_store_wiring import (
    resolve_idempotency_store_from_metadata,
)
from intergrax.agents.persistence.tool_invoker_wiring import (
    resolve_declarative_tool_invoker_from_metadata,
)
from intergrax.agents.configure_run_strict import ConfigureRunStrictViolation
from intergrax.agents.run_environment import EffectiveAgentRunEnvironment, merge_environment
from intergrax.tools.tool_execution_profile import build_profile_map
from intergrax.contracts.acp_metadata_keys import AcpMetadataKey, AcpRunContextKey, AcpStructuredDataKey
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.agents.acp_token_metering_bridge import (
    initial_invocation_usage,
    seed_state_root_budget_limits,
)
from intergrax.contracts.acp_state import ACP_STATE_KEY, ACP_USAGE_KEY
from intergrax.contracts.agent_run import AgentRunError, AgentRunRequest, AgentRunResult
from intergrax.contracts.agent_run_trace import AgentRunTrace
from intergrax.contracts.agent_run_enums import (
    AgentRunErrorCode,
    AgentRunStatus,
    StepNextAction,
    TerminalReason,
)
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.runtime.kernel.session_reliability import AgentSessionReliability
from intergrax.runtime.kernel.step_kernel import StepKernelContext
from intergrax.contracts.execution_identity import (
    AttemptId,
    RunId,
    TaskId,
    bind_active_execution_identity,
    mint_attempt_id,
    mint_run_id,
    mint_task_id,
    reset_active_execution_identity,
    validate_run_id,
    validate_task_id,
)
from intergrax.runtime.policy.policy_engine import PolicyEngine


def _resolve_acp_session_identity(request: AgentRunRequest) -> tuple[TaskId, RunId, AttemptId]:
    """Canonical ACP session identity boundary — mint once when absent; validate when supplied."""
    metadata_run_id = request.metadata.get("run_id")
    if metadata_run_id is not None:
        run_id = validate_run_id(metadata_run_id)
    elif request.correlation_id is not None:
        run_id = validate_run_id(request.correlation_id)
    else:
        run_id = mint_run_id()

    metadata_task_id = request.metadata.get("task_id")
    if metadata_task_id is not None:
        task_id = validate_task_id(metadata_task_id)
    else:
        task_id = mint_task_id()

    return task_id, run_id, mint_attempt_id()


def _malformed_identity_failure(
    request: AgentRunRequest,
    exc: TypeError | ValueError,
    *,
    started: float,
) -> AgentRunResult:
    """Terminal ingress failure — do not publish malformed or replacement execution identity."""
    trace_id = str(request.metadata.get("trace_id") or "")
    return AgentRunResult(
        run_id="",
        trace_id=trace_id,
        status=AgentRunStatus.FAILED,
        terminal_reason=TerminalReason.ERROR,
        errors=[
            AgentRunError(
                code=AgentRunErrorCode.INTERNAL_ERROR,
                message=f"malformed execution identity: {exc}",
            ),
        ],
        trace=AgentRunTrace(),
        duration_ms=int((time.perf_counter() - started) * 1000),
    )


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


def _exec_ctx_from_step(step_ctx: AgentStepContext) -> RuntimeExecutionContext | None:
    raw = step_ctx.metadata.get("uaep_exec_ctx")
    if isinstance(raw, RuntimeExecutionContext):
        return raw
    return None


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
    try:
        task_id, run_id, attempt_id = _resolve_acp_session_identity(request)
    except (TypeError, ValueError) as exc:
        return _malformed_identity_failure(request, exc, started=started)

    contract = agent.get_contract()
    host = _host_context_from_metadata(request.metadata)
    base_merged = merge_environment(
        contract=contract,
        request=request,
        app_profile=host.runtime_profile if host else None,
        binding=host.binding if host else None,
    )
    overlay = agent.configure_run(base_merged)
    try:
        merged = merge_environment(
            contract=contract,
            request=request,
            app_profile=host.runtime_profile if host else None,
            binding=host.binding if host else None,
            configure_run_overlay=overlay,
        )
    except ConfigureRunStrictViolation as exc:
        trace_id = str(request.metadata.get("trace_id") or run_id)
        return AgentRunResult(
            run_id=str(run_id),
            trace_id=trace_id,
            status=AgentRunStatus.FAILED,
            terminal_reason=TerminalReason.POLICY_DENIED,
            errors=[
                AgentRunError(
                    code=AgentRunErrorCode.POLICY_DENIED,
                    message="; ".join(exc.violations),
                    details={"violations": exc.violations},
                ),
            ],
            trace=AgentRunTrace(run_id=str(run_id)),
            duration_ms=int((time.perf_counter() - started) * 1000),
        )

    trace_id = str(request.metadata.get("trace_id") or run_id)

    from intergrax.llm.messages import model_input_messages_from_metadata

    try:
        model_messages = model_input_messages_from_metadata(request.metadata)
    except ValueError as exc:
        return AgentRunResult(
            run_id=str(run_id),
            trace_id=trace_id,
            status=AgentRunStatus.FAILED,
            terminal_reason=TerminalReason.ERROR,
            errors=[
                AgentRunError(
                    code=AgentRunErrorCode.INTERNAL_ERROR,
                    message=str(exc),
                ),
            ],
            trace=AgentRunTrace(run_id=str(run_id)),
            duration_ms=int((time.perf_counter() - started) * 1000),
        )

    identity_token = bind_active_execution_identity(run_id=run_id, attempt_id=attempt_id)
    try:
        return await _run_acp_session_bound(
            agent=agent,
            request=request,
            merged=merged,
            host=host,
            contract=contract,
            task_id=task_id,
            run_id=run_id,
            trace_id=trace_id,
            model_messages=model_messages,
            started=started,
        )
    finally:
        reset_active_execution_identity(identity_token)


async def _run_acp_session_bound(
    *,
    agent: object,
    request: AgentRunRequest,
    merged: EffectiveAgentRunEnvironment,
    host: ACPSessionHostContext | None,
    contract: object,
    task_id: TaskId,
    run_id: RunId,
    trace_id: str,
    model_messages: object,
    started: float,
) -> AgentRunResult:
    await agent.on_run_start(merged)

    persistence, resume = resolve_session_persistence(
        request,
        run_id=run_id,
        tenant_id=merged.tenant_id,
    )
    state_root = _initial_state_root(request)
    start_step_index = 0
    if resume is not None:
        state_root = resume.state_root
        start_step_index = resume.start_step_index
    else:
        state_root = seed_state_root_budget_limits(state_root, merged.resolved_budget_limits)

    run_trace = AgentRunTrace(run_id=str(run_id))
    reliability: AgentSessionReliability | None = None
    if host is not None and host.runtime_profile is not None:
        reliability = AgentSessionReliability.from_wiring_options(
            resolve_reliability_wiring_options(host.runtime_profile.reliability_profile)
        )
    declarative_invoker = None
    if host is not None and host.declarative_tool_invoker is not None:
        declarative_invoker = resolve_declarative_tool_invoker_from_metadata(
            {AcpMetadataKey.DECLARATIVE_TOOL_INVOKER: host.declarative_tool_invoker},
        )
    if declarative_invoker is None:
        declarative_invoker = resolve_declarative_tool_invoker_from_metadata(request.metadata)
    if isinstance(declarative_invoker, CatalogDeclarativeToolInvoker):
        declarative_invoker.bind_run(
            run_id=run_id,
            task_id=task_id,
            agent_id=merged.agent_id,
            tenant_id=merged.tenant_id,
            user_id=str(request.identity.user_id or ""),
        )

    kernel_ctx = StepKernelContext(
        agent_id=merged.agent_id,
        run_id=run_id,
        task_id=task_id,
        tenant_id=merged.tenant_id,
        side_effect_mode=merged.side_effect_mode,
        max_steps=merged.max_steps,
        checkpoint_every_step=merged.checkpoint_every_step,
        policy_engine=PolicyEngine(),
        production_mode=(
            host.runtime_profile.execution_mode.value == "strict"
            if host is not None and host.runtime_profile is not None
            else False
        ),
        organizational=merged.organizational,
        side_effect_ledger=persistence.side_effect_ledger,
        declarative_tool_invoker=declarative_invoker,
        compensation_queue=resolve_compensation_queue_from_metadata(request.metadata),
        idempotency_store=resolve_idempotency_store_from_metadata(request.metadata),
        tool_profiles=build_profile_map(list(contract.extra_tools)),
        reliability=reliability,
        state_root=state_root,
        run_trace=run_trace,
        resolved_budget_limits=merged.resolved_budget_limits,
        budget_reaction=merged.budget_reaction,
        notification_adapter=host.notification_adapter if host is not None else None,
        budget_reaction_hook=host.budget_reaction_hook if host is not None else None,
    )
    kernel_ctx_holder: list[StepKernelContext] = [kernel_ctx]
    kernel_ctx.checkpoint_hook = make_checkpoint_hook(
        persistence=persistence,
        run_id=run_id,
        tenant_id=merged.tenant_id,
        agent_id=merged.agent_id,
        trace_step_count_fn=lambda: len(kernel_ctx.run_trace.steps),
    )

    router_runtime_config = None
    if host is not None and host.runtime_profile is not None:
        from intergrax.runtime.wiring.context_runtime_bridge import (
            apply_context_profile_to_runtime_config,
        )
        from intergrax.runtime.wiring.llm_resolver import resolve_llm_adapter
        from intergrax.llm_adapters.routing.context_bridge import build_routing_context_from_runtime
        from intergrax.runtime.nexus.config import RuntimeConfig

        acp_routing_context = build_routing_context_from_runtime(
            tenant_id=merged.tenant_id,
            agent_id=merged.agent_id,
            metadata=request.metadata,
            budget_limits=merged.resolved_budget_limits,
        )
        router_runtime_config = RuntimeConfig(
            llm_adapter=resolve_llm_adapter(
                host.runtime_profile,
                routing_context=acp_routing_context,
            ),
            production_mode=host.runtime_profile.execution_mode.value == "strict",
            llm_routing_context=acp_routing_context,
        )
        apply_context_profile_to_runtime_config(
            router_runtime_config,
            host.runtime_profile.context_profile,
        )
        from intergrax.runtime.wiring.attestation_runtime_bridge import (
            apply_attestation_profile_to_runtime_config,
        )
        from intergrax.runtime.attestation.kernel_wiring import apply_boundary_export_to_kernel

        apply_attestation_profile_to_runtime_config(
            router_runtime_config,
            host.runtime_profile,
        )
        apply_boundary_export_to_kernel(kernel_ctx_holder[0], router_runtime_config)

    base_llm_router = StepLLMRouter(
        allowed_models=tuple(merged.allowed_llm_models),
        default_model=merged.default_llm_model,
        runtime_config=router_runtime_config,
        llm_adapter=router_runtime_config.llm_adapter if router_runtime_config is not None else None,
        require_real_llm=(
            host.runtime_profile.execution_mode.value == "strict"
            if host is not None and host.runtime_profile is not None
            else False
        ),
        model_input_messages=model_messages,
    )
    step_ctx_holder: list[AgentStepContext] = []
    llm_router = wrap_budget_enforcing_router(
        base_llm_router,
        limits=merged.resolved_budget_limits,
        usage_provider=lambda: (
            step_ctx_holder[0].invocation_usage if step_ctx_holder else None
        ),
        degrade_provider=lambda: kernel_ctx_holder[0].budget_degrade_active,
    )
    if host is not None and host.runtime_profile is not None and host.runtime_profile.llm_routing_profile is not None:
        from intergrax.agents.authoring.dynamic_llm_router import wrap_dynamic_llm_router
        from intergrax.agents.authoring.acp_routing_trace_bridge import (
            record_acp_routing_rule_evaluation,
        )
        from intergrax.runtime.wiring.llm_routing_context_bridge import (
            make_acp_routing_context_provider,
        )
        from intergrax.llm_adapters.routing.contracts import RoutingEvaluation

        def _on_routing_evaluated(evaluation: RoutingEvaluation) -> None:
            record_acp_routing_rule_evaluation(kernel_ctx_holder[0], evaluation)

        llm_router = wrap_dynamic_llm_router(
            llm_router,
            routing_profile=host.runtime_profile.llm_routing_profile,
            context_provider=make_acp_routing_context_provider(
                kernel_ctx_holder=kernel_ctx_holder,
                step_ctx_holder=step_ctx_holder,
                tenant_id=merged.tenant_id,
                agent_id=merged.agent_id,
                task_class=str(request.metadata.get("task_class", "")) or None,
                metadata=request.metadata,
            ),
            on_evaluated=_on_routing_evaluated,
        )
    shared_context = load_view(request.metadata) or view_from_task_metadata(
        request.metadata,
        task_id=task_id,
    )

    step_metadata = {
        **merged.merged_metadata,
        AcpRunContextKey.RUN_INPUT: request.input,
        AcpRunContextKey.TENANT_ID: merged.tenant_id,
        "memory_namespace": merged.memory_namespace,
        "memory_scope": merged.memory_scope.value,
        "allowed_tools": (
            list(merged.allowed_tools)
            or [
                str(tool_id)
                for tool_id in (request.metadata.get("allowed_tools") or [])
                if str(tool_id).strip()
            ]
        ),
        AcpRunContextKey.ORGANIZATIONAL: (
            merged.organizational.model_dump(mode="json")
            if merged.organizational is not None
            else None
        ),
        **(
            {AcpRunContextKey.CRITIC_HOOKS: host.critic_graph_hooks}
            if host is not None and host.critic_graph_hooks is not None
            else {}
        ),
    }
    hints = contract.context_hints
    step_ctx = AgentStepContext(
        step_index=start_step_index,
        run_id=run_id,
        task_id=task_id,
        tenant_id=merged.tenant_id,
        message=str(request.input or ""),
        step_kind=hints.step_kind if hints is not None else None,
        agent_id=merged.agent_id,
        contract_id=merged.contract_id,
        side_effect_mode=merged.side_effect_mode,
        state_snapshot=dict(kernel_ctx.state_root),
        metadata=step_metadata,
        llm_router=llm_router,
        shared_context=shared_context,
        invocation_usage=initial_invocation_usage(
            kernel_ctx.state_root,
            step_metadata,
            merged.resolved_budget_limits,
        ),
    )
    step_ctx_holder.append(step_ctx)

    max_iterations = merged.max_steps or contract.max_steps or 32
    last_outcome = None
    last_record = None

    from intergrax.agents.authoring.acp_uaep_shim import attach_acp_catalog_exec_ctx

    for _ in range(max_iterations):
        attach_acp_catalog_exec_ctx(
            step_ctx,
            kernel_ctx=kernel_ctx,
            request=request,
            contract=contract,
        )
        outcome, record = await AgentRuntime.advance_step(agent, step_ctx, kernel_ctx)
        last_outcome = outcome
        last_record = record
        if record.error_code == AgentRunErrorCode.MAX_STEPS_EXCEEDED:
            break
        if record.error_code is not None and not record.outcome_applied:
            fail_terminal = (
                TerminalReason.BUDGET_EXCEEDED
                if record.error_code == AgentRunErrorCode.BUDGET_EXCEEDED
                else TerminalReason.ERROR
            )
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
                terminal_reason=fail_terminal,
            )
        if outcome.is_terminal or outcome.next_action == StepNextAction.PAUSE_HITL:
            break
        step_ctx = step_ctx.model_copy(
            update={
                "step_index": step_ctx.step_index + 1,
                "state_snapshot": dict(kernel_ctx.state_root),
                "invocation_usage": step_ctx.invocation_usage,
            },
        )
        step_ctx.metadata.pop("uaep_exec_ctx", None)
        step_ctx_holder[0] = step_ctx

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
    if last_record.error_code == AgentRunErrorCode.BUDGET_EXCEEDED:
        status = AgentRunStatus.FAILED
        terminal_reason = TerminalReason.BUDGET_EXCEEDED
    elif last_record.budget_exceeded:
        status = AgentRunStatus.FAILED
        terminal_reason = TerminalReason.MAX_STEPS_EXCEEDED

    if step_ctx.shared_context is not None:
        persist_view(request.metadata, step_ctx.shared_context)
    if ACP_USAGE_KEY in step_ctx.metadata:
        request.metadata[ACP_USAGE_KEY] = step_ctx.metadata[ACP_USAGE_KEY]

    artifact_refs = artifact_refs_from_payloads(
        list(last_outcome.artifacts),
        run_id=run_id,
        trace_id=trace_id,
        agent_id=merged.agent_id,
        step_index=step_ctx.step_index,
    )
    from intergrax.runtime.workspace.exec_ctx_isolation import isolation_structured_data_from_exec_ctx

    structured_data: dict[str, Any] = {
        AcpStructuredDataKey.TRACE_SUMMARY: _trace_summary_payload(
            kernel_ctx.run_trace,
            terminal_reason=terminal_reason,
        ),
    }
    structured_data.update(isolation_structured_data_from_exec_ctx(_exec_ctx_from_step(step_ctx)))
    # Preserve typed domain summaries for TaskResult / product handoff (LKW search/index).
    if isinstance(last_outcome.output, dict):
        for key in ("search_summary", "ingest_summary", "domain_summary"):
            value = last_outcome.output.get(key)
            if isinstance(value, dict) and value:
                structured_data[key] = dict(value)

    result = AgentRunResult(
        status=status,
        output=last_outcome.output if last_outcome.output is not None else "",
        state=dict(kernel_ctx.state_root),
        artifacts=list(last_outcome.artifacts),
        artifact_refs=artifact_refs,
        errors=list(last_outcome.errors),
        trace_id=trace_id,
        run_id=run_id,
        trace=kernel_ctx.run_trace,
        terminal_reason=terminal_reason,
        duration_ms=duration_ms,
        compliance_summary=build_compliance_summary(kernel_ctx.run_trace),
        structured_data=structured_data,
    )
    if status == AgentRunStatus.PAUSED:
        await agent.on_run_end(result)
        return result

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
    from intergrax.agents.authoring.diagnostic_serialization import aggregate_step_diagnostics

    return {
        "total_steps": trace.total_steps,
        "total_llm_tokens": trace.total_llm_tokens,
        "total_tool_calls": trace.total_tool_calls,
        "total_rag_calls": trace.total_rag_calls,
        "terminal_reason": terminal_reason.value,
        "step_diagnostics": aggregate_step_diagnostics(trace),
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
