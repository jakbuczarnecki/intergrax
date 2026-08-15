# © Artur Czarnecki. All rights reserved.

"""HarnessKernel L1 step cycle (architecture §38 · ACP-STEP-2b)."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from intergrax.agents.authoring.side_effect_validation import validate_side_effect_mode
from intergrax.agents.authoring.state_merge import extract_acp_state_blob, merge_session_state
from intergrax.agents.authoring.step_outcome import StepOutcome
from intergrax.contracts.agent_run_enums import (
    AgentRunErrorCode,
    SideEffectMode,
    StepNextAction,
    TerminalReason,
)
from intergrax.contracts.execution_identity import (
    require_active_execution_identity,
    validate_run_id,
    validate_task_id,
)
from intergrax.contracts.privacy_redaction import redact_pii_text
from intergrax.contracts.agent_run_trace import (
    AgentRunTrace,
    AgentStepRecord,
    AgentStepStatus,
    PolicyCheckPhase,
    PolicyVerdictRecord,
    RagCallRecord,
    ToolCallRecord,
)
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.agents.acp_token_metering_bridge import apply_llm_metering_after_step
from intergrax.contracts.acp_budget_enforcement import (
    evaluate_hard_budget_violation,
    is_budget_exceeded_outcome,
)
from intergrax.contracts.agent_budget import ResolvedBudgetLimits
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.agents.persistence.compensation_enqueue import (
    enqueue_compensations_for_step_failure,
)
from intergrax.agents.persistence.declarative_tool_executor import (
    DeclarativeToolInvoker,
    execute_declarative_actions,
)
from intergrax.contracts.side_effect import CompensationRequest
from intergrax.agents.persistence.compensation_queue_store import CompensationQueueStore
from intergrax.agents.persistence.side_effect_ledger import SideEffectLedger
from intergrax.contracts.idempotency_store import IdempotencyStore
from intergrax.agents.persistence.tool_action_validation import (
    ToolActionValidationError,
    validate_requested_actions,
)
from intergrax.contracts.org_policy import OrganizationalPolicyContext
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.tools.tool_execution_profile import ToolExecutionProfile
from intergrax.runtime.kernel.session_reliability import AgentSessionReliability
from intergrax.runtime.policy.org_enforcement import (
    evaluate_org_policy_pre,
    extract_requested_tool_ids,
)
from intergrax.contracts.step_execution import StepExecutionRecord
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.policy.policy_engine import PolicyEngine
from intergrax.runtime.attestation.buffer import BoundaryEventBuffer
from intergrax.runtime.attestation.settings import ExecutionBoundaryExportRuntimeSettings

EventEmitter = Callable[[RuntimeEvent], Awaitable[None]]
CheckpointHook = Callable[[dict[str, Any], int], Awaitable[None]]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class StepKernelContext:
    """Harness-owned execution context for one agent session loop."""

    agent_id: str
    run_id: str = ""
    task_id: str = ""
    tenant_id: str = "default"
    side_effect_mode: SideEffectMode = SideEffectMode.IMMEDIATE
    max_steps: int | None = None
    checkpoint_every_step: bool = True
    policy_engine: PolicyEngine | None = None
    production_mode: bool = False
    allow_permissive_missing_policy: bool = False
    organizational: OrganizationalPolicyContext | None = None
    side_effect_ledger: SideEffectLedger | None = None
    declarative_tool_invoker: DeclarativeToolInvoker | None = None
    compensation_queue: CompensationQueueStore | None = None
    idempotency_store: IdempotencyStore | None = None
    compensation_requests: list[CompensationRequest] = field(default_factory=list)
    tool_profiles: dict[str, ToolExecutionProfile] = field(default_factory=dict)
    reliability: AgentSessionReliability | None = None
    emit_event: EventEmitter | None = None
    checkpoint_hook: CheckpointHook | None = None
    state_root: dict[str, Any] = field(default_factory=dict)
    run_trace: AgentRunTrace = field(default_factory=AgentRunTrace)
    events: list[RuntimeEvent] = field(default_factory=list)
    resolved_budget_limits: ResolvedBudgetLimits = field(
        default_factory=ResolvedBudgetLimits
    )
    budget_reaction: Any = None
    notification_adapter: Any = None
    budget_reaction_hook: Any = None
    budget_threshold_emitted: set[str] = field(default_factory=set)
    budget_degrade_active: bool = False
    routing_rule_evaluations: list[dict[str, Any]] = field(default_factory=list)
    execution_boundary_export: ExecutionBoundaryExportRuntimeSettings | None = None
    boundary_event_buffer: BoundaryEventBuffer | None = None


def _missing_policy_engine_decision(kernel_ctx: StepKernelContext) -> PolicyDecision:
    """Fail closed unless dev/test explicitly opts into permissive missing-policy wiring."""
    if (
        not kernel_ctx.production_mode
        and kernel_ctx.allow_permissive_missing_policy
    ):
        return PolicyDecision(
            action=PolicyAction.ALLOW,
            reason="permissive_missing_policy_engine",
            policy_rule_id="kernel.permissive_missing_policy",
        )
    return PolicyDecision(
        action=PolicyAction.DENY,
        reason="missing_policy_engine",
        policy_rule_id="kernel.missing_policy_engine",
    )


class HarnessKernel:
    """Deterministic harness primitive — policy, merge, budgets, trace; no domain planning."""

    @staticmethod
    async def execute_step(
        outcome: StepOutcome,
        step_ctx: AgentStepContext,
        kernel_ctx: StepKernelContext,
    ) -> StepExecutionRecord:
        started = time.perf_counter()
        trace_events = 0

        if kernel_ctx.reliability is not None and kernel_ctx.reliability.circuit_open:
            record = StepExecutionRecord(
                step_index=step_ctx.step_index,
                outcome_applied=False,
                error_code=AgentRunErrorCode.INTERNAL_ERROR,
                step_record=HarnessKernel._build_step_record(
                    step_ctx=step_ctx,
                    outcome=outcome,
                    state_version=int(
                        extract_acp_state_blob(kernel_ctx.state_root).get("_version", 0)
                    ),
                    error_code=AgentRunErrorCode.INTERNAL_ERROR,
                    terminal_reason=TerminalReason.ERROR,
                    next_action=StepNextAction.FAIL,
                    diagnostics={"circuit_breaker": "open"},
                    finished_at=_utc_now(),
                ),
            )
            return await HarnessKernel._finish_step(kernel_ctx, step_ctx, record)

        mode_error = validate_side_effect_mode(outcome, kernel_ctx.side_effect_mode)
        if mode_error is not None:
            record = StepExecutionRecord(
                step_index=step_ctx.step_index,
                outcome_applied=False,
                side_effect_mode_violation=True,
                error_code=AgentRunErrorCode.VALIDATION_FAILED,
                step_record=HarnessKernel._build_step_record(
                    step_ctx=step_ctx,
                    outcome=outcome,
                    state_version=int(
                        extract_acp_state_blob(kernel_ctx.state_root).get("_version", 0)
                    ),
                    error_code=AgentRunErrorCode.VALIDATION_FAILED,
                    terminal_reason=TerminalReason.VALIDATION_FAILED,
                    next_action=StepNextAction.FAIL,
                    diagnostics={"side_effect_mode": mode_error},
                    finished_at=_utc_now(),
                ),
            )
            return await HarnessKernel._finish_step(kernel_ctx, step_ctx, record)

        policy_pre = HarnessKernel._policy_pre_check(outcome, step_ctx, kernel_ctx)
        trace_events += await HarnessKernel._emit_policy(kernel_ctx, policy_pre, phase="pre")
        if is_budget_exceeded_outcome(outcome):
            record = StepExecutionRecord(
                step_index=step_ctx.step_index,
                outcome_applied=False,
                policy_pre=policy_pre,
                error_code=AgentRunErrorCode.BUDGET_EXCEEDED,
                trace_event_count=trace_events,
                step_record=HarnessKernel._build_step_record(
                    step_ctx=step_ctx,
                    outcome=outcome,
                    state_version=int(
                        extract_acp_state_blob(kernel_ctx.state_root).get("_version", 0)
                    ),
                    policy_pre=policy_pre,
                    error_code=AgentRunErrorCode.BUDGET_EXCEEDED,
                    terminal_reason=TerminalReason.BUDGET_EXCEEDED,
                    next_action=StepNextAction.FAIL,
                    finished_at=_utc_now(),
                ),
            )
            return await HarnessKernel._finish_step(kernel_ctx, step_ctx, record)
        if outcome.errors and kernel_ctx.reliability is not None:
            for error in outcome.errors:
                kernel_ctx.reliability.record_failure(error.code)
        if policy_pre.action == PolicyAction.DENY:
            record = StepExecutionRecord(
                step_index=step_ctx.step_index,
                outcome_applied=False,
                policy_pre=policy_pre,
                error_code=AgentRunErrorCode.POLICY_DENIED,
                trace_event_count=trace_events,
                step_record=HarnessKernel._build_step_record(
                    step_ctx=step_ctx,
                    outcome=outcome,
                    state_version=int(
                        extract_acp_state_blob(kernel_ctx.state_root).get("_version", 0)
                    ),
                    policy_pre=policy_pre,
                    error_code=AgentRunErrorCode.POLICY_DENIED,
                    terminal_reason=TerminalReason.POLICY_DENIED,
                    next_action=StepNextAction.FAIL,
                    finished_at=_utc_now(),
                ),
            )
            return await HarnessKernel._finish_step(kernel_ctx, step_ctx, record)

        merge_result = merge_session_state(
            kernel_ctx.state_root,
            outcome.state_delta,
            incoming_version=extract_acp_state_blob(kernel_ctx.state_root).get("_version"),
        )
        if merge_result.error_code is not None:
            record = StepExecutionRecord(
                step_index=step_ctx.step_index,
                outcome_applied=False,
                policy_pre=policy_pre,
                error_code=merge_result.error_code,
                trace_event_count=trace_events,
                step_record=HarnessKernel._build_step_record(
                    step_ctx=step_ctx,
                    outcome=outcome,
                    state_version=int(
                        extract_acp_state_blob(kernel_ctx.state_root).get("_version", 0)
                    ),
                    policy_pre=policy_pre,
                    error_code=merge_result.error_code,
                    terminal_reason=TerminalReason.VALIDATION_FAILED,
                    next_action=StepNextAction.FAIL,
                    diagnostics={"merge_error": merge_result.error_message},
                    finished_at=_utc_now(),
                ),
            )
            return await HarnessKernel._finish_step(kernel_ctx, step_ctx, record)

        kernel_ctx.state_root = merge_result.state
        step_ctx.state_snapshot = merge_result.state
        state_version = int(extract_acp_state_blob(merge_result.state).get("_version", 0))
        tool_execution_diagnostics: dict[str, Any] | None = None
        step_action_args: dict[str, dict[str, Any]] = {}

        if outcome.requested_actions:
            try:
                normalized_actions = validate_requested_actions(
                    requested_actions=outcome.requested_actions,
                    side_effect_mode=kernel_ctx.side_effect_mode,
                    tool_profiles=kernel_ctx.tool_profiles,
                    run_id=kernel_ctx.run_id,
                    step_index=step_ctx.step_index,
                    ledger=kernel_ctx.side_effect_ledger,
                    idempotency_store=kernel_ctx.idempotency_store,
                    tenant_id=kernel_ctx.tenant_id,
                )
            except ToolActionValidationError as exc:
                record = StepExecutionRecord(
                    step_index=step_ctx.step_index,
                    outcome_applied=False,
                    policy_pre=policy_pre,
                    error_code=AgentRunErrorCode.VALIDATION_FAILED,
                    trace_event_count=trace_events,
                    step_record=HarnessKernel._build_step_record(
                        step_ctx=step_ctx,
                        outcome=outcome,
                        state_version=state_version,
                        policy_pre=policy_pre,
                        error_code=AgentRunErrorCode.VALIDATION_FAILED,
                        terminal_reason=TerminalReason.VALIDATION_FAILED,
                        next_action=StepNextAction.FAIL,
                        diagnostics={
                            "tool_validation": exc.code,
                            "tool_id": exc.tool_id,
                            "message": exc.message,
                        },
                        finished_at=_utc_now(),
                    ),
                )
                return await HarnessKernel._finish_step(kernel_ctx, step_ctx, record)
            step_action_args = {
                str(action["tool_id"]): (
                    action.get("args") if isinstance(action.get("args"), dict) else {}
                )
                for action in normalized_actions
                if isinstance(action.get("tool_id"), str)
            }
            if (
                kernel_ctx.side_effect_mode == SideEffectMode.DECLARATIVE
                and normalized_actions
            ):
                trace_events += await HarnessKernel._emit(
                    kernel_ctx,
                    RuntimeEventType.TOOL_REQUESTED,
                    {
                        "mode": "declarative",
                        "action_count": len(normalized_actions),
                        "replay_skipped": sum(
                            1 for action in normalized_actions if action.get("replay_skipped")
                        ),
                    },
                )
                execution = await execute_declarative_actions(
                    actions=normalized_actions,
                    ledger=kernel_ctx.side_effect_ledger,
                    invoker=kernel_ctx.declarative_tool_invoker,
                    idempotency_store=kernel_ctx.idempotency_store,
                    tenant_id=kernel_ctx.tenant_id,
                )
                tool_execution_diagnostics = {
                    "declarative_tool_execution": [
                        {
                            "tool_id": item.tool_id,
                            "status": item.status,
                            "idempotency_key": item.idempotency_key,
                            "replay_skipped": item.replay_skipped,
                            "external_ref": item.external_ref,
                            "error": item.error,
                        }
                        for item in execution.results
                    ],
                }
                for item in execution.results:
                    if item.status == "replay_skipped":
                        event_type = RuntimeEventType.TOOL_COMPLETED
                        status = "replay_skipped"
                    elif item.status == "success":
                        event_type = RuntimeEventType.TOOL_COMPLETED
                        status = "success"
                    elif item.status == "denied":
                        event_type = RuntimeEventType.TOOL_DENIED
                        status = "denied"
                    elif item.status in ("failed",):
                        event_type = RuntimeEventType.TOOL_FAILED
                        status = "failed"
                    else:
                        continue
                    trace_events += await HarnessKernel._emit(
                        kernel_ctx,
                        event_type,
                        {
                            "tool_name": item.tool_id,
                            "status": status,
                            "duration_ms": item.duration_ms,
                            "idempotency_key": item.idempotency_key,
                            "replay_skipped": item.replay_skipped,
                        },
                    )
                failed_tool_id = execution.failed_tool_id
                if failed_tool_id is not None:
                    record = StepExecutionRecord(
                        step_index=step_ctx.step_index,
                        outcome_applied=False,
                        policy_pre=policy_pre,
                        state_version=state_version,
                        error_code=AgentRunErrorCode.TOOL_FAILED,
                        trace_event_count=trace_events,
                        step_record=HarnessKernel._build_step_record(
                            step_ctx=step_ctx,
                            outcome=outcome,
                            state_version=state_version,
                            policy_pre=policy_pre,
                            error_code=AgentRunErrorCode.TOOL_FAILED,
                            terminal_reason=TerminalReason.ERROR,
                            next_action=StepNextAction.FAIL,
                            diagnostics={
                                **(tool_execution_diagnostics or {}),
                                "tool_id": failed_tool_id,
                            },
                            finished_at=_utc_now(),
                        ),
                    )
                    return await HarnessKernel._finish_step(kernel_ctx, step_ctx, record)

        policy_post = HarnessKernel._policy_post_check(outcome, step_ctx, kernel_ctx)
        trace_events += await HarnessKernel._emit_policy(kernel_ctx, policy_post, phase="post")
        if policy_post.action == PolicyAction.DENY:
            compensation_diagnostics, compensation_events = (
                await HarnessKernel._enqueue_step_failure_compensations(
                    kernel_ctx=kernel_ctx,
                    step_ctx=step_ctx,
                    action_args=step_action_args or None,
                )
            )
            trace_events += compensation_events
            failure_diagnostics: dict[str, Any] = {}
            if tool_execution_diagnostics:
                failure_diagnostics.update(tool_execution_diagnostics)
            if compensation_diagnostics:
                failure_diagnostics.update(compensation_diagnostics)
            record = StepExecutionRecord(
                step_index=step_ctx.step_index,
                outcome_applied=False,
                policy_pre=policy_pre,
                policy_post=policy_post,
                state_version=state_version,
                error_code=AgentRunErrorCode.POLICY_DENIED,
                trace_event_count=trace_events,
                step_record=HarnessKernel._build_step_record(
                    step_ctx=step_ctx,
                    outcome=outcome,
                    state_version=state_version,
                    policy_pre=policy_pre,
                    policy_post=policy_post,
                    error_code=AgentRunErrorCode.POLICY_DENIED,
                    terminal_reason=TerminalReason.POLICY_DENIED,
                    next_action=StepNextAction.FAIL,
                    diagnostics=failure_diagnostics or None,
                    finished_at=_utc_now(),
                ),
            )
            return await HarnessKernel._finish_step(kernel_ctx, step_ctx, record)

        budget_exceeded = HarnessKernel._budget_exceeded(step_ctx, kernel_ctx)
        if budget_exceeded:
            trace_events += await HarnessKernel._emit(
                kernel_ctx,
                RuntimeEventType.STEP_FAILED,
                {
                    "step_index": step_ctx.step_index,
                    "max_steps": kernel_ctx.max_steps,
                    "reason": "max_steps_exceeded",
                },
            )
            record = StepExecutionRecord(
                step_index=step_ctx.step_index,
                outcome_applied=True,
                policy_pre=policy_pre,
                policy_post=policy_post,
                state_version=state_version,
                error_code=AgentRunErrorCode.MAX_STEPS_EXCEEDED,
                budget_exceeded=True,
                trace_event_count=trace_events,
                step_record=HarnessKernel._build_step_record(
                    step_ctx=step_ctx,
                    outcome=outcome,
                    state_version=state_version,
                    policy_pre=policy_pre,
                    policy_post=policy_post,
                    error_code=AgentRunErrorCode.MAX_STEPS_EXCEEDED,
                    terminal_reason=TerminalReason.MAX_STEPS_EXCEEDED,
                    next_action=StepNextAction.FAIL,
                    finished_at=_utc_now(),
                ),
            )
            return await HarnessKernel._finish_step(kernel_ctx, step_ctx, record)

        trace_events += await HarnessKernel._emit(
            kernel_ctx,
            RuntimeEventType.STEP_COMPLETED,
            {
                "step_index": step_ctx.step_index,
                "next_action": outcome.next_action.value,
                "is_terminal": outcome.is_terminal,
            },
        )

        should_checkpoint = kernel_ctx.checkpoint_every_step
        if kernel_ctx.reliability is not None:
            should_checkpoint = should_checkpoint and kernel_ctx.reliability.should_checkpoint(
                step_ctx.step_index
            )
        if should_checkpoint and kernel_ctx.checkpoint_hook is not None:
            await kernel_ctx.checkpoint_hook(kernel_ctx.state_root, step_ctx.step_index)

        if kernel_ctx.reliability is not None:
            kernel_ctx.reliability.record_success()

        duration_ms = int((time.perf_counter() - started) * 1000)
        _ = duration_ms

        record = StepExecutionRecord(
            step_index=step_ctx.step_index,
            outcome_applied=True,
            policy_pre=policy_pre,
            policy_post=policy_post,
            state_version=state_version,
            trace_event_count=trace_events,
            step_record=HarnessKernel._build_step_record(
                step_ctx=step_ctx,
                outcome=outcome,
                state_version=state_version,
                policy_pre=policy_pre,
                policy_post=policy_post,
                diagnostics=tool_execution_diagnostics,
                finished_at=_utc_now(),
            ),
        )
        return await HarnessKernel._finish_step(kernel_ctx, step_ctx, record)

    @staticmethod
    def _build_step_record(
        *,
        step_ctx: AgentStepContext,
        outcome: StepOutcome,
        state_version: int,
        policy_pre: PolicyDecision | None = None,
        policy_post: PolicyDecision | None = None,
        error_code: AgentRunErrorCode | None = None,
        terminal_reason: TerminalReason | None = None,
        next_action: StepNextAction | None = None,
        diagnostics: dict[str, Any] | None = None,
        finished_at: datetime | None = None,
    ) -> AgentStepRecord:
        resolved_action = next_action or outcome.next_action
        if error_code is not None and next_action is None:
            resolved_action = StepNextAction.FAIL

        if error_code is not None:
            status = AgentStepStatus.FAILED
        elif resolved_action == StepNextAction.PAUSE_HITL:
            status = AgentStepStatus.PAUSED
        else:
            status = AgentStepStatus.SUCCEEDED

        resolved_terminal = terminal_reason or outcome.terminal_reason
        verdicts: list[PolicyVerdictRecord] = []
        if policy_pre is not None:
            verdicts.append(
                PolicyVerdictRecord(
                    phase=PolicyCheckPhase.PRE,
                    action=policy_pre.action,
                    reason=redact_pii_text(policy_pre.reason),
                    policy_rule_id=policy_pre.policy_rule_id,
                )
            )
        if policy_post is not None:
            verdicts.append(
                PolicyVerdictRecord(
                    phase=PolicyCheckPhase.POST,
                    action=policy_post.action,
                    reason=redact_pii_text(policy_post.reason),
                    policy_rule_id=policy_post.policy_rule_id,
                )
            )

        llm_calls = []
        if step_ctx.llm_router is not None:
            llm_calls = step_ctx.llm_router.drain_pending_calls()

        tool_calls: list[ToolCallRecord] = []
        rag_calls: list[RagCallRecord] = []
        exec_ctx_raw = step_ctx.metadata.get("uaep_exec_ctx")
        if isinstance(exec_ctx_raw, RuntimeExecutionContext):
            tool_calls = exec_ctx_raw.drain_pending_tool_calls()
            rag_calls = exec_ctx_raw.drain_pending_rag_calls()

        merged_diagnostics = dict(outcome.diagnostics or {})
        if diagnostics:
            merged_diagnostics.update(diagnostics)

        return AgentStepRecord(
            step_id=str(step_ctx.metadata.get("step_id") or f"step-{step_ctx.step_index:04d}"),
            step_index=step_ctx.step_index,
            finished_at=finished_at,
            status=status,
            next_action=resolved_action,
            terminal_reason=resolved_terminal,
            state_version=state_version,
            tool_calls=tool_calls,
            rag_calls=rag_calls,
            llm_calls=llm_calls,
            policy_verdicts=verdicts,
            diagnostics=merged_diagnostics,
            error_code=error_code,
        )

    @staticmethod
    def _policy_pre_check(
        outcome: StepOutcome,
        step_ctx: AgentStepContext,
        kernel_ctx: StepKernelContext,
    ) -> PolicyDecision:
        if kernel_ctx.policy_engine is None:
            return _missing_policy_engine_decision(kernel_ctx)
        if step_ctx.metadata.get("policy_pre_deny"):
            return PolicyDecision(
                action=PolicyAction.DENY,
                reason="test_policy_pre_deny",
                policy_rule_id="test.pre_deny",
            )
        org_decision = evaluate_org_policy_pre(
            org=kernel_ctx.organizational,
            channel=step_ctx.metadata.get("channel"),
            requested_tool_ids=extract_requested_tool_ids(outcome.requested_actions),
        )
        if org_decision is not None:
            return org_decision
        if outcome.errors and not is_budget_exceeded_outcome(outcome):
            return PolicyDecision(
                action=PolicyAction.DENY,
                reason="step_errors_present",
                policy_rule_id="kernel.pre_step_errors",
            )
        return kernel_ctx.policy_engine.evaluate_pre_llm(
            tenant_id=kernel_ctx.tenant_id,
            agent_id=kernel_ctx.agent_id,
            message_count=1,
            context=step_ctx.metadata.get("policy_context"),
        )

    @staticmethod
    def _policy_post_check(
        outcome: StepOutcome,
        step_ctx: AgentStepContext,
        kernel_ctx: StepKernelContext,
    ) -> PolicyDecision:
        if kernel_ctx.policy_engine is None:
            return _missing_policy_engine_decision(kernel_ctx)
        if outcome.is_terminal and outcome.output is not None:
            output_text = str(outcome.output)
            return kernel_ctx.policy_engine.evaluate_pre_output(
                tenant_id=kernel_ctx.tenant_id,
                agent_id=kernel_ctx.agent_id,
                output_chars=len(output_text),
            )
        return PolicyDecision(
            action=PolicyAction.ALLOW,
            reason="non_terminal_step",
            policy_rule_id="kernel.post_allow",
        )

    @staticmethod
    def check_hard_budget_before_llm(
        step_ctx: AgentStepContext,
        kernel_ctx: StepKernelContext,
        *,
        pending_agent_tokens: int = 0,
    ):
        return evaluate_hard_budget_violation(
            step_ctx.invocation_usage,
            kernel_ctx.resolved_budget_limits,
            pending_agent_tokens=pending_agent_tokens,
        )

    @staticmethod
    def _budget_exceeded(step_ctx: AgentStepContext, kernel_ctx: StepKernelContext) -> bool:
        if kernel_ctx.max_steps is None:
            return False
        return (step_ctx.step_index + 1) > kernel_ctx.max_steps

    @staticmethod
    async def _enqueue_step_failure_compensations(
        *,
        kernel_ctx: StepKernelContext,
        step_ctx: AgentStepContext,
        action_args: dict[str, dict[str, Any]] | None,
    ) -> tuple[dict[str, Any] | None, int]:
        enqueue_result = await enqueue_compensations_for_step_failure(
            ledger=kernel_ctx.side_effect_ledger,
            tool_profiles=kernel_ctx.tool_profiles,
            step_index=step_ctx.step_index,
            invoker=kernel_ctx.declarative_tool_invoker,
            action_args=action_args,
            compensation_queue=kernel_ctx.compensation_queue,
            run_id=kernel_ctx.run_id,
            tenant_id=kernel_ctx.tenant_id,
            agent_id=kernel_ctx.agent_id,
        )
        if not enqueue_result.actions:
            return None, 0
        for item in enqueue_result.actions:
            kernel_ctx.compensation_requests.append(item.request)
        trace_events = 0
        for item in enqueue_result.actions:
            if item.status == "manual_required":
                trace_events += await HarnessKernel._emit(
                    kernel_ctx,
                    RuntimeEventType.HUMAN_APPROVAL_REQUESTED,
                    {
                        "reason": "manual_compensation_required",
                        "original_side_effect_id": item.request.original_side_effect_id,
                        "tool_id": item.request.compensation_tool_id,
                    },
                )
                continue
            if item.status in ("enqueued", "compensated", "failed", "skipped"):
                trace_events += await HarnessKernel._emit(
                    kernel_ctx,
                    RuntimeEventType.TOOL_REQUESTED,
                    {
                        "mode": "compensation",
                        "tool_name": item.request.compensation_tool_id,
                        "status": item.status,
                        "original_side_effect_id": item.request.original_side_effect_id,
                        "idempotency_key": item.request.idempotency_key,
                    },
                )
                if item.status == "compensated":
                    trace_events += await HarnessKernel._emit(
                        kernel_ctx,
                        RuntimeEventType.TOOL_COMPLETED,
                        {
                            "mode": "compensation",
                            "tool_name": item.request.compensation_tool_id,
                            "status": "success",
                            "original_side_effect_id": item.request.original_side_effect_id,
                        },
                    )
                elif item.status == "failed":
                    trace_events += await HarnessKernel._emit(
                        kernel_ctx,
                        RuntimeEventType.TOOL_FAILED,
                        {
                            "mode": "compensation",
                            "tool_name": item.request.compensation_tool_id,
                            "status": "failed",
                            "error": item.error,
                        },
                    )
        return enqueue_result.diagnostics(), trace_events

    @staticmethod
    async def emit_runtime_event(
        kernel_ctx: StepKernelContext,
        event_type: RuntimeEventType,
        payload: dict[str, Any],
    ) -> int:
        return await HarnessKernel._emit(kernel_ctx, event_type, payload)

    @staticmethod
    async def _emit(
        kernel_ctx: StepKernelContext,
        event_type: RuntimeEventType,
        payload: dict[str, Any],
    ) -> int:
        active_run_id, attempt_id = require_active_execution_identity()
        try:
            resolved_task_id = validate_task_id(kernel_ctx.task_id)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("kernel task_id must be canonical") from exc
        resolved_run_id = validate_run_id(kernel_ctx.run_id)
        if resolved_run_id != active_run_id:
            raise RuntimeError("kernel run_id conflicts with active execution identity")
        event = RuntimeEvent(
            event_type=event_type,
            phase=ExecutionPhase.STEP_EXECUTION,
            payload=payload,
            agent_id=kernel_ctx.agent_id,
            tenant_id=kernel_ctx.tenant_id,
            task_id=resolved_task_id,
            run_id=resolved_run_id,
            attempt_id=attempt_id,
        )
        kernel_ctx.events.append(event)
        if kernel_ctx.emit_event is not None:
            await kernel_ctx.emit_event(event)
        return 1

    @staticmethod
    async def _emit_policy(
        kernel_ctx: StepKernelContext,
        decision: PolicyDecision,
        *,
        phase: str,
    ) -> int:
        return await HarnessKernel._emit(
            kernel_ctx,
            RuntimeEventType.POLICY_DECISION,
            {
                "phase": phase,
                "action": decision.action.value,
                "reason": decision.reason,
                "policy_rule_id": decision.policy_rule_id,
            },
        )

    @staticmethod
    async def _finish_step(
        kernel_ctx: StepKernelContext,
        step_ctx: AgentStepContext,
        record: StepExecutionRecord,
    ) -> StepExecutionRecord:
        await HarnessKernel._append_trace(kernel_ctx, step_ctx, record)
        from intergrax.runtime.attestation.harness_boundary_emitter import HarnessBoundaryEmitter

        HarnessBoundaryEmitter.maybe_emit(
            kernel_ctx=kernel_ctx,
            step_ctx=step_ctx,
            record=record,
        )
        return record

    @staticmethod
    async def _append_trace(
        kernel_ctx: StepKernelContext,
        step_ctx: AgentStepContext,
        record: StepExecutionRecord,
    ) -> None:
        if record.step_record is None:
            return
        updated_root, usage_view = apply_llm_metering_after_step(
            state_root=kernel_ctx.state_root,
            step_metadata=step_ctx.metadata,
            llm_calls=record.step_record.llm_calls,
            limits=kernel_ctx.resolved_budget_limits,
        )
        kernel_ctx.state_root = updated_root
        step_ctx.state_snapshot = updated_root
        step_ctx.invocation_usage = usage_view
        from intergrax.agents.acp_budget_reactions import maybe_emit_budget_threshold

        await maybe_emit_budget_threshold(step_ctx, kernel_ctx)
        step_record = record.step_record
        if kernel_ctx.routing_rule_evaluations:
            diagnostics = dict(step_record.diagnostics)
            diagnostics["llm_routing_evaluations"] = list(kernel_ctx.routing_rule_evaluations)
            step_record = step_record.model_copy(update={"diagnostics": diagnostics})
            kernel_ctx.routing_rule_evaluations.clear()
        kernel_ctx.run_trace.steps.append(step_record)
        kernel_ctx.run_trace.total_steps = len(kernel_ctx.run_trace.steps)
        kernel_ctx.run_trace.total_llm_tokens += sum(
            call.tokens_in + call.tokens_out for call in step_record.llm_calls
        )
        kernel_ctx.run_trace.total_tool_calls += len(step_record.tool_calls)
        kernel_ctx.run_trace.total_rag_calls += len(step_record.rag_calls)
