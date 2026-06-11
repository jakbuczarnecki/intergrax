# © Artur Czarnecki. All rights reserved.

"""HarnessKernel L1 step cycle (architecture §38 · ACP-STEP-2b)."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from intergrax.agents.authoring.side_effect_validation import validate_side_effect_mode
from intergrax.agents.authoring.state_merge import extract_acp_state_blob, merge_session_state
from intergrax.agents.authoring.step_outcome import StepOutcome
from intergrax.contracts.agent_run import AgentRunError
from intergrax.contracts.agent_run_enums import (
    AgentRunErrorCode,
    SideEffectMode,
    StepNextAction,
    TerminalReason,
)
from intergrax.contracts.agent_run import AgentRunTrace
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.contracts.step_execution import AgentStepRecord, StepExecutionRecord
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.policy.policy_engine import PolicyEngine

EventEmitter = Callable[[RuntimeEvent], Awaitable[None]]
CheckpointHook = Callable[[dict[str, Any], int], Awaitable[None]]


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
    emit_event: EventEmitter | None = None
    checkpoint_hook: CheckpointHook | None = None
    state_root: dict[str, Any] = field(default_factory=dict)
    run_trace: AgentRunTrace = field(default_factory=AgentRunTrace)
    events: list[RuntimeEvent] = field(default_factory=list)


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

        mode_error = validate_side_effect_mode(outcome, kernel_ctx.side_effect_mode)
        if mode_error is not None:
            record = StepExecutionRecord(
                step_index=step_ctx.step_index,
                outcome_applied=False,
                side_effect_mode_violation=True,
                error_code=AgentRunErrorCode.VALIDATION_FAILED,
                step_record=AgentStepRecord(
                    step_index=step_ctx.step_index,
                    next_action=outcome.next_action.value,
                    is_terminal=True,
                    terminal_reason=TerminalReason.VALIDATION_FAILED.value,
                    error_code=AgentRunErrorCode.VALIDATION_FAILED,
                    diagnostics={"side_effect_mode": mode_error},
                ),
            )
            await HarnessKernel._append_trace(kernel_ctx, record)
            return record

        policy_pre = HarnessKernel._policy_pre_check(outcome, step_ctx, kernel_ctx)
        trace_events += await HarnessKernel._emit_policy(kernel_ctx, policy_pre, phase="pre")
        if policy_pre.action == PolicyAction.DENY:
            record = StepExecutionRecord(
                step_index=step_ctx.step_index,
                outcome_applied=False,
                policy_pre=policy_pre,
                error_code=AgentRunErrorCode.POLICY_DENIED,
                trace_event_count=trace_events,
                step_record=AgentStepRecord(
                    step_index=step_ctx.step_index,
                    next_action=StepNextAction.FAIL.value,
                    is_terminal=True,
                    terminal_reason=TerminalReason.POLICY_DENIED.value,
                    policy_pre=policy_pre,
                    error_code=AgentRunErrorCode.POLICY_DENIED,
                ),
            )
            await HarnessKernel._append_trace(kernel_ctx, record)
            return record

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
                step_record=AgentStepRecord(
                    step_index=step_ctx.step_index,
                    next_action=StepNextAction.FAIL.value,
                    is_terminal=True,
                    terminal_reason=TerminalReason.VALIDATION_FAILED.value,
                    error_code=merge_result.error_code,
                    diagnostics={"merge_error": merge_result.error_message},
                ),
            )
            await HarnessKernel._append_trace(kernel_ctx, record)
            return record

        kernel_ctx.state_root = merge_result.state
        step_ctx.state_snapshot = merge_result.state
        state_version = int(extract_acp_state_blob(merge_result.state).get("_version", 0))

        if (
            kernel_ctx.side_effect_mode == SideEffectMode.DECLARATIVE
            and outcome.requested_actions
        ):
            trace_events += await HarnessKernel._emit(
                kernel_ctx,
                RuntimeEventType.TOOL_REQUESTED,
                {"mode": "declarative", "action_count": len(outcome.requested_actions)},
            )

        policy_post = HarnessKernel._policy_post_check(outcome, step_ctx, kernel_ctx)
        trace_events += await HarnessKernel._emit_policy(kernel_ctx, policy_post, phase="post")
        if policy_post.action == PolicyAction.DENY:
            record = StepExecutionRecord(
                step_index=step_ctx.step_index,
                outcome_applied=False,
                policy_pre=policy_pre,
                policy_post=policy_post,
                state_version=state_version,
                error_code=AgentRunErrorCode.POLICY_DENIED,
                trace_event_count=trace_events,
                step_record=AgentStepRecord(
                    step_index=step_ctx.step_index,
                    next_action=StepNextAction.FAIL.value,
                    is_terminal=True,
                    terminal_reason=TerminalReason.POLICY_DENIED.value,
                    policy_pre=policy_pre,
                    policy_post=policy_post,
                    state_version=state_version,
                    error_code=AgentRunErrorCode.POLICY_DENIED,
                ),
            )
            await HarnessKernel._append_trace(kernel_ctx, record)
            return record

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
                step_record=AgentStepRecord(
                    step_index=step_ctx.step_index,
                    next_action=StepNextAction.FAIL.value,
                    is_terminal=True,
                    terminal_reason=TerminalReason.MAX_STEPS_EXCEEDED.value,
                    policy_pre=policy_pre,
                    policy_post=policy_post,
                    state_version=state_version,
                    error_code=AgentRunErrorCode.MAX_STEPS_EXCEEDED,
                ),
            )
            await HarnessKernel._append_trace(kernel_ctx, record)
            return record

        trace_events += await HarnessKernel._emit(
            kernel_ctx,
            RuntimeEventType.STEP_COMPLETED,
            {
                "step_index": step_ctx.step_index,
                "next_action": outcome.next_action.value,
                "is_terminal": outcome.is_terminal,
            },
        )

        if kernel_ctx.checkpoint_every_step and kernel_ctx.checkpoint_hook is not None:
            await kernel_ctx.checkpoint_hook(kernel_ctx.state_root, step_ctx.step_index)

        duration_ms = int((time.perf_counter() - started) * 1000)
        _ = duration_ms

        record = StepExecutionRecord(
            step_index=step_ctx.step_index,
            outcome_applied=True,
            policy_pre=policy_pre,
            policy_post=policy_post,
            state_version=state_version,
            trace_event_count=trace_events,
            step_record=AgentStepRecord(
                step_index=step_ctx.step_index,
                next_action=outcome.next_action.value,
                is_terminal=outcome.is_terminal,
                terminal_reason=(
                    outcome.terminal_reason.value if outcome.terminal_reason else None
                ),
                policy_pre=policy_pre,
                policy_post=policy_post,
                state_version=state_version,
                diagnostics=outcome.diagnostics,
            ),
        )
        await HarnessKernel._append_trace(kernel_ctx, record)
        return record

    @staticmethod
    def _policy_pre_check(
        outcome: StepOutcome,
        step_ctx: AgentStepContext,
        kernel_ctx: StepKernelContext,
    ) -> PolicyDecision:
        if kernel_ctx.policy_engine is None:
            return PolicyDecision(
                action=PolicyAction.ALLOW,
                reason="no_policy_engine",
                policy_rule_id="kernel.default_allow",
            )
        if step_ctx.metadata.get("policy_pre_deny"):
            return PolicyDecision(
                action=PolicyAction.DENY,
                reason="test_policy_pre_deny",
                policy_rule_id="test.pre_deny",
            )
        if outcome.errors:
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
            return PolicyDecision(
                action=PolicyAction.ALLOW,
                reason="no_policy_engine",
                policy_rule_id="kernel.default_allow",
            )
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
    def _budget_exceeded(step_ctx: AgentStepContext, kernel_ctx: StepKernelContext) -> bool:
        if kernel_ctx.max_steps is None:
            return False
        return (step_ctx.step_index + 1) > kernel_ctx.max_steps

    @staticmethod
    async def _emit(
        kernel_ctx: StepKernelContext,
        event_type: RuntimeEventType,
        payload: dict[str, Any],
    ) -> int:
        event = RuntimeEvent(
            event_type=event_type,
            phase=ExecutionPhase.STEP_EXECUTION,
            payload=payload,
            agent_id=kernel_ctx.agent_id,
            tenant_id=kernel_ctx.tenant_id,
            task_id=kernel_ctx.task_id or kernel_ctx.run_id or "task",
            run_id=kernel_ctx.run_id or "run",
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
    async def _append_trace(
        kernel_ctx: StepKernelContext,
        record: StepExecutionRecord,
    ) -> None:
        if record.step_record is not None:
            kernel_ctx.run_trace.steps.append(record.step_record.model_dump(mode="json"))
