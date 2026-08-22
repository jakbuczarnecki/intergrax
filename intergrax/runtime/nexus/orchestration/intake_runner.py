# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Task intake and HITL preamble (Phase Q+-N.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from intergrax.contracts.execution_identity import ActiveExecutionIdentity
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.events.trace_bridge import runtime_event_from_task_state
from intergrax.runtime.human.hitl_hooks import HumanApprovalHookCoordinator
from intergrax.runtime.human.declarative_hitl_grant import DeclarativeHitlGrantCoordinator
from intergrax.runtime.human.governed_continuation_grant import GovernedContinuationGrantCoordinator
from intergrax.contracts.human_approver import human_approval_event_payload
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.human.pause import HumanPauseCoordinator
from intergrax.runtime.long_running.coordinator import LongRunningCoordinator
from intergrax.runtime.nexus.orchestration.hitl_runner import NexusHitlRunner
from intergrax.runtime.nexus.orchestration.human_response import (
    clear_consumed_human_input,
    normalize_human_response,
)
from intergrax.runtime.task.task import Task, TaskResult, TaskState
from intergrax.runtime.task.task_lifecycle import TaskLifecycle
from intergrax.runtime.task.task_trace import TaskTraceEmitter

PublishFn = Callable[[RuntimeEvent], Awaitable[None]]
RestoreFn = Callable[[Task], Awaitable[None]]


@dataclass(slots=True)
class IntakePhaseOutcome:
    early_result: Optional[TaskResult] = None


@dataclass
class NexusIntakeRunner:
    hitl: NexusHitlRunner
    human_hooks: HumanApprovalHookCoordinator
    publish: PublishFn
    restore_long_running: RestoreFn
    execution_identity: ActiveExecutionIdentity | None = None

    async def run(
        self,
        task: Task,
        *,
        lifecycle: TaskLifecycle,
        trace_emitter: TaskTraceEmitter,
    ) -> IntakePhaseOutcome:
        normalize_human_response(task)
        await self.restore_long_running(task)

        if (
            LongRunningCoordinator.is_long_running(task)
            and HumanPauseCoordinator.is_resumed(task)
            and task.state in LongRunningCoordinator.paused_states()
        ):
            task.state = TaskState.CREATED
        elif (
            LongRunningCoordinator.is_long_running(task)
            and task.options.long_running.resume_token
            and task.state == TaskState.FAILED
        ):
            task.state = TaskState.CREATED

        verdict = HumanPauseCoordinator.verdict_from_task(task)
        response_pause_id = task.options.human.pause_id
        response_request_id = task.options.human.human_request_id
        approver = task.options.human.approver
        if verdict in {
            HumanResponseVerdict.REJECT,
            HumanResponseVerdict.ESCALATE,
            HumanResponseVerdict.APPROVE,
        }:
            if approver is None:
                raise RuntimeError("approver evidence required for human approval resolution")
        if verdict == HumanResponseVerdict.REJECT:
            HumanPauseCoordinator.resolve_human_response(
                task,
                HumanResponseVerdict.REJECT,
                approver=approver,  # type: ignore[arg-type]
                pause_id=response_pause_id,
                human_request_id=response_request_id,
                response_text=task.options.human.response_text,
            )
            DeclarativeHitlGrantCoordinator.clear_pending_and_grant(task)
            GovernedContinuationGrantCoordinator.clear_grant(task)
            result = await self.hitl.handle_human_rejection(
                task, trace_emitter, lifecycle
            )
            clear_consumed_human_input(task)
            return IntakePhaseOutcome(early_result=result)
        if verdict == HumanResponseVerdict.ESCALATE:
            HumanPauseCoordinator.resolve_human_response(
                task,
                HumanResponseVerdict.ESCALATE,
                approver=approver,  # type: ignore[arg-type]
                pause_id=response_pause_id,
                human_request_id=response_request_id,
                response_text=task.options.human.response_text,
            )
            DeclarativeHitlGrantCoordinator.clear_pending_and_grant(task)
            GovernedContinuationGrantCoordinator.clear_grant(task)
            result = await self.hitl.handle_human_escalation(
                task, trace_emitter, lifecycle
            )
            clear_consumed_human_input(task)
            return IntakePhaseOutcome(early_result=result)

        if HumanPauseCoordinator.is_resumed(task):
            if self.execution_identity is None:
                raise RuntimeError("active execution identity required for intake emission")
            run_id, attempt_id = self.execution_identity.require()
            HumanPauseCoordinator.resolve_human_response(
                task,
                HumanResponseVerdict.APPROVE,
                approver=approver,  # type: ignore[arg-type]
                pause_id=response_pause_id,
                human_request_id=response_request_id,
                run_id=run_id,
                response_text=task.options.human.response_text,
            )
            resolution = task.runtime.governance.hitl_resolution
            assert resolution is not None
            self.hitl.persist_human_decision(
                task,
                HumanResponseVerdict.APPROVE,
                response_text=task.options.human.response_text or "",
            )
            await self.publish(
                runtime_event_from_task_state(
                    task,
                    run_id=run_id,
                    attempt_id=attempt_id,
                    message="human approval received",
                ).model_copy(
                    update={
                        "event_type": RuntimeEventType.HUMAN_APPROVAL_RECEIVED,
                        "phase": ExecutionPhase.HUMAN_APPROVAL,
                        "payload": human_approval_event_payload(
                            task_id=task.task_id,
                            pause_id=resolution.pause_id,
                            human_request_id=resolution.human_request_id,
                            verdict=HumanResponseVerdict.APPROVE,
                            approver=resolution.approver,
                            response_text=task.options.human.response_text,
                        ),
                    }
                )
            )
            await self.human_hooks.after_response(
                task,
                verdict=HumanResponseVerdict.APPROVE.value,
            )
            if task.runtime.governance.declarative_hitl_pending is not None:
                DeclarativeHitlGrantCoordinator.create_grant_from_pending(task)
                task.sync_metadata()
            if task.runtime.governance.human_request is not None:
                GovernedContinuationGrantCoordinator.create_grant_from_approval(task)
                task.sync_metadata()
            HumanPauseCoordinator.clear_pause(task)
            clear_consumed_human_input(task)

        return IntakePhaseOutcome()
