# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Task intake and HITL preamble (Phase Q+-N.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.events.trace_bridge import runtime_event_from_task_state
from intergrax.runtime.human.hitl_hooks import HumanApprovalHookCoordinator
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.human.pause import HumanPauseCoordinator
from intergrax.runtime.long_running.coordinator import LongRunningCoordinator
from intergrax.runtime.nexus.orchestration.hitl_runner import NexusHitlRunner
from intergrax.runtime.nexus.orchestration.human_response import normalize_human_response
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
        if verdict == HumanResponseVerdict.REJECT:
            return IntakePhaseOutcome(
                early_result=await self.hitl.handle_human_rejection(
                    task, trace_emitter, lifecycle
                )
            )
        if verdict == HumanResponseVerdict.ESCALATE:
            return IntakePhaseOutcome(
                early_result=await self.hitl.handle_human_escalation(
                    task, trace_emitter, lifecycle
                )
            )

        if HumanPauseCoordinator.is_resumed(task):
            await self.publish(
                runtime_event_from_task_state(
                    task,
                    run_id=task.task_id,
                    message="human approval received",
                ).model_copy(
                    update={
                        "event_type": RuntimeEventType.HUMAN_APPROVAL_RECEIVED,
                        "phase": ExecutionPhase.HUMAN_APPROVAL,
                        "payload": {
                            "response": task.options.human.response_text or "approve",
                        },
                    }
                )
            )
            await self.human_hooks.after_response(
                task,
                verdict=HumanResponseVerdict.APPROVE.value,
            )
            HumanPauseCoordinator.clear_pause(task)

        return IntakePhaseOutcome()
