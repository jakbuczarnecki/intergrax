# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""HITL branches extracted from NexusLoop (Phase Q-N.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from intergrax.contracts.agent_execution_result import AgentExecutionResult
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.validation import ValidationResult
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.trace_bridge import runtime_event_from_task_state
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.hooks.nexus_lifecycle_hooks import NexusLifecycleHookCoordinator, NexusLifecycleHookError
from intergrax.runtime.human.hitl_hooks import HumanApprovalHookCoordinator, HumanApprovalHookError
from intergrax.runtime.human.escalation import EscalationRouter
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.human.pause import HumanPauseCoordinator
from intergrax.runtime.long_running.coordinator import LongRunningCoordinator
from intergrax.runtime.long_running.notification import NotificationAdapter
from intergrax.runtime.nexus.planning.task_planner import NexusPlan
from intergrax.runtime.task.task import Task, TaskResult, TaskState
from intergrax.runtime.task.task_lifecycle import TaskLifecycle
from intergrax.runtime.task.task_trace import PersistingTaskTraceEmitter, TaskTraceEmitter


PublishFn = Callable[..., Awaitable[None]]
FinishFn = Callable[..., Awaitable[TaskResult]]
FinalizeFn = Callable[..., Awaitable[None]]
CheckpointFn = Callable[..., Awaitable[None]]
PersistHumanFn = Callable[..., None]


@dataclass
class NexusHitlRunner:
    publish: PublishFn
    human_hooks: HumanApprovalHookCoordinator
    lifecycle_hooks: NexusLifecycleHookCoordinator
    escalation_router: EscalationRouter
    notification_adapter: Optional[NotificationAdapter]
    finish_task: FinishFn
    finalize_trace: FinalizeFn
    maybe_checkpoint: CheckpointFn
    persist_human_decision: PersistHumanFn

    async def run_lifecycle_hook(
        self,
        *,
        before: bool,
        point: HookPoint,
        task: Task,
        phase: ExecutionPhase,
        trace_emitter: TaskTraceEmitter,
        lifecycle: TaskLifecycle,
        extra: Optional[dict] = None,
    ) -> Optional[TaskResult]:
        try:
            if before:
                await self.lifecycle_hooks.before(point, task, phase=phase, extra=extra)
            else:
                await self.lifecycle_hooks.after(point, task, phase=phase, extra=extra)
        except NexusLifecycleHookError as exc:
            lifecycle.transition(task, TaskState.FAILED)
            if isinstance(trace_emitter, PersistingTaskTraceEmitter):
                await self.finalize_trace(trace_emitter, [], task_id=task.task_id)
            return await self.finish_task(
                task,
                trace_emitter,
                answer="",
                executions=[],
                validation=ValidationResult(valid=False, errors=[str(exc)]),
                plan=None,
                retry_records=[],
                graph_id="",
            )
        return None

    async def run_before_human_pause(
        self,
        task: Task,
        trace_emitter: TaskTraceEmitter,
        lifecycle: TaskLifecycle,
        *,
        agent_id: Optional[str] = None,
        execution: Optional[AgentExecutionResult] = None,
    ) -> Optional[TaskResult]:
        try:
            await self.human_hooks.before_pause(
                task,
                agent_id=agent_id,
                execution=execution,
            )
        except HumanApprovalHookError as exc:
            lifecycle.transition(task, TaskState.FAILED)
            if isinstance(trace_emitter, PersistingTaskTraceEmitter):
                await self.finalize_trace(trace_emitter, [], task_id=task.task_id)
            return await self.finish_task(
                task,
                trace_emitter,
                answer="",
                executions=[],
                validation=ValidationResult(valid=False, errors=[str(exc)]),
                plan=None,
                retry_records=[],
                graph_id="",
            )
        return None

    async def handle_human_rejection(
        self,
        task: Task,
        trace_emitter: TaskTraceEmitter,
        lifecycle: TaskLifecycle,
    ) -> TaskResult:
        await self.publish(
            runtime_event_from_task_state(
                task,
                run_id=task.task_id,
                message="human rejection received",
            ).model_copy(
                update={
                    "event_type": RuntimeEventType.HUMAN_APPROVAL_RECEIVED,
                    "phase": ExecutionPhase.HUMAN_APPROVAL,
                    "payload": {
                        "decision": HumanResponseVerdict.REJECT.value,
                        "response": task.options.human.response_text,
                    },
                }
            ),
            task=task,
        )
        await self.human_hooks.after_response(
            task,
            verdict=HumanResponseVerdict.REJECT.value,
        )
        self.persist_human_decision(task, HumanResponseVerdict.REJECT)
        lifecycle.transition(task, TaskState.FAILED)
        if isinstance(trace_emitter, PersistingTaskTraceEmitter):
            await self.finalize_trace(trace_emitter, [], task_id=task.task_id)
        return await self.finish_task(
            task,
            trace_emitter,
            answer="",
            executions=[],
            validation=ValidationResult(valid=False, errors=["human rejected"]),
            plan=None,
            retry_records=[],
            graph_id="",
        )

    async def handle_human_escalation(
        self,
        task: Task,
        trace_emitter: TaskTraceEmitter,
        lifecycle: TaskLifecycle,
    ) -> TaskResult:
        outcome = self.escalation_router.route(task)
        self.escalation_router.apply_to_task(task, outcome)
        self.persist_human_decision(task, HumanResponseVerdict.ESCALATE)

        await self.publish(
            runtime_event_from_task_state(
                task,
                run_id=task.task_id,
                message="human escalation requested",
            ).model_copy(
                update={
                    "event_type": RuntimeEventType.INTERRUPT_ESCALATED,
                    "phase": ExecutionPhase.HUMAN_APPROVAL,
                    "payload": {
                        "level": outcome.level,
                        "target": outcome.target.value,
                        "message": outcome.message,
                    },
                }
            ),
            task=task,
        )
        await self.human_hooks.after_response(
            task,
            verdict=HumanResponseVerdict.ESCALATE.value,
        )

        progress_message = "awaiting escalated human review"
        await LongRunningCoordinator.notify_escalation(
            task,
            outcome=outcome,
            progress_message=progress_message,
            adapter=self.notification_adapter,
        )

        task.options.human.response_text = None
        task.options.human.verdict = None
        task.sync_metadata()

        if outcome.fail_task:
            lifecycle.transition(task, TaskState.FAILED)
            if isinstance(trace_emitter, PersistingTaskTraceEmitter):
                await self.finalize_trace(trace_emitter, [], task_id=task.task_id)
            return await self.finish_task(
                task,
                trace_emitter,
                answer="",
                executions=[],
                validation=ValidationResult(
                    valid=False,
                    errors=[outcome.message or "escalation limit reached"],
                ),
                plan=None,
                retry_records=[],
                graph_id="",
            )

        if task.state == TaskState.CREATED:
            lifecycle.transition(task, TaskState.CLASSIFIED)
            lifecycle.transition(task, TaskState.PLANNED)
        hook_failure = await self.run_before_human_pause(task, trace_emitter, lifecycle)
        if hook_failure is not None:
            if isinstance(trace_emitter, PersistingTaskTraceEmitter):
                await self.finalize_trace(trace_emitter, [], task_id=task.task_id)
            return hook_failure
        lifecycle.transition(task, TaskState.WAITING_FOR_HUMAN)
        await self.maybe_checkpoint(
            task,
            progress_message=progress_message,
        )
        if isinstance(trace_emitter, PersistingTaskTraceEmitter):
            await self.finalize_trace(trace_emitter, [], task_id=task.task_id)
        return await self.finish_task(
            task,
            trace_emitter,
            answer="",
            executions=[],
            validation=ValidationResult(valid=False, errors=["awaiting escalated human review"]),
            plan=None,
            retry_records=[],
            graph_id="",
        )
