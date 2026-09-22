# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""HITL branches extracted from NexusLoop (Phase Q-N.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from intergrax.contracts.execution_identity import (
    ActiveExecutionIdentity,
    validate_attempt_id,
    validate_run_id,
)
from intergrax.contracts.runtime_event_metric import RuntimeEventMetricScope
from intergrax.contracts.agent_execution_result import AgentExecutionResult
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.structured_json_value import JsonObject, normalize_structured_json_object
from intergrax.contracts.validation import ValidationResult
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.trace_bridge import runtime_event_from_task_notification
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.hooks.nexus_lifecycle_hooks import (
    NexusLifecycleHookCoordinator,
    NexusLifecycleHookError,
)
from intergrax.runtime.human.hitl_hooks import (
    HumanApprovalHookCoordinator,
    HumanApprovalHookError,
)
from intergrax.runtime.human.escalation import EscalationRouter
from intergrax.contracts.human_approver import human_approval_event_payload
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.long_running.coordinator import LongRunningCoordinator
from intergrax.runtime.long_running.notification import NotificationAdapter
from intergrax.runtime.nexus.orchestration.internal_finish_task_fn import (
    NexusFinishTaskFn,
)
from intergrax.runtime.task.task import Task, TaskResult, TaskState
from intergrax.runtime.task.task_lifecycle import TaskLifecycle
from intergrax.runtime.task.task_trace import (
    PersistingTaskTraceEmitter,
    TaskTraceEmitter,
)


PublishFn = Callable[..., Awaitable[None]]
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
    finish_task: NexusFinishTaskFn
    finalize_trace: FinalizeFn
    maybe_checkpoint: CheckpointFn
    persist_human_decision: PersistHumanFn
    execution_identity: ActiveExecutionIdentity | None = None

    def _require_execution_identity(self) -> tuple[str, str]:
        if self.execution_identity is None:
            raise RuntimeError("active execution identity required for HITL provenance")
        return self.execution_identity.require()

    async def run_lifecycle_hook(
        self,
        *,
        before: bool,
        point: HookPoint,
        task: Task,
        phase: ExecutionPhase,
        trace_emitter: TaskTraceEmitter,
        lifecycle: TaskLifecycle,
        runtime_event_metric_scope: RuntimeEventMetricScope,
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
                runtime_event_metric_scope=runtime_event_metric_scope,
            )
        return None

    async def run_before_human_pause(
        self,
        task: Task,
        trace_emitter: TaskTraceEmitter,
        lifecycle: TaskLifecycle,
        *,
        runtime_event_metric_scope: RuntimeEventMetricScope,
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
                runtime_event_metric_scope=runtime_event_metric_scope,
            )
        return None

    async def handle_human_rejection(
        self,
        task: Task,
        trace_emitter: TaskTraceEmitter,
        lifecycle: TaskLifecycle,
        runtime_event_metric_scope: RuntimeEventMetricScope,
    ) -> TaskResult:
        resolution = task.runtime.governance.hitl_resolution
        run_id, attempt_id = self._require_execution_identity()
        validated_run_id = validate_run_id(run_id)
        validated_attempt_id = validate_attempt_id(attempt_id)
        payload: JsonObject = (
            normalize_structured_json_object(
                human_approval_event_payload(
                    task_id=resolution.task_id,
                    pause_id=resolution.pause_id,
                    human_request_id=resolution.human_request_id,
                    verdict=HumanResponseVerdict.REJECT.value,
                    approver=resolution.approver,
                    response_text=task.options.human.response_text,
                ),
                field_name="human_rejection_payload",
            )
            if resolution is not None
            else {
                "decision": HumanResponseVerdict.REJECT.value,
                "response": task.options.human.response_text,
            }
        )
        await self.publish(
            runtime_event_from_task_notification(
                task,
                run_id=validated_run_id,
                attempt_id=validated_attempt_id,
                message="human rejection received",
                event_type=RuntimeEventType.HUMAN_APPROVAL_RECEIVED,
                phase=ExecutionPhase.HUMAN_APPROVAL,
                payload_raw=payload,
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
            runtime_event_metric_scope=runtime_event_metric_scope,
        )

    async def handle_human_escalation(
        self,
        task: Task,
        trace_emitter: TaskTraceEmitter,
        lifecycle: TaskLifecycle,
        runtime_event_metric_scope: RuntimeEventMetricScope,
    ) -> TaskResult:
        outcome = self.escalation_router.route(task)
        self.escalation_router.apply_to_task(task, outcome)
        self.persist_human_decision(task, HumanResponseVerdict.ESCALATE)

        run_id, attempt_id = self._require_execution_identity()
        validated_run_id = validate_run_id(run_id)
        validated_attempt_id = validate_attempt_id(attempt_id)
        await self.publish(
            runtime_event_from_task_notification(
                task,
                run_id=validated_run_id,
                attempt_id=validated_attempt_id,
                message="human escalation requested",
                event_type=RuntimeEventType.INTERRUPT_ESCALATED,
                phase=ExecutionPhase.HUMAN_APPROVAL,
                payload_raw={
                    "level": outcome.level,
                    "target": outcome.target.value,
                    "message": outcome.message,
                },
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
                runtime_event_metric_scope=runtime_event_metric_scope,
            )

        if task.state == TaskState.CREATED:
            lifecycle.transition(task, TaskState.CLASSIFIED)
            lifecycle.transition(task, TaskState.PLANNED)
        hook_failure = await self.run_before_human_pause(
            task,
            trace_emitter,
            lifecycle,
            runtime_event_metric_scope=runtime_event_metric_scope,
        )
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
            validation=ValidationResult(
                valid=False, errors=["awaiting escalated human review"]
            ),
            plan=None,
            retry_records=[],
            graph_id="",
            runtime_event_metric_scope=runtime_event_metric_scope,
        )
