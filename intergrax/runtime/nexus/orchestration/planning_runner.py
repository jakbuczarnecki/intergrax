# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Classification and planning phase (Phase Q+-N.2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.validation import ValidationResult
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.events.trace_bridge import runtime_event_from_task_state
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.human.pause import HumanPauseCoordinator
from intergrax.runtime.nexus.orchestration.hitl_runner import NexusHitlRunner
from intergrax.runtime.nexus.planning.nexus_planner_protocol import NexusTaskPlannerProtocol
from intergrax.runtime.nexus.planning.task_planner import NexusPlan
from intergrax.runtime.nexus.task_classifier_protocol import NexusTaskClassifierProtocol
from intergrax.runtime.nexus.task_classifier import TaskClassification
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskResult, TaskState
from intergrax.runtime.task.task_lifecycle import TaskLifecycle
from intergrax.runtime.task.task_trace import TaskTraceEmitter

PublishFn = Callable[[RuntimeEvent], Awaitable[None]]
FinishFn = Callable[..., Awaitable[TaskResult]]
CheckpointFn = Callable[..., Awaitable[None]]


@dataclass(slots=True)
class PlanningPhaseOutcome:
    early_result: Optional[TaskResult] = None
    plan: Optional[NexusPlan] = None
    classification: str = ""


@dataclass
class NexusPlanningRunner:
    classifier: NexusTaskClassifierProtocol
    planner: NexusTaskPlannerProtocol
    registry: AgentRegistry
    hitl: NexusHitlRunner
    publish: PublishFn
    finish_task: FinishFn
    maybe_checkpoint: CheckpointFn

    async def run(
        self,
        task: Task,
        *,
        lifecycle: TaskLifecycle,
        trace_emitter: TaskTraceEmitter,
    ) -> PlanningPhaseOutcome:
        hook_failure = await self.hitl.run_lifecycle_hook(
            before=True,
            point=HookPoint.BEFORE_TASK_INTAKE,
            task=task,
            phase=ExecutionPhase.INTAKE,
            trace_emitter=trace_emitter,
            lifecycle=lifecycle,
        )
        if hook_failure is not None:
            return PlanningPhaseOutcome(early_result=hook_failure)

        await self.publish(
            runtime_event_from_task_state(
                task,
                run_id=task.task_id,
                message="task intake",
            ).model_copy(
                update={
                    "event_type": RuntimeEventType.TASK_CREATED,
                    "phase": ExecutionPhase.INTAKE,
                }
            )
        )

        hook_failure = await self.hitl.run_lifecycle_hook(
            before=False,
            point=HookPoint.AFTER_TASK_INTAKE,
            task=task,
            phase=ExecutionPhase.INTAKE,
            trace_emitter=trace_emitter,
            lifecycle=lifecycle,
        )
        if hook_failure is not None:
            return PlanningPhaseOutcome(early_result=hook_failure)

        hook_failure = await self.hitl.run_lifecycle_hook(
            before=True,
            point=HookPoint.BEFORE_CLASSIFICATION,
            task=task,
            phase=ExecutionPhase.CLASSIFICATION,
            trace_emitter=trace_emitter,
            lifecycle=lifecycle,
        )
        if hook_failure is not None:
            return PlanningPhaseOutcome(early_result=hook_failure)

        task = self.classifier.classify(task)
        classification = task.classification or ""
        lifecycle.transition(task, TaskState.CLASSIFIED)

        hook_failure = await self.hitl.run_lifecycle_hook(
            before=False,
            point=HookPoint.AFTER_CLASSIFICATION,
            task=task,
            phase=ExecutionPhase.CLASSIFICATION,
            trace_emitter=trace_emitter,
            lifecycle=lifecycle,
            extra={"classification": classification},
        )
        if hook_failure is not None:
            return PlanningPhaseOutcome(early_result=hook_failure)

        if classification == TaskClassification.UNSUPPORTED.value:
            lifecycle.transition(task, TaskState.FAILED)
            return PlanningPhaseOutcome(
                early_result=await self.finish_task(
                    task,
                    trace_emitter,
                    answer="",
                    executions=[],
                    validation=ValidationResult(
                        valid=False,
                        errors=[
                            task.runtime.classification.unsupported_reason
                            or "unsupported task"
                        ],
                    ),
                    plan=None,
                    retry_records=[],
                    graph_id="",
                ),
                classification=classification,
            )

        hook_failure = await self.hitl.run_lifecycle_hook(
            before=True,
            point=HookPoint.BEFORE_PLANNING,
            task=task,
            phase=ExecutionPhase.PLANNING,
            trace_emitter=trace_emitter,
            lifecycle=lifecycle,
            extra={"classification": classification},
        )
        if hook_failure is not None:
            return PlanningPhaseOutcome(early_result=hook_failure)

        plan = self.planner.plan(task, self.registry)
        task.runtime.orchestration.plan_id = plan.plan_id
        task.sync_metadata()
        lifecycle.transition(task, TaskState.PLANNED)
        await self.publish(
            runtime_event_from_task_state(
                task, run_id=task.task_id, message="plan created"
            ).model_copy(
                update={
                    "event_type": RuntimeEventType.PLAN_CREATED,
                    "phase": ExecutionPhase.PLANNING,
                    "payload": {
                        "plan_id": plan.plan_id,
                        "step_count": len(plan.steps),
                        "task_state": task.state.value,
                    },
                }
            )
        )

        hook_failure = await self.hitl.run_lifecycle_hook(
            before=False,
            point=HookPoint.AFTER_PLANNING,
            task=task,
            phase=ExecutionPhase.PLANNING,
            trace_emitter=trace_emitter,
            lifecycle=lifecycle,
            extra={"plan_id": plan.plan_id, "step_count": len(plan.steps)},
        )
        if hook_failure is not None:
            return PlanningPhaseOutcome(early_result=hook_failure)

        if classification == TaskClassification.HUMAN_APPROVAL_REQUIRED.value:
            if not HumanPauseCoordinator.is_resumed(task):
                hook_failure = await self.hitl.run_before_human_pause(
                    task, trace_emitter, lifecycle
                )
                if hook_failure is not None:
                    return PlanningPhaseOutcome(early_result=hook_failure)
                lifecycle.transition(task, TaskState.WAITING_FOR_HUMAN)
                await self.maybe_checkpoint(
                    task,
                    progress_message="awaiting human approval",
                    plan=plan,
                )
                return PlanningPhaseOutcome(
                    early_result=await self.finish_task(
                        task,
                        trace_emitter,
                        answer="",
                        executions=[],
                        validation=ValidationResult(
                            valid=False,
                            errors=["awaiting human approval"],
                        ),
                        plan=plan,
                        retry_records=[],
                        graph_id="",
                    ),
                    plan=plan,
                    classification=classification,
                )
            lifecycle.transition(task, TaskState.RUNNING)
        else:
            lifecycle.transition(task, TaskState.RUNNING)

        return PlanningPhaseOutcome(plan=plan, classification=classification)
