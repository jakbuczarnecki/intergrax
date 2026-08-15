# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Classification and planning phase (Phase Q+-N.2)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from intergrax.contracts.decision_record import DecisionRecord
from intergrax.contracts.execution_identity import ActiveExecutionIdentity
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.reasoning_failure import ReasoningFailureKind
from intergrax.contracts.validation import ValidationResult
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.events.trace_bridge import runtime_event_from_task_state
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.human.pause import HumanPauseCoordinator
from intergrax.runtime.nexus.orchestration.hitl_runner import NexusHitlRunner
from intergrax.runtime.nexus.orchestration.planning_coordination_advisory import (
    build_coordination_advisory_event,
)
from intergrax.runtime.nexus.planning.nexus_planner_protocol import NexusTaskPlannerProtocol
from intergrax.runtime.nexus.observability.planning_metrics import (
    record_planning_failure,
    record_planner_fallback,
    record_planner_latency,
)
from intergrax.runtime.nexus.planning.plan_validator import validate_nexus_plan
from intergrax.runtime.nexus.planning.task_planner import NexusPlan
from intergrax.runtime.nexus.task_classifier_protocol import NexusTaskClassifierProtocol
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.runtime.nexus.task_classifier import TaskClassification
from intergrax.runtime.policy.policy_engine import PolicyEngine
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
    policy_engine: PolicyEngine | None = None
    emit_coordination_advisory: bool = False
    denied_planner_model_ids: tuple[str, ...] = ()
    planner_model_id: str | None = None
    execution_identity: ActiveExecutionIdentity | None = None

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

        if self.execution_identity is None:
            raise RuntimeError("active execution identity required for planning emission")
        run_id, attempt_id = self.execution_identity.require()
        await self.publish(
            runtime_event_from_task_state(
                task,
                run_id=run_id,
                attempt_id=attempt_id,
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
        failure_kind = task.metadata.get("reasoning_failure_kind")
        if failure_kind:
            record_planning_failure(kind=str(failure_kind))
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
            unsupported_kind = ReasoningFailureKind.CLASSIFIER_UNSUPPORTED.value
            task.metadata["reasoning_failure_kind"] = unsupported_kind
            record_planning_failure(kind=unsupported_kind)
            task.sync_metadata()
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

        planning_policy_action = PolicyAction.ALLOW.value
        if self.policy_engine is not None:
            policy_decision = self.policy_engine.evaluate_pre_llm(
                tenant_id=task.tenant_id,
                agent_id=task.agent_id or "",
                message_count=1,
                context={
                    "phase": "nexus_planning",
                    "classification": classification,
                    "planner_model_id": self.planner_model_id or "",
                    "denied_planner_model_ids": self.denied_planner_model_ids,
                },
            )
            planning_policy_action = policy_decision.action.value
            if policy_decision.action is PolicyAction.DENY:
                failure_kind = ReasoningFailureKind.PLANNER_POLICY_BLOCKED
                task.metadata["reasoning_failure_kind"] = failure_kind.value
                record_planning_failure(kind=failure_kind.value)
                task.sync_metadata()
                lifecycle.transition(task, TaskState.FAILED)
                return PlanningPhaseOutcome(
                    early_result=await self.finish_task(
                        task,
                        trace_emitter,
                        answer="",
                        executions=[],
                        validation=ValidationResult(
                            valid=False,
                            errors=[policy_decision.reason or "planning_blocked_by_policy"],
                        ),
                        plan=None,
                        retry_records=[],
                        graph_id="",
                    ),
                    classification=classification,
                )

        plan_started = time.perf_counter()
        plan = self.planner.plan(task, self.registry)
        record_planner_latency(latency_ms=(time.perf_counter() - plan_started) * 1000.0)
        plan_errors = validate_nexus_plan(plan, self.registry)
        if plan_errors:
            lifecycle.transition(task, TaskState.FAILED)
            return PlanningPhaseOutcome(
                early_result=await self.finish_task(
                    task,
                    trace_emitter,
                    answer="",
                    executions=[],
                    validation=ValidationResult(valid=False, errors=plan_errors),
                    plan=plan,
                    retry_records=[],
                    graph_id="",
                ),
                classification=classification,
            )

        if plan.plan_metadata.get("used_fallback") == "true":
            record_planner_fallback()
            failure_kind = plan.plan_metadata.get("failure_kind")
            if failure_kind:
                record_planning_failure(kind=failure_kind)

        task.runtime.orchestration.plan_id = plan.plan_id
        for key, value in plan.plan_metadata.items():
            task.metadata[key] = value
        task.sync_metadata()
        lifecycle.transition(task, TaskState.PLANNED)
        planner_source = str(plan.plan_metadata.get("planner_source", "default"))
        decision = DecisionRecord(
            trace_id=task.task_id,
            run_id=task.task_id,
            tenant_id=task.tenant_id,
            task_id=task.task_id,
            decision_type="nexus_planning",
            rationale=(
                f"classification={classification}; planner_source={planner_source}; "
                f"planner_model_id={self.planner_model_id or ''}"
            ),
            policy_action=planning_policy_action,
            metadata={
                "classification": classification,
                "plan_id": plan.plan_id,
                "step_count": str(len(plan.steps)),
                "planner_source": planner_source,
                "planner_model_id": self.planner_model_id or "",
                "used_fallback": plan.plan_metadata.get("used_fallback", "false"),
                "failure_kind": plan.plan_metadata.get("failure_kind", ""),
            },
        )
        plan_payload: dict[str, object] = {
            "plan_id": plan.plan_id,
            "step_count": len(plan.steps),
            "task_state": task.state.value,
            "decision_record": decision.model_dump(mode="json"),
        }
        plan_payload.update(plan.plan_metadata)
        run_id, attempt_id = self.execution_identity.require()
        await self.publish(
            runtime_event_from_task_state(
                task,
                run_id=run_id,
                attempt_id=attempt_id,
                message="plan created",
            ).model_copy(
                update={
                    "event_type": RuntimeEventType.PLAN_CREATED,
                    "phase": ExecutionPhase.PLANNING,
                    "payload": plan_payload,
                }
            )
        )
        if self.emit_coordination_advisory:
            await self.publish(
                build_coordination_advisory_event(
                    task_id=task.task_id,
                    tenant_id=task.tenant_id,
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
