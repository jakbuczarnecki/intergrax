# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import List, Optional

from intergrax.agents.agent_engine import AgentEngine
from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.runtime.nexus.execution.execution_graph import ExecutionNodeStatus
from intergrax.contracts.validation import ValidationResult
from intergrax.runtime.nexus.agent_router import AgentRouter
from intergrax.runtime.nexus.context.context_manager import ContextManager
from intergrax.runtime.nexus.execution.graph_builder import plan_to_execution_graph
from intergrax.runtime.nexus.execution.graph_executor import GraphExecutor
from intergrax.runtime.nexus.planning.task_planner import NexusPlan, TaskPlanner
from intergrax.runtime.nexus.response.final_response_composer import FinalResponseComposer
from intergrax.runtime.nexus.retry.retry_engine import RetryEngine, RetryPolicy, RetryRecord
from intergrax.runtime.nexus.task_classifier import ClassifyingTaskClassifier, TaskClassification
from intergrax.runtime.nexus.tracing.persistence_models import RunTraceWriter
from intergrax.runtime.nexus.validation.validation_engine import NexusValidationEngine
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.trace_bridge import runtime_event_from_task_state
from intergrax.runtime.task.task import Task, TaskResult, TaskState
from intergrax.runtime.task.task_lifecycle import TaskLifecycle
from intergrax.runtime.task.task_trace import (
    PersistingTaskTraceEmitter,
    TaskTraceEmitter,
    lifecycle_with_persisting_trace,
    lifecycle_with_trace,
)


class NexusLoop:
    """
    Global Nexus loop (§9.1, §41).

    Task → classify → plan → execution graph → validate → compose response.
    """

    def __init__(
        self,
        registry: AgentRegistry,
        *,
        classifier: Optional[ClassifyingTaskClassifier] = None,
        planner: Optional[TaskPlanner] = None,
        validation_engine: Optional[NexusValidationEngine] = None,
        retry_engine: Optional[RetryEngine] = None,
        graph_executor: Optional[GraphExecutor] = None,
        context_manager: Optional[ContextManager] = None,
        lifecycle: Optional[TaskLifecycle] = None,
        trace_emitter: Optional[TaskTraceEmitter] = None,
        trace_store: Optional[RunTraceWriter] = None,
        event_bus: Optional[RuntimeEventBus] = None,
        retry_policy: Optional[RetryPolicy] = None,
    ) -> None:
        self._registry = registry
        self._engine = AgentEngine(registry)
        self._classifier = classifier or ClassifyingTaskClassifier(registry)
        self._planner = planner or TaskPlanner()
        self._validation_engine = validation_engine or NexusValidationEngine()
        self._retry_engine = retry_engine or RetryEngine(
            registry,
            policy=retry_policy or RetryPolicy(),
        )
        self._router = AgentRouter(registry)
        self._context_manager = context_manager or ContextManager()
        self._graph_executor = graph_executor or GraphExecutor(
            registry,
            engine=self._engine,
            router=self._router,
            validation_engine=self._validation_engine,
            retry_engine=self._retry_engine,
            context_manager=self._context_manager,
        )
        self._composer = FinalResponseComposer()
        self._lifecycle = lifecycle
        self._trace_emitter = trace_emitter
        self._trace_store = trace_store
        self._event_bus = event_bus or RuntimeEventBus()

    @property
    def registry(self) -> AgentRegistry:
        return self._registry

    @property
    def trace_emitter(self) -> Optional[TaskTraceEmitter]:
        return self._trace_emitter

    @property
    def event_bus(self) -> RuntimeEventBus:
        return self._event_bus

    async def handle_task(self, task: Task) -> TaskResult:
        lifecycle, trace_emitter = self._resolve_lifecycle(task)
        self._trace_emitter = trace_emitter

        await self._publish_runtime_event(
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

        task = self._classifier.classify(task)
        classification = task.metadata.get("classification", "")
        lifecycle.transition(task, TaskState.CLASSIFIED)

        if classification == TaskClassification.UNSUPPORTED.value:
            lifecycle.transition(task, TaskState.FAILED)
            return await self._finish_task(
                task,
                trace_emitter,
                answer="",
                executions=[],
                validation=ValidationResult(
                    valid=False,
                    errors=[task.metadata.get("unsupported_reason", "unsupported task")],
                ),
                plan=None,
                retry_records=[],
                graph_id="",
            )

        plan = self._planner.plan(task, self._registry)
        task.metadata["plan_id"] = plan.plan_id
        lifecycle.transition(task, TaskState.PLANNED)
        await self._publish_runtime_event(
            runtime_event_from_task_state(task, run_id=task.task_id, message="plan created").model_copy(
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

        if classification == TaskClassification.HUMAN_APPROVAL_REQUIRED.value:
            if not task.metadata.get("human_approved"):
                lifecycle.transition(task, TaskState.WAITING_FOR_HUMAN)
                return await self._finish_task(
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
                )
            lifecycle.transition(task, TaskState.RUNNING)
        else:
            lifecycle.transition(task, TaskState.RUNNING)

        graph = plan_to_execution_graph(plan)
        task.metadata["graph_id"] = graph.graph_id

        def _on_retry(record: RetryRecord) -> None:
            trace_emitter.emit(
                task,
                message=(
                    f"retry attempt {record.attempt}: {record.reason} "
                    f"-> {record.alternate_agent_id}"
                ),
            )

        def _on_node_start(node: object) -> None:
            trace_emitter.emit(task, message=f"graph node start: {getattr(node, 'node_id', node)}")

        def _on_node_complete(node: object) -> None:
            trace_emitter.emit(
                task,
                message=f"graph node complete: {getattr(node, 'node_id', node)} "
                f"status={getattr(node, 'status', None)}",
            )

        executions, retry_records, graph = await self._graph_executor.execute(
            graph,
            task,
            plan_criteria=plan.validation_criteria,
            on_retry=_on_retry,
            on_node_start=_on_node_start,
            on_node_complete=_on_node_complete,
        )

        failed_nodes = [
            n.node_id for n in graph.nodes if n.status == ExecutionNodeStatus.FAILED
        ]
        if failed_nodes:
            lifecycle.transition(task, TaskState.VALIDATING)
            lifecycle.transition(task, TaskState.FAILED)
            if isinstance(trace_emitter, PersistingTaskTraceEmitter):
                trace_emitter.finalize()
            return await self._finish_task(
                task,
                trace_emitter,
                answer=self._composer.compose_summary(executions),
                executions=executions,
                validation=ValidationResult(
                    valid=False,
                    errors=[f"graph node failed: {failed_nodes}"],
                ),
                plan=plan,
                retry_records=retry_records,
                graph_id=graph.graph_id,
            )

        lifecycle.transition(task, TaskState.VALIDATING)

        final_validation = ValidationResult(valid=True)
        if executions:
            final_agent = self._registry.get(executions[-1].agent_id)
            final_validation = self._validation_engine.validate(
                executions[-1],
                contract=final_agent.get_contract(),
                capability=task.context.capability,
                plan_criteria=plan.validation_criteria,
            )

        if not final_validation.valid:
            lifecycle.transition(task, TaskState.FAILED)
        elif len(executions) > 1 and not all(
            e.status == AgentExecutionStatus.COMPLETED for e in executions
        ):
            lifecycle.transition(task, TaskState.PARTIALLY_COMPLETED)
        elif task.metadata.get("needs_more_information"):
            lifecycle.transition(task, TaskState.NEEDS_MORE_INFORMATION)
        else:
            lifecycle.transition(task, TaskState.COMPLETED)

        if isinstance(trace_emitter, PersistingTaskTraceEmitter):
            trace_emitter.finalize()

        return await self._finish_task(
            task,
            trace_emitter,
            answer=self._composer.compose_summary(executions),
            executions=executions,
            validation=final_validation,
            plan=plan,
            retry_records=retry_records,
            graph_id=graph.graph_id,
        )

    async def _finish_task(
        self,
        task: Task,
        trace_emitter: TaskTraceEmitter,
        *,
        answer: str,
        executions: List[AgentExecutionResult],
        validation: ValidationResult,
        plan: Optional[NexusPlan],
        retry_records: List[RetryRecord],
        graph_id: str,
    ) -> TaskResult:
        await self._publish_terminal_runtime_event(task)
        return self._build_result(
            task,
            trace_emitter,
            answer=answer,
            executions=executions,
            validation=validation,
            plan=plan,
            retry_records=retry_records,
            graph_id=graph_id,
        )

    def _build_result(
        self,
        task: Task,
        trace_emitter: TaskTraceEmitter,
        *,
        answer: str,
        executions: List[AgentExecutionResult],
        validation: ValidationResult,
        plan: Optional[NexusPlan],
        retry_records: List[RetryRecord],
        graph_id: str,
    ) -> TaskResult:
        primary = executions[-1] if executions else None
        composer_meta = self._composer.compose_metadata(
            executions,
            classification=task.metadata.get("classification", ""),
            plan_id=plan.plan_id if plan else "",
            retry_count=len(retry_records),
        )
        composer_meta["graph_id"] = graph_id
        composer_meta["graph_node_count"] = len(plan.steps) if plan else 0

        metadata = {
            **composer_meta,
            "validation_valid": validation.valid,
            "validation_errors": validation.errors,
            "validation_warnings": validation.warnings,
            "task_trace_events": len(trace_emitter.events),
            "runtime_events": len(self._event_bus.history),
            "retries": [
                {
                    "attempt": r.attempt,
                    "agent_id": r.agent_id,
                    "alternate_agent_id": r.alternate_agent_id,
                    "reason": r.reason,
                }
                for r in retry_records
            ],
        }

        return TaskResult(
            task_id=task.task_id,
            run_id=primary.run_id if primary else task.task_id,
            state=task.state,
            answer=answer,
            agent_id=primary.agent_id if primary else task.agent_id,
            execution_result=primary,
            metadata=metadata,
        )

    async def _publish_runtime_event(self, event: object) -> None:
        from intergrax.runtime.events.runtime_event import RuntimeEvent

        if isinstance(event, RuntimeEvent):
            await self._event_bus.publish(event)

    async def _publish_terminal_runtime_event(self, task: Task) -> None:
        await self._publish_runtime_event(
            runtime_event_from_task_state(task, run_id=task.task_id, message="task terminal")
        )

    def _resolve_lifecycle(self, task: Task) -> tuple[TaskLifecycle, TaskTraceEmitter]:
        if self._lifecycle is not None:
            emitter = self._trace_emitter or TaskTraceEmitter(
                run_id=task.task_id,
                event_bus=self._event_bus,
            )
            return self._lifecycle, emitter
        if self._trace_store is not None:
            return lifecycle_with_persisting_trace(
                run_id=task.task_id,
                trace_store=self._trace_store,
                tenant_id=task.tenant_id,
                user_id=task.user_id,
                session_id=task.session_id or "",
                event_bus=self._event_bus,
            )
        return lifecycle_with_trace(run_id=task.task_id, event_bus=self._event_bus)
