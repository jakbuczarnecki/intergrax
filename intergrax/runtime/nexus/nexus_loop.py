# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from intergrax.agents.agent_engine import AgentEngine
from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.runtime_cost import aggregate_execution_metrics
from intergrax.contracts.validation import ValidationResult
from intergrax.runtime.nexus.execution.execution_graph import ExecutionNodeStatus
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
from intergrax.runtime.human.pause import HumanPauseCoordinator
from intergrax.runtime.human.hitl_hooks import HumanApprovalHookCoordinator, HumanApprovalHookError
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.hooks.nexus_lifecycle_hooks import (
    NexusLifecycleHookCoordinator,
    NexusLifecycleHookError,
)
from intergrax.runtime.human.escalation import EscalationRouter
from intergrax.runtime.human.models import HumanResponseVerdict, EscalationTarget
from intergrax.runtime.human.store import SQLiteHumanDecisionStore
from intergrax.runtime.cancellation.coordinator import CancellationCoordinator
from intergrax.runtime.long_running.coordinator import LongRunningCoordinator
from intergrax.runtime.long_running.notification import NotificationAdapter
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.runtime.interrupts.handler import ExecutionInterruptHandler
from intergrax.runtime.hooks.governance_hooks import hook_context_for_task, run_hook_pair
from intergrax.runtime.policy.policy_engine import PolicyEngine, coerce_policy_engine
from intergrax.runtime.nexus.orchestration.human_response import (
    normalize_human_response,
    persist_human_decision,
)
from intergrax.runtime.nexus.orchestration.long_running_bridge import (
    maybe_checkpoint_long_running,
    maybe_restore_long_running,
)
from intergrax.runtime.nexus.orchestration.task_finisher import build_nexus_task_result
from intergrax.runtime.nexus.execution.execution_graph import ExecutionNode
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.events.store import resolve_runtime_event_persistence
from intergrax.runtime.task_memory.persistence_contract import TaskMemoryPersistence
from intergrax.runtime.task_memory.store import resolve_task_memory_persistence
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.trace_bridge import runtime_event_from_task_state
from intergrax.runtime.task.task import Task, TaskResult, TaskState
from intergrax.runtime.task.task_contract import (
    TaskExecutionMetrics,
    TaskIsolationSummary,
    TaskOrchestrationSummary,
    TaskResultSummary,
    TaskRetryRecord,
    TaskValidationSummary,
)
from intergrax.runtime.task.task_metadata_keys import TaskMetadataKey
from intergrax.runtime.task.task_lifecycle import TaskLifecycle
from intergrax.runtime.task.task_trace import (
    PersistingTaskTraceEmitter,
    TaskTraceEmitter,
    lifecycle_with_persisting_trace,
    lifecycle_with_trace,
)
from intergrax.agents.uaep import UAEPExecutor
from intergrax.runtime.workspace.manager import ShadowWorkspaceManager
from intergrax.runtime.workspace.shadow_workspace import SHADOW_WORKSPACE_ID_KEY
from intergrax.runtime.sandbox.manager import SandboxSessionManager
from intergrax.runtime.sandbox.sandbox_runtime import SANDBOX_SESSION_ID_KEY
from intergrax.runtime.human.request_contract import human_request_event_payload
from intergrax.utils.time_provider import SystemTimeProvider
from intergrax.runtime.middleware.pipeline import MiddlewarePipeline
from intergrax.runtime.middleware.trace_middleware import TraceEmittingMiddleware

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
        policy_engine: PolicyEngine | None = None,
        interrupt_handler: Optional[ExecutionInterruptHandler] = None,
        shadow_manager: Optional[ShadowWorkspaceManager] = None,
        sandbox_manager: Optional[SandboxSessionManager] = None,
        human_decision_store: Optional[SQLiteHumanDecisionStore] = None,
        escalation_router: Optional[EscalationRouter] = None,
        checkpoint_store: Optional[SQLiteTaskCheckpointStore] = None,
        notification_adapter: Optional[NotificationAdapter] = None,
        middleware: Optional[MiddlewarePipeline] = None,
        runtime_event_store: Optional[RuntimeEventPersistence] = None,
        runtime_events_db_path: Optional[Path] = None,
        task_memory_store: Optional[TaskMemoryPersistence] = None,
        task_memory_db_path: Optional[Path] = None,
    ) -> None:
        self._registry = registry
        self._runtime_event_store = resolve_runtime_event_persistence(
            db_path=runtime_events_db_path,
            implementation=runtime_event_store,
        )
        self._task_memory_store = resolve_task_memory_persistence(
            db_path=task_memory_db_path,
            implementation=task_memory_store,
        )
        self._event_bus = event_bus or RuntimeEventBus(persistence=self._runtime_event_store)
        if event_bus is not None and self._runtime_event_store is not None:
            event_bus.attach_persistence(self._runtime_event_store)
        self._middleware = middleware or MiddlewarePipeline(
            middleware=[TraceEmittingMiddleware(self._event_bus)],
        )
        self._human_hooks = HumanApprovalHookCoordinator(self._middleware)
        self._lifecycle_hooks = NexusLifecycleHookCoordinator(self._middleware)
        self._policy_engine = coerce_policy_engine(policy_engine)
        self._interrupt_handler = interrupt_handler or ExecutionInterruptHandler(
            policy_engine=self._policy_engine,
        )
        self._shadow_manager = shadow_manager or ShadowWorkspaceManager()
        self._sandbox_manager = sandbox_manager or SandboxSessionManager()
        self._human_store = human_decision_store
        self._escalation_router = escalation_router or EscalationRouter()
        self._checkpoint_store = checkpoint_store
        self._notification_adapter = notification_adapter
        self._engine = AgentEngine(
            registry,
            event_bus=self._event_bus,
            policy_engine=self._policy_engine,
            uaep_executor=UAEPExecutor(
                event_bus=self._event_bus,
                policy_engine=self._policy_engine,
                shadow_manager=self._shadow_manager,
                sandbox_manager=self._sandbox_manager,
                middleware=self._middleware,
                task_memory_store=self._task_memory_store,
            ),
        )
        self._classifier = classifier or ClassifyingTaskClassifier(registry)
        self._planner = planner or TaskPlanner()
        self._validation_engine = validation_engine or NexusValidationEngine()
        self._retry_engine = retry_engine or RetryEngine(
            registry,
            policy=retry_policy or RetryPolicy(),
            middleware=self._middleware,
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
            event_bus=self._event_bus,
            middleware=self._middleware,
        )
        self._composer = FinalResponseComposer()
        self._lifecycle = lifecycle
        self._trace_emitter = trace_emitter
        self._trace_store = trace_store
        self._current_task: Optional[Task] = None

    @property
    def registry(self) -> AgentRegistry:
        return self._registry

    @property
    def trace_emitter(self) -> Optional[TaskTraceEmitter]:
        return self._trace_emitter

    @property
    def event_bus(self) -> RuntimeEventBus:
        return self._event_bus

    @property
    def runtime_event_store(self) -> Optional[RuntimeEventPersistence]:
        return self._runtime_event_store

    @property
    def task_memory_store(self) -> Optional[TaskMemoryPersistence]:
        return self._task_memory_store

    @property
    def interrupt_handler(self) -> ExecutionInterruptHandler:
        return self._interrupt_handler

    @property
    def shadow_manager(self) -> ShadowWorkspaceManager:
        return self._shadow_manager

    @property
    def sandbox_manager(self) -> SandboxSessionManager:
        return self._sandbox_manager

    @property
    def middleware(self) -> MiddlewarePipeline:
        return self._middleware

    @property
    def policy_engine(self) -> PolicyEngine:
        return self._policy_engine

    async def handle_task(self, task: Task) -> TaskResult:
        self._current_task = task
        try:
            return await self._handle_task_impl(task)
        finally:
            self._current_task = None

    async def _handle_task_impl(self, task: Task) -> TaskResult:
        lifecycle, trace_emitter = self._resolve_lifecycle(task)
        self._trace_emitter = trace_emitter

        normalize_human_response(task)
        await self._maybe_restore_long_running(task)
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
            return await self._handle_human_rejection(task, trace_emitter, lifecycle)
        if verdict == HumanResponseVerdict.ESCALATE:
            return await self._handle_human_escalation(task, trace_emitter, lifecycle)

        if HumanPauseCoordinator.is_resumed(task):
            await self._publish_runtime_event(
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
            await self._human_hooks.after_response(
                task,
                verdict=HumanResponseVerdict.APPROVE.value,
            )
            HumanPauseCoordinator.clear_pause(task)

        hook_failure = await self._run_lifecycle_hook(
            before=True,
            point=HookPoint.BEFORE_TASK_INTAKE,
            task=task,
            phase=ExecutionPhase.INTAKE,
            trace_emitter=trace_emitter,
            lifecycle=lifecycle,
        )
        if hook_failure is not None:
            return hook_failure

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

        hook_failure = await self._run_lifecycle_hook(
            before=False,
            point=HookPoint.AFTER_TASK_INTAKE,
            task=task,
            phase=ExecutionPhase.INTAKE,
            trace_emitter=trace_emitter,
            lifecycle=lifecycle,
        )
        if hook_failure is not None:
            return hook_failure

        hook_failure = await self._run_lifecycle_hook(
            before=True,
            point=HookPoint.BEFORE_CLASSIFICATION,
            task=task,
            phase=ExecutionPhase.CLASSIFICATION,
            trace_emitter=trace_emitter,
            lifecycle=lifecycle,
        )
        if hook_failure is not None:
            return hook_failure

        task = self._classifier.classify(task)
        classification = task.classification or ""
        lifecycle.transition(task, TaskState.CLASSIFIED)

        hook_failure = await self._run_lifecycle_hook(
            before=False,
            point=HookPoint.AFTER_CLASSIFICATION,
            task=task,
            phase=ExecutionPhase.CLASSIFICATION,
            trace_emitter=trace_emitter,
            lifecycle=lifecycle,
            extra={"classification": classification},
        )
        if hook_failure is not None:
            return hook_failure

        if classification == TaskClassification.UNSUPPORTED.value:
            lifecycle.transition(task, TaskState.FAILED)
            return await self._finish_task(
                task,
                trace_emitter,
                answer="",
                executions=[],
                validation=ValidationResult(
                    valid=False,
                    errors=[
                        task.runtime.classification.unsupported_reason or "unsupported task"
                    ],
                ),
                plan=None,
                retry_records=[],
                graph_id="",
            )

        hook_failure = await self._run_lifecycle_hook(
            before=True,
            point=HookPoint.BEFORE_PLANNING,
            task=task,
            phase=ExecutionPhase.PLANNING,
            trace_emitter=trace_emitter,
            lifecycle=lifecycle,
            extra={"classification": classification},
        )
        if hook_failure is not None:
            return hook_failure

        plan = self._planner.plan(task, self._registry)
        task.runtime.orchestration.plan_id = plan.plan_id
        task.sync_metadata()
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

        hook_failure = await self._run_lifecycle_hook(
            before=False,
            point=HookPoint.AFTER_PLANNING,
            task=task,
            phase=ExecutionPhase.PLANNING,
            trace_emitter=trace_emitter,
            lifecycle=lifecycle,
            extra={"plan_id": plan.plan_id, "step_count": len(plan.steps)},
        )
        if hook_failure is not None:
            return hook_failure

        if classification == TaskClassification.HUMAN_APPROVAL_REQUIRED.value:
            if not HumanPauseCoordinator.is_resumed(task):
                hook_failure = await self._run_before_human_pause(task, trace_emitter, lifecycle)
                if hook_failure is not None:
                    return hook_failure
                lifecycle.transition(task, TaskState.WAITING_FOR_HUMAN)
                await self._maybe_checkpoint_long_running(
                    task,
                    progress_message="awaiting human approval",
                    plan=plan,
                )
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
        task.runtime.orchestration.graph_id = graph.graph_id
        task.sync_metadata()

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

        def _on_node_complete(node: ExecutionNode) -> None:
            trace_emitter.emit(
                task,
                message=f"graph node complete: {getattr(node, 'node_id', node)} "
                f"status={getattr(node, 'status', None)}",
            )

        executions, retry_records, graph, graph_cancelled = await self._graph_executor.execute(
            graph,
            task,
            plan_criteria=plan.validation_criteria,
            on_retry=_on_retry,
            on_node_start=_on_node_start,
            on_node_complete=_on_node_complete,
        )

        if graph_cancelled or CancellationCoordinator.is_requested(task.metadata):
            await self._publish_runtime_event(
                runtime_event_from_task_state(
                    task,
                    run_id=task.task_id,
                    message="task cancellation propagated",
                ).model_copy(
                    update={
                        "event_type": RuntimeEventType.CANCELLED,
                        "phase": ExecutionPhase.COMPLETION,
                        "payload": {
                            "reason": task.metadata.get("cancellation_reason", ""),
                        },
                    }
                )
            )
            lifecycle.transition(task, TaskState.VALIDATING)
            lifecycle.transition(task, TaskState.CANCELLED)
            CancellationCoordinator.clear_checkpoint_state(task)
            CancellationCoordinator.clear(task)
            if isinstance(trace_emitter, PersistingTaskTraceEmitter):
                await self._finalize_persisting_trace(
                    trace_emitter, executions, task_id=task.task_id
                )
            return await self._finish_task(
                task,
                trace_emitter,
                answer=self._composer.compose_summary(executions),
                executions=executions,
                validation=ValidationResult(valid=False, errors=["task_cancelled"]),
                plan=plan,
                retry_records=retry_records,
                graph_id=graph.graph_id,
            )

        if executions and executions[-1].status == AgentExecutionStatus.NEEDS_INPUT:
            paused = executions[-1]
            created_at_utc = SystemTimeProvider.utc_now().isoformat()
            human_payload = (
                human_request_event_payload(
                    paused.human_request,
                    created_at_utc=created_at_utc,
                )
                if paused.human_request
                else {}
            )
            await self._publish_runtime_event(
                runtime_event_from_task_state(
                    task,
                    run_id=task.task_id,
                    message="human approval requested",
                ).model_copy(
                    update={
                        "event_type": RuntimeEventType.HUMAN_APPROVAL_REQUESTED,
                        "phase": ExecutionPhase.HUMAN_APPROVAL,
                        "payload": {
                            "human_request": human_payload,
                        },
                    }
                )
            )
            hook_failure = await self._run_before_human_pause(
                task,
                trace_emitter,
                lifecycle,
                agent_id=paused.agent_id,
                execution=paused,
            )
            if hook_failure is not None:
                if isinstance(trace_emitter, PersistingTaskTraceEmitter):
                    await self._finalize_persisting_trace(
                    trace_emitter, executions, task_id=task.task_id
                )
                return hook_failure
            HumanPauseCoordinator.apply_pause(task, paused)
            lifecycle.transition(task, TaskState.WAITING_FOR_HUMAN)
            await self._maybe_checkpoint_long_running(
                task,
                progress_message="awaiting human input",
                plan=plan,
                graph=graph,
                last_execution=paused,
            )
            if isinstance(trace_emitter, PersistingTaskTraceEmitter):
                await self._finalize_persisting_trace(
                    trace_emitter, executions, task_id=task.task_id
                )
            return await self._finish_task(
                task,
                trace_emitter,
                answer=paused.summary,
                executions=executions,
                validation=ValidationResult(
                    valid=False,
                    errors=["awaiting human input"],
                ),
                plan=plan,
                retry_records=retry_records,
                graph_id=graph.graph_id,
            )

        failed_nodes = [
            n.node_id for n in graph.nodes if n.status == ExecutionNodeStatus.FAILED
        ]
        if failed_nodes:
            lifecycle.transition(task, TaskState.VALIDATING)
            lifecycle.transition(task, TaskState.FAILED)
            if LongRunningCoordinator.is_long_running(task):
                await self._maybe_checkpoint_long_running(
                    task,
                    progress_message=f"graph failed at {failed_nodes}",
                    plan=plan,
                    graph=graph,
                    last_execution=executions[-1] if executions else None,
                )
            if isinstance(trace_emitter, PersistingTaskTraceEmitter):
                await self._finalize_persisting_trace(
                    trace_emitter, executions, task_id=task.task_id
                )
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
        elif task.runtime.orchestration.needs_more_information:
            lifecycle.transition(task, TaskState.NEEDS_MORE_INFORMATION)
        else:
            lifecycle.transition(task, TaskState.COMPLETED)

        if isinstance(trace_emitter, PersistingTaskTraceEmitter):
            await self._finalize_persisting_trace(
                trace_emitter, executions, task_id=task.task_id
            )

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
        try:
            await self._lifecycle_hooks.before(
                HookPoint.BEFORE_FINALIZATION,
                task,
                phase=ExecutionPhase.COMPLETION,
                extra={"task_state": task.state.value},
            )
        except NexusLifecycleHookError as exc:
            await self._publish_terminal_runtime_event(task)
            return self._build_result(
                task,
                trace_emitter,
                answer=answer,
                executions=executions,
                validation=ValidationResult(valid=False, errors=[str(exc)]),
                plan=plan,
                retry_records=retry_records,
                graph_id=graph_id,
            )

        await self._publish_terminal_runtime_event(task)
        result = self._build_result(
            task,
            trace_emitter,
            answer=answer,
            executions=executions,
            validation=validation,
            plan=plan,
            retry_records=retry_records,
            graph_id=graph_id,
        )
        try:
            await self._lifecycle_hooks.after(
                HookPoint.AFTER_FINALIZATION,
                task,
                phase=ExecutionPhase.COMPLETION,
                extra={"task_state": task.state.value},
            )
        except NexusLifecycleHookError:
            pass
        return result

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
        return build_nexus_task_result(
            task,
            trace_emitter,
            answer=answer,
            executions=executions,
            validation=validation,
            plan=plan,
            retry_records=retry_records,
            graph_id=graph_id,
            composer=self._composer,
            event_bus=self._event_bus,
            shadow_manager=self._shadow_manager,
            sandbox_manager=self._sandbox_manager,
        )

    async def _maybe_restore_long_running(self, task: Task) -> None:
        await maybe_restore_long_running(
            task,
            checkpoint_store=self._checkpoint_store,
            publish=self._publish_runtime_event,
            notification_adapter=self._notification_adapter,
        )

    async def _maybe_checkpoint_long_running(
        self,
        task: Task,
        *,
        progress_message: str,
        plan: Optional[NexusPlan] = None,
        graph: Optional[object] = None,
        last_execution: Optional[AgentExecutionResult] = None,
    ) -> None:
        await maybe_checkpoint_long_running(
            task,
            checkpoint_store=self._checkpoint_store,
            publish=self._publish_runtime_event,
            notification_adapter=self._notification_adapter,
            progress_message=progress_message,
            plan=plan,
            graph=graph,
            last_execution=last_execution,
        )

    async def _publish_runtime_event(
        self,
        event: object,
        *,
        task: Optional[Task] = None,
    ) -> None:
        from intergrax.runtime.events.runtime_event import RuntimeEvent

        if isinstance(event, RuntimeEvent):
            scoped_task = task or self._current_task
            if scoped_task is not None and not event.tenant_id:
                event = event.model_copy(update={"tenant_id": scoped_task.tenant_id})
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

    async def _finalize_persisting_trace(
        self,
        trace_emitter: PersistingTaskTraceEmitter,
        executions: List[AgentExecutionResult],
        *,
        task_id: str = "",
    ) -> None:
        ctx = hook_context_for_task(
            task_id=task_id or trace_emitter._run_id,
            run_id=trace_emitter._run_id,
            phase=ExecutionPhase.TRACE_PERSISTENCE,
        )
        await run_hook_pair(
            self._middleware,
            HookPoint.BEFORE_TRACE_PERSIST,
            HookPoint.AFTER_TRACE_PERSIST,
            ctx,
        )
        metrics = aggregate_execution_metrics(executions)
        trace_emitter.finalize(
            duration_ms=metrics.duration_ms,
            llm_usage=metrics.as_llm_usage(),
        )

    def _persist_human_decision(
        self,
        task: Task,
        verdict: HumanResponseVerdict,
        *,
        response_text: str = "",
    ) -> None:
        persist_human_decision(
            task,
            verdict,
            human_store=self._human_store,
            response_text=response_text,
        )

    async def _run_before_human_pause(
        self,
        task: Task,
        trace_emitter: TaskTraceEmitter,
        lifecycle: TaskLifecycle,
        *,
        agent_id: Optional[str] = None,
        execution: Optional[AgentExecutionResult] = None,
    ) -> Optional[TaskResult]:
        try:
            await self._human_hooks.before_pause(
                task,
                agent_id=agent_id,
                execution=execution,
            )
        except HumanApprovalHookError as exc:
            lifecycle.transition(task, TaskState.FAILED)
            if isinstance(trace_emitter, PersistingTaskTraceEmitter):
                await self._finalize_persisting_trace(
                    trace_emitter, [], task_id=task.task_id
                )
            return await self._finish_task(
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

    async def _run_lifecycle_hook(
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
                await self._lifecycle_hooks.before(point, task, phase=phase, extra=extra)
            else:
                await self._lifecycle_hooks.after(point, task, phase=phase, extra=extra)
        except NexusLifecycleHookError as exc:
            lifecycle.transition(task, TaskState.FAILED)
            if isinstance(trace_emitter, PersistingTaskTraceEmitter):
                await self._finalize_persisting_trace(
                    trace_emitter, [], task_id=task.task_id
                )
            return await self._finish_task(
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

    async def _handle_human_rejection(
        self,
        task: Task,
        trace_emitter: TaskTraceEmitter,
        lifecycle: TaskLifecycle,
    ) -> TaskResult:
        await self._publish_runtime_event(
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
            )
        )
        await self._human_hooks.after_response(
            task,
            verdict=HumanResponseVerdict.REJECT.value,
        )
        self._persist_human_decision(task, HumanResponseVerdict.REJECT)
        lifecycle.transition(task, TaskState.FAILED)
        if isinstance(trace_emitter, PersistingTaskTraceEmitter):
            await self._finalize_persisting_trace(trace_emitter, [], task_id=task.task_id)
        return await self._finish_task(
            task,
            trace_emitter,
            answer="",
            executions=[],
            validation=ValidationResult(valid=False, errors=["human rejected"]),
            plan=None,
            retry_records=[],
            graph_id="",
        )

    async def _handle_human_escalation(
        self,
        task: Task,
        trace_emitter: TaskTraceEmitter,
        lifecycle: TaskLifecycle,
    ) -> TaskResult:
        outcome = self._escalation_router.route(task)
        self._escalation_router.apply_to_task(task, outcome)
        self._persist_human_decision(task, HumanResponseVerdict.ESCALATE)

        await self._publish_runtime_event(
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
            )
        )
        await self._human_hooks.after_response(
            task,
            verdict=HumanResponseVerdict.ESCALATE.value,
        )

        progress_message = "awaiting escalated human review"
        await LongRunningCoordinator.notify_escalation(
            task,
            outcome=outcome,
            progress_message=progress_message,
            adapter=self._notification_adapter,
        )

        task.options.human.response_text = None
        task.options.human.verdict = None
        task.sync_metadata()

        if outcome.fail_task:
            lifecycle.transition(task, TaskState.FAILED)
            if isinstance(trace_emitter, PersistingTaskTraceEmitter):
                await self._finalize_persisting_trace(
                    trace_emitter, [], task_id=task.task_id
                )
            return await self._finish_task(
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
        hook_failure = await self._run_before_human_pause(task, trace_emitter, lifecycle)
        if hook_failure is not None:
            if isinstance(trace_emitter, PersistingTaskTraceEmitter):
                await self._finalize_persisting_trace(
                    trace_emitter, [], task_id=task.task_id
                )
            return hook_failure
        lifecycle.transition(task, TaskState.WAITING_FOR_HUMAN)
        await self._maybe_checkpoint_long_running(
            task,
            progress_message=progress_message,
        )
        if isinstance(trace_emitter, PersistingTaskTraceEmitter):
            await self._finalize_persisting_trace(trace_emitter, [], task_id=task.task_id)
        return await self._finish_task(
            task,
            trace_emitter,
            answer="",
            executions=[],
            validation=ValidationResult(valid=False, errors=["awaiting escalated human review"]),
            plan=None,
            retry_records=[],
            graph_id="",
        )
