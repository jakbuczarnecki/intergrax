# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

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
from intergrax.runtime.human.escalation import EscalationRouter
from intergrax.runtime.human.models import HumanResponseVerdict, EscalationTarget
from intergrax.runtime.human.store import SQLiteHumanDecisionStore
from intergrax.runtime.cancellation.coordinator import CancellationCoordinator
from intergrax.runtime.long_running.coordinator import LongRunningCoordinator
from intergrax.runtime.long_running.notification import NotificationAdapter
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.runtime.interrupts.handler import ExecutionInterruptHandler
from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.events.store_factory import (
    RuntimeEventStoreSettings,
    create_runtime_event_store,
)
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
from intergrax.runtime.task.task_metadata_keys import (
    GOVERNANCE_HUMAN_REQUEST_KEY,
    HUMAN_REQUEST_CREATED_AT_KEY,
    HUMAN_REQUEST_EXPIRES_AT_KEY,
)
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
from intergrax.runtime.human.request_contract import (
    human_request_event_payload,
    human_request_notification_extra,
)
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
        policy_engine: Optional[RuntimePolicyEngine] = None,
        interrupt_handler: Optional[ExecutionInterruptHandler] = None,
        shadow_manager: Optional[ShadowWorkspaceManager] = None,
        sandbox_manager: Optional[SandboxSessionManager] = None,
        human_decision_store: Optional[SQLiteHumanDecisionStore] = None,
        escalation_router: Optional[EscalationRouter] = None,
        checkpoint_store: Optional[SQLiteTaskCheckpointStore] = None,
        notification_adapter: Optional[NotificationAdapter] = None,
        middleware: Optional[MiddlewarePipeline] = None,
        runtime_event_store: Optional[RuntimeEventPersistence] = None,
        runtime_event_store_settings: Optional[RuntimeEventStoreSettings] = None,
    ) -> None:
        self._registry = registry
        self._runtime_event_store = create_runtime_event_store(
            runtime_event_store_settings,
            implementation=runtime_event_store,
        )
        self._event_bus = event_bus or RuntimeEventBus(persistence=self._runtime_event_store)
        if event_bus is not None and self._runtime_event_store is not None:
            event_bus.attach_persistence(self._runtime_event_store)
        self._middleware = middleware or MiddlewarePipeline(
            middleware=[TraceEmittingMiddleware(self._event_bus)],
        )
        self._human_hooks = HumanApprovalHookCoordinator(self._middleware)
        self._policy_engine = policy_engine or RuntimePolicyEngine()
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
            ),
        )
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

    async def handle_task(self, task: Task) -> TaskResult:
        self._current_task = task
        try:
            return await self._handle_task_impl(task)
        finally:
            self._current_task = None

    async def _handle_task_impl(self, task: Task) -> TaskResult:
        lifecycle, trace_emitter = self._resolve_lifecycle(task)
        self._trace_emitter = trace_emitter

        self._normalize_human_response(task)
        await self._maybe_restore_long_running(task)
        self._normalize_human_response(task)
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
        classification = task.classification or ""
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
                    errors=[
                        task.runtime.classification.unsupported_reason or "unsupported task"
                    ],
                ),
                plan=None,
                retry_records=[],
                graph_id="",
            )

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

        def _on_node_complete(node: object) -> None:
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
                self._finalize_persisting_trace(trace_emitter, executions)
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
                    self._finalize_persisting_trace(trace_emitter, executions)
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
                self._finalize_persisting_trace(trace_emitter, executions)
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
                self._finalize_persisting_trace(trace_emitter, executions)
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
            self._finalize_persisting_trace(trace_emitter, executions)

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
            classification=task.classification or "",
            plan_id=plan.plan_id if plan else "",
            retry_count=len(retry_records),
        )

        execution_metrics = aggregate_execution_metrics(executions)

        gov_human_request = None
        gov = task.runtime.governance
        if gov.human_request is not None:
            gov_human_request = human_request_event_payload(
                gov.human_request,
                created_at_utc=gov.human_request_created_at,
                expires_at_utc=gov.human_request_expires_at,
            )
        elif executions and executions[-1].human_request:
            gov_human_request = human_request_event_payload(
                executions[-1].human_request,
                created_at_utc=SystemTimeProvider.utc_now().isoformat(),
            )

        isolation = TaskIsolationSummary()
        if primary and primary.structured_data.get(SHADOW_WORKSPACE_ID_KEY):
            isolation.shadow_workspace_id = str(primary.structured_data[SHADOW_WORKSPACE_ID_KEY])
            artifact_count = primary.structured_data.get("shadow_artifact_count")
            if artifact_count is not None:
                isolation.shadow_artifact_count = int(artifact_count)

        if primary and primary.structured_data.get(SANDBOX_SESSION_ID_KEY):
            isolation.sandbox_session_id = str(primary.structured_data[SANDBOX_SESSION_ID_KEY])
            operation_count = primary.structured_data.get("sandbox_operation_count")
            if operation_count is not None:
                isolation.sandbox_operation_count = int(operation_count)

        escalation_level = HumanPauseCoordinator.escalation_level(task)
        escalation_chain = list(task.runtime.governance.escalation_chain)

        summary = TaskResultSummary(
            validation=TaskValidationSummary(
                valid=validation.valid,
                errors=list(validation.errors),
                warnings=list(validation.warnings),
            ),
            metrics=TaskExecutionMetrics(
                cost=execution_metrics.cost,
                total_tokens=execution_metrics.total_tokens,
                runtime_events=len(self._event_bus.history),
                task_trace_events=len(trace_emitter.events),
            ),
            isolation=isolation,
            orchestration=TaskOrchestrationSummary(
                classification=composer_meta.get("classification", ""),
                plan_id=composer_meta.get("plan_id", ""),
                graph_id=graph_id,
                graph_node_count=len(plan.steps) if plan else 0,
                agent_count=composer_meta.get("agent_count", 0),
                agent_ids=list(composer_meta.get("agent_ids") or []),
                retry_count=composer_meta.get("retry_count", 0),
                retries=[
                    TaskRetryRecord(
                        attempt=r.attempt,
                        agent_id=r.agent_id,
                        alternate_agent_id=r.alternate_agent_id,
                        reason=r.reason,
                    )
                    for r in retry_records
                ],
                all_completed=bool(composer_meta.get("all_completed")),
            ),
            escalation_level=escalation_level,
            escalation_chain=escalation_chain,
            governance_human_request=gov_human_request,
            checkpoint_id=task.runtime.orchestration.checkpoint_id,
            resume_token=task.runtime.orchestration.resume_token,
            progress_message=task.runtime.orchestration.progress_message,
        )

        self._maybe_cleanup_shadow(task, executions)
        self._maybe_cleanup_sandbox(task, executions)

        task.sync_metadata()
        result = TaskResult(
            task_id=task.task_id,
            run_id=primary.run_id if primary else task.task_id,
            state=task.state,
            answer=answer,
            agent_id=primary.agent_id if primary else task.agent_id,
            execution_result=primary,
            summary=summary,
            metadata=dict(composer_meta),
        )
        for key in (
            GOVERNANCE_HUMAN_REQUEST_KEY,
            HUMAN_REQUEST_CREATED_AT_KEY,
            HUMAN_REQUEST_EXPIRES_AT_KEY,
        ):
            if key in task.metadata:
                result.metadata[key] = task.metadata[key]
        result.sync_metadata()
        return result

    async def _maybe_restore_long_running(self, task: Task) -> None:
        if self._checkpoint_store is None:
            return
        restored = LongRunningCoordinator.restore_if_resuming(task, self._checkpoint_store)
        if restored is None:
            return
        await self._publish_runtime_event(
            runtime_event_from_task_state(
                task,
                run_id=task.task_id,
                message="long-running task restored from checkpoint",
            ).model_copy(
                update={
                    "event_type": RuntimeEventType.RESUMED,
                    "phase": ExecutionPhase.HUMAN_APPROVAL,
                    "payload": {
                        "checkpoint_id": restored.checkpoint_id,
                        "resume_token": restored.resume_token,
                    },
                }
            )
        )
        await LongRunningCoordinator.notify_progress(
            task,
            subject="Task resumed",
            body=restored.progress_message or "checkpoint restored",
            adapter=self._notification_adapter,
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
        if self._checkpoint_store is None or not LongRunningCoordinator.should_checkpoint(task):
            return
        from intergrax.runtime.nexus.execution.execution_graph import ExecutionGraph

        graph_obj = graph if isinstance(graph, ExecutionGraph) else None
        checkpoint = LongRunningCoordinator.persist_checkpoint(
            task,
            self._checkpoint_store,
            progress_message=progress_message,
            plan=plan,
            graph=graph_obj,
            last_execution=last_execution,
        )
        await self._publish_runtime_event(
            runtime_event_from_task_state(
                task,
                run_id=task.task_id,
                message="long-running checkpoint saved",
            ).model_copy(
                update={
                    "event_type": RuntimeEventType.PAUSED,
                    "phase": ExecutionPhase.HUMAN_APPROVAL,
                    "payload": {
                        "checkpoint_id": checkpoint.checkpoint_id,
                        "resume_token": checkpoint.resume_token,
                        "progress_message": progress_message,
                    },
                }
            )
        )
        await LongRunningCoordinator.notify_progress(
            task,
            subject="Task paused",
            body=progress_message,
            adapter=self._notification_adapter,
            extra={
                "checkpoint_id": checkpoint.checkpoint_id,
                **human_request_notification_extra(task),
            },
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

    @staticmethod
    def _finalize_persisting_trace(
        trace_emitter: PersistingTaskTraceEmitter,
        executions: List[AgentExecutionResult],
    ) -> None:
        metrics = aggregate_execution_metrics(executions)
        trace_emitter.finalize(
            duration_ms=metrics.duration_ms,
            llm_usage=metrics.as_llm_usage(),
        )

    def _maybe_cleanup_shadow(self, task: Task, executions: List[AgentExecutionResult]) -> None:
        iso = task.options.isolation
        if not iso.shadow_workspace:
            return
        if not iso.shadow_workspace_cleanup:
            return

        workspace_id = None
        if executions:
            workspace_id = executions[-1].structured_data.get(SHADOW_WORKSPACE_ID_KEY)
        if workspace_id:
            self._shadow_manager.cleanup(str(workspace_id))
        else:
            self._shadow_manager.cleanup_for_task(
                tenant_id=task.tenant_id,
                task_id=task.task_id,
            )

    def _maybe_cleanup_sandbox(self, task: Task, executions: List[AgentExecutionResult]) -> None:
        iso = task.options.isolation
        if not iso.sandbox:
            return
        if not iso.sandbox_cleanup:
            return

        session_id = None
        if executions:
            session_id = executions[-1].structured_data.get(SANDBOX_SESSION_ID_KEY)
        if session_id:
            self._sandbox_manager.cleanup(str(session_id))
        else:
            self._sandbox_manager.cleanup_for_task(
                tenant_id=task.tenant_id,
                task_id=task.task_id,
            )

    @staticmethod
    def _normalize_human_response(task: Task) -> None:
        response = task.options.human.response_text
        if response and task.options.human.verdict is None:
            HumanPauseCoordinator.record_human_response(task, str(response))

    def _persist_human_decision(
        self,
        task: Task,
        verdict: HumanResponseVerdict,
        *,
        response_text: str = "",
    ) -> None:
        if self._human_store is None:
            return
        human_request = HumanPauseCoordinator.human_request_from_task(task)
        target_raw = task.runtime.governance.escalation_target
        target = EscalationTarget(str(target_raw)) if target_raw else None
        record = SQLiteHumanDecisionStore.build_record(
            task_id=task.task_id,
            tenant_id=task.tenant_id,
            user_id=task.user_id,
            verdict=verdict,
            response_text=response_text or str(task.options.human.response_text or ""),
            human_request_id=human_request.request_id if human_request else "",
            escalation_level=HumanPauseCoordinator.escalation_level(task),
            escalation_target=target,
            agent_id=task.agent_id,
            run_id=task.task_id,
        )
        self._human_store.record(record)

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
                self._finalize_persisting_trace(trace_emitter, [])
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
            self._finalize_persisting_trace(trace_emitter, [])
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

        task.options.human.response_text = None
        task.options.human.verdict = None
        task.sync_metadata()

        if outcome.fail_task:
            lifecycle.transition(task, TaskState.FAILED)
            if isinstance(trace_emitter, PersistingTaskTraceEmitter):
                self._finalize_persisting_trace(trace_emitter, [])
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
                self._finalize_persisting_trace(trace_emitter, [])
            return hook_failure
        lifecycle.transition(task, TaskState.WAITING_FOR_HUMAN)
        await self._maybe_checkpoint_long_running(
            task,
            progress_message="awaiting escalated human review",
        )
        if isinstance(trace_emitter, PersistingTaskTraceEmitter):
            self._finalize_persisting_trace(trace_emitter, [])
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
