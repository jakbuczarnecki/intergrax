# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from intergrax.agents.agent_engine import AgentEngine
from intergrax.agents.persistence.checkpoint_store import AgentCheckpointStore
from intergrax.agents.persistence.compensation_queue_store import CompensationQueueStore
from intergrax.contracts.idempotency_store import IdempotencyStore
from intergrax.contracts.execution_identity import (
    ActiveExecutionIdentity,
    AttemptId,
    ExecutionId,
    RunId,
    require_active_execution_id,
    require_active_execution_identity,
    validate_attempt_id,
    validate_run_id,
)
from intergrax.runtime.governance.active_execution_authority import (
    require_active_execution_authority,
)
from intergrax.runtime.governance.service import GovernanceService
from intergrax.contracts.agent_execution_result import (
    AgentExecutionResult,
    AgentExecutionStatus,
)
from intergrax.contracts.validation import ValidationResult
from intergrax.runtime.nexus.agent_router import AgentRouter
from intergrax.runtime.nexus.context.context_manager import ContextManager
from intergrax.runtime.nexus.execution.graph_builder import plan_to_execution_graph
from intergrax.runtime.nexus.execution.graph_executor import GraphExecutor
from intergrax.runtime.nexus.planning.nexus_planner_protocol import NexusTaskPlannerProtocol
from intergrax.runtime.nexus.planning.task_planner import NexusPlan, TaskPlanner
from intergrax.runtime.nexus.task_classifier_protocol import NexusTaskClassifierProtocol
from intergrax.contracts.orchestration_enums import MergeStrategy
from intergrax.runtime.architecture.online_evaluation_models import (
    OnlineEvaluationMode,
    OnlineEvaluationObservation,
)
from intergrax.runtime.nexus.response.final_response_composer import FinalResponseComposer
from intergrax.runtime.nexus.retry.retry_engine import RetryEngine, RetryPolicy, RetryRecord
from intergrax.runtime.nexus.task_classifier import ClassifyingTaskClassifier
from intergrax.runtime.nexus.tracing.persistence_models import RunTraceReader, RunTraceWriter
from intergrax.runtime.nexus.validation.validation_engine import NexusValidationEngine
from intergrax.runtime.human.hitl_hooks import HumanApprovalHookCoordinator, HumanApprovalHookError
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.hooks.nexus_lifecycle_hooks import (
    NexusLifecycleHookCoordinator,
    NexusLifecycleHookError,
    publish_nexus_lifecycle_hook_failure,
)
from intergrax.runtime.human.escalation import EscalationRouter
from intergrax.runtime.human.models import HumanResponseVerdict, EscalationTarget
from intergrax.runtime.human.persistence_contract import HumanDecisionPersistence
from intergrax.runtime.long_running.notification import NotificationAdapter
from intergrax.runtime.long_running.persistence_contract import TaskCheckpointPersistence
from intergrax.runtime.interrupts.handler import ExecutionInterruptHandler
from intergrax.runtime.policy.policy_engine import PolicyEngine, coerce_policy_engine
from intergrax.runtime.nexus.orchestration.human_response import persist_human_decision
from intergrax.runtime.nexus.orchestration.long_running_bridge import (
    maybe_checkpoint_long_running,
    maybe_restore_long_running,
)
from intergrax.runtime.nexus.orchestration.graph_runner import NexusGraphRunner
from intergrax.runtime.nexus.orchestration.hitl_runner import NexusHitlRunner
from intergrax.runtime.nexus.orchestration.intake_runner import NexusIntakeRunner
from intergrax.runtime.nexus.orchestration.planning_runner import NexusPlanningRunner
from intergrax.runtime.nexus.orchestration.lifecycle_bridge import (
    finalize_persisting_trace,
    resolve_nexus_lifecycle,
)
from intergrax.runtime.nexus.orchestration.task_events import NexusRuntimeEventPublisher
from intergrax.runtime.nexus.orchestration.task_finisher import build_nexus_task_result
from intergrax.runtime.nexus.execution.execution_graph import ExecutionGraph
from intergrax.runtime.registry.agent_registry_read import AgentRegistryRead
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.events.runtime_event import RuntimeEvent
from intergrax.runtime.events.store import resolve_runtime_event_persistence
from intergrax.runtime.task_memory.persistence_contract import TaskMemoryPersistence
from intergrax.runtime.task_memory.store import resolve_task_memory_persistence
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.task.task import Task, TaskResult, TaskState
from intergrax.runtime.task.task_lifecycle import TaskLifecycle
from intergrax.runtime.task.task_trace import PersistingTaskTraceEmitter, TaskTraceEmitter
from intergrax.agents.uaep import UAEPExecutor
from intergrax.runtime.workspace.manager import ShadowWorkspaceManager
from intergrax.runtime.sandbox.manager import SandboxSessionManager
from intergrax.runtime.adaptive.signal_collector import SignalCollector
from intergrax.runtime.adaptive.signal_emission import record_task_outcome_signal
from intergrax.runtime.architecture.online_evaluation_registry import OnlineEvaluationRegistry
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.execution.active_execution_budget import (
    require_active_execution_budget,
)
from intergrax.runtime.execution.budget import create_execution_budget_ledger_factory
from intergrax.runtime.execution.attempt_lifecycle import (
    AttemptLifecycleService,
    InMemoryAttemptLifecycleStore,
)
from intergrax.runtime.execution.attempt_lifecycle.durability_policy import (
    validate_durable_attempt_lifecycle_for_composition,
)
from intergrax.runtime.execution.execution_terminal import (
    ExecutionTerminalService,
    wire_execution_terminal_store,
)
from intergrax.runtime.execution.execution_terminal.durability_policy import (
    validate_durable_execution_terminal_for_composition,
)
from intergrax.contracts.execution_terminal import (
    ExecutionTerminalConflictError,
    ExecutionTerminalError,
)
from intergrax.runtime.execution.execution_terminal.persistence import (
    TerminalCommitResolution,
    reconcile_task_state_with_terminal_outcome,
    terminal_outcome_from_task_state,
    terminal_reason_for_task_state,
    validate_terminal_run_id_consistency,
)
from intergrax.runtime.diagnostics.terminal_execution_diagnostic_trigger import (
    TerminalExecutionDiagnosticTriggerProtocol,
)
from intergrax.runtime.middleware.pipeline import MiddlewarePipeline
from intergrax.runtime.middleware.trace_middleware import TraceEmittingMiddleware

if TYPE_CHECKING:
    from intergrax.runtime.decision_flow import DecisionFlowGate
    from intergrax.contracts.agent_execution_result import AgentExecutionResult
    from intergrax.runtime.execution.authority.policy import ExecutionAuthorityPolicy
    from intergrax.runtime.execution.budget.ledger import (
        ExecutionBudgetLedger,
        ExecutionBudgetLedgerFactory,
    )
    from intergrax.runtime.execution.budget.policy import ExecutionBudgetAllocationPolicy

class NexusLoop:
    """
    Global Nexus loop (§9.1, §41).

    Task → classify → plan → execution graph → validate → compose response.
    """

    def __init__(
        self,
        registry: AgentRegistryRead,
        *,
        classifier: NexusTaskClassifierProtocol | None = None,
        planner: NexusTaskPlannerProtocol | None = None,
        max_parallel_nodes: int | None = None,
        max_inflight_nodes: int | None = None,
        max_delegation_depth: int | None = None,
        max_run_retries: int = 0,
        merge_strategy: MergeStrategy = MergeStrategy.CONCAT,
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
        human_decision_store: HumanDecisionPersistence | None = None,
        escalation_router: Optional[EscalationRouter] = None,
        checkpoint_store: Optional[TaskCheckpointPersistence] = None,
        agent_checkpoint_store: AgentCheckpointStore | None = None,
        compensation_queue_store: CompensationQueueStore | None = None,
        idempotency_store: IdempotencyStore | None = None,
        declarative_tool_invoker: DeclarativeToolInvoker | None = None,
        notification_adapter: Optional[NotificationAdapter] = None,
        middleware: Optional[MiddlewarePipeline] = None,
        production_mode: bool = False,
        runtime_event_store: Optional[RuntimeEventPersistence] = None,
        runtime_events_db_path: Optional[Path] = None,
        task_memory_store: Optional[TaskMemoryPersistence] = None,
        task_memory_db_path: Optional[Path] = None,
        signal_collector: SignalCollector | None = None,
        evaluation_registry: OnlineEvaluationRegistry | None = None,
        run_budget: RunBudget | None = None,
        decision_flow_gate: Optional["DecisionFlowGate[AgentExecutionResult]"] = None,
        emit_coordination_advisory: bool = False,
        allow_dynamic_replan: bool = False,
        denied_planner_model_ids: tuple[str, ...] = (),
        planner_model_id: str | None = None,
        governance_service: GovernanceService | None = None,
        terminal_diagnostic_trigger: TerminalExecutionDiagnosticTriggerProtocol | None = None,
        authority_policy: "ExecutionAuthorityPolicy | None" = None,
        budget_allocation_policy: "ExecutionBudgetAllocationPolicy | None" = None,
        execution_budget_ledger_factory: "ExecutionBudgetLedgerFactory | None" = None,
        attempt_lifecycle: AttemptLifecycleService | None = None,
        execution_terminal: ExecutionTerminalService | None = None,
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
            event_bus=self._event_bus,
        )
        if middleware is not None and isinstance(middleware, MiddlewarePipeline):
            middleware.configure_hook_runtime(
                hook_timeout_seconds=middleware.hook_timeout_seconds,
                event_bus=self._event_bus,
            )
        self._human_hooks = HumanApprovalHookCoordinator(self._middleware)
        self._lifecycle_hooks = NexusLifecycleHookCoordinator(self._middleware)
        self._policy_engine = coerce_policy_engine(policy_engine)
        self._governance_service = governance_service
        self._terminal_diagnostic_trigger = terminal_diagnostic_trigger
        self._interrupt_handler = interrupt_handler or ExecutionInterruptHandler(
            policy_engine=self._policy_engine,
            allow_dynamic_replan=allow_dynamic_replan,
        )
        self._shadow_manager = shadow_manager or ShadowWorkspaceManager()
        self._sandbox_manager = sandbox_manager or SandboxSessionManager()
        self._human_store = human_decision_store
        self._escalation_router = escalation_router or EscalationRouter()
        self._checkpoint_store = checkpoint_store
        self._agent_checkpoint_store = agent_checkpoint_store
        self._compensation_queue_store = compensation_queue_store
        self._idempotency_store = idempotency_store
        self._declarative_tool_invoker = declarative_tool_invoker
        self._notification_adapter = notification_adapter
        self._execution_identity = ActiveExecutionIdentity()
        if context_manager is None:
            self._context_manager = ContextManager(
                event_bus=self._event_bus,
                execution_identity=self._execution_identity,
            )
        else:
            self._context_manager = context_manager
            self._context_manager.use_execution_identity(self._execution_identity)
        self._engine = AgentEngine(
            registry,
            production_mode=production_mode,
            event_bus=self._event_bus,
            policy_engine=self._policy_engine,
            uaep_executor=UAEPExecutor(
                event_bus=self._event_bus,
                policy_engine=self._policy_engine,
                governance_service=governance_service,
                shadow_manager=self._shadow_manager,
                sandbox_manager=self._sandbox_manager,
                middleware=self._middleware,
                task_memory_store=self._task_memory_store,
                context_engine=self._context_manager.context_engine,
                llm_adapter=self._context_manager.llm_adapter,
            ),
        )
        self._classifier = classifier or ClassifyingTaskClassifier(registry)
        self._planner = planner or TaskPlanner()
        self._validation_engine = validation_engine or NexusValidationEngine()
        resolved_retry_policy = retry_policy or RetryPolicy()
        self._retry_engine = retry_engine or RetryEngine(
            registry,
            policy=resolved_retry_policy,
            middleware=self._middleware,
        )
        self._router = AgentRouter(
            registry,
            production_mode=production_mode,
            event_bus=self._event_bus,
        )
        self._context_manager.bind_middleware(self._middleware)
        self._graph_executor = graph_executor or GraphExecutor(
            registry,
            engine=self._engine,
            router=self._router,
            validation_engine=self._validation_engine,
            retry_engine=self._retry_engine,
            context_manager=self._context_manager,
            event_bus=self._event_bus,
            middleware=self._middleware,
            max_parallel_nodes=max_parallel_nodes,
            max_inflight_nodes=max_inflight_nodes,
            max_delegation_depth=max_delegation_depth,
            decision_flow_gate=decision_flow_gate,
            agent_checkpoint_store=agent_checkpoint_store,
            compensation_queue_store=compensation_queue_store,
            idempotency_store=idempotency_store,
            declarative_tool_invoker=declarative_tool_invoker,
            execution_identity=self._execution_identity,
            authority_policy=authority_policy,
            budget_allocation_policy=budget_allocation_policy,
        )
        self._composer = FinalResponseComposer(merge_strategy=merge_strategy)
        self._lifecycle = lifecycle
        self._trace_emitter = trace_emitter
        self._trace_store = trace_store
        self._current_task: Optional[Task] = None
        self._signal_collector = signal_collector
        self._evaluation_registry = evaluation_registry
        self._run_budget = run_budget
        self._execution_budget_ledger_factory = (
            execution_budget_ledger_factory
            or create_execution_budget_ledger_factory(run_budget)
        )
        self._attempt_lifecycle = attempt_lifecycle or AttemptLifecycleService(
            InMemoryAttemptLifecycleStore(),
        )
        self._execution_terminal = execution_terminal or ExecutionTerminalService(
            wire_execution_terminal_store(checkpoint_store=self._checkpoint_store),
        )
        validate_durable_attempt_lifecycle_for_composition(
            production_mode=production_mode,
            store=self._attempt_lifecycle.store,
            agent_retry_max=resolved_retry_policy.max_retries,
            run_retry_max=max_run_retries,
        )
        validate_durable_execution_terminal_for_composition(
            production_mode=production_mode,
            checkpoint_store=self._checkpoint_store,
            store=self._execution_terminal.store,
        )
        self._production_mode = production_mode
        trace_reader = trace_store if isinstance(trace_store, RunTraceReader) else None
        self._events = NexusRuntimeEventPublisher(
            self._event_bus,
            current_task=lambda: self._current_task,
            execution_identity=self._execution_identity,
            trace_reader=trace_reader,
            runtime_event_store=self._runtime_event_store,
        )
        self._hitl = NexusHitlRunner(
            publish=self._publish_runtime_event,
            human_hooks=self._human_hooks,
            lifecycle_hooks=self._lifecycle_hooks,
            escalation_router=self._escalation_router,
            notification_adapter=self._notification_adapter,
            finish_task=self._finish_task,
            finalize_trace=self._finalize_persisting_trace,
            maybe_checkpoint=self._maybe_checkpoint_long_running,
            persist_human_decision=self._persist_human_decision,
            execution_identity=self._execution_identity,
        )
        self._graph_runner = NexusGraphRunner(
            registry=self._registry,
            graph_executor=self._graph_executor,
            validation_engine=self._validation_engine,
            composer=self._composer,
            hitl=self._hitl,
            events=self._events,
            finish_task=self._finish_task,
            finalize_trace=self._finalize_persisting_trace,
            maybe_checkpoint=self._maybe_checkpoint_long_running,
            attempt_lifecycle=self._attempt_lifecycle,
            execution_terminal=self._execution_terminal,
            max_run_retries=max_run_retries,
            production_mode=production_mode,
            decision_flow_gate=decision_flow_gate,
        )
        self._intake_runner = NexusIntakeRunner(
            hitl=self._hitl,
            human_hooks=self._human_hooks,
            publish=self._publish_runtime_event,
            restore_long_running=self._maybe_restore_long_running,
            execution_identity=self._execution_identity,
        )
        self._planning_runner = NexusPlanningRunner(
            classifier=self._classifier,
            planner=self._planner,
            registry=self._registry,
            hitl=self._hitl,
            publish=self._publish_runtime_event,
            finish_task=self._finish_task,
            maybe_checkpoint=self._maybe_checkpoint_long_running,
            policy_engine=self._policy_engine,
            emit_coordination_advisory=emit_coordination_advisory,
            denied_planner_model_ids=denied_planner_model_ids,
            planner_model_id=planner_model_id,
            execution_identity=self._execution_identity,
        )

    @property
    def registry(self) -> AgentRegistryRead:
        return self._registry

    @property
    def agent_engine(self) -> AgentEngine:
        return self._engine

    def apply_validation_engine(self, validation_engine: NexusValidationEngine) -> None:
        """Replace the active validation engine across Nexus execution surfaces."""
        self._validation_engine = validation_engine
        self._graph_executor.apply_validation_engine(validation_engine)
        self._graph_runner.validation_engine = validation_engine

    def apply_decision_flow_gate(
        self,
        gate: Optional["DecisionFlowGate[AgentExecutionResult]"],
        *,
        verify_uaep_step: bool = False,
    ) -> None:
        """Attach Decision flow authority to graph and UAEP execution surfaces."""
        self._graph_executor.apply_decision_flow_gate(gate)
        self._graph_runner.decision_flow_gate = gate
        self._engine.uaep_executor.set_decision_flow_gate(
            gate,
            verify_uaep_step=verify_uaep_step,
        )

    def peek_decision_flow_gate(
        self,
    ) -> Optional["DecisionFlowGate[AgentExecutionResult]"]:
        """Return wired Decision flow gate when application decision wiring is active."""
        return self._graph_executor.peek_decision_flow_gate()

    @property
    def trace_emitter(self) -> Optional[TaskTraceEmitter]:
        return self._trace_emitter

    @property
    def trace_store(self) -> Optional[RunTraceWriter]:
        return self._trace_store

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

    @property
    def run_budget(self) -> RunBudget | None:
        return self._run_budget

    @property
    def execution_budget_ledger_factory(self) -> "ExecutionBudgetLedgerFactory":
        return self._execution_budget_ledger_factory

    @property
    def execution_terminal(self) -> ExecutionTerminalService:
        return self._execution_terminal

    async def handle_task(
        self,
        task: Task,
        *,
        run_id: RunId,
        attempt_id: Optional[AttemptId] = None,
    ) -> TaskResult:
        resolved_run_id = validate_run_id(run_id)
        active_execution_id = require_active_execution_id()
        active_run_id, active_attempt_id = require_active_execution_identity()
        resolved_attempt_id = (
            validate_attempt_id(attempt_id)
            if attempt_id is not None
            else active_attempt_id
        )
        if resolved_run_id != active_run_id:
            raise RuntimeError(
                "active execution identity run_id mismatch with Nexus handle_task",
            )
        if resolved_attempt_id != active_attempt_id:
            raise RuntimeError(
                "active execution identity attempt_id mismatch with Nexus handle_task",
            )
        require_active_execution_authority()
        active_budget = require_active_execution_budget()
        if active_budget.execution_id != active_execution_id:
            raise RuntimeError(
                "active execution budget execution_id mismatch with Nexus handle_task",
            )
        self._current_task = task
        try:
            return await self._handle_task_impl(task)
        finally:
            self._current_task = None

    async def _handle_task_impl(self, task: Task) -> TaskResult:
        lifecycle, trace_emitter = self._resolve_lifecycle(task)
        self._trace_emitter = trace_emitter

        intake = await self._intake_runner.run(
            task,
            lifecycle=lifecycle,
            trace_emitter=trace_emitter,
        )
        if intake.early_result is not None:
            return intake.early_result

        planning = await self._planning_runner.run(
            task,
            lifecycle=lifecycle,
            trace_emitter=trace_emitter,
        )
        if planning.early_result is not None:
            return planning.early_result
        plan = planning.plan
        if plan is None:
            raise RuntimeError("planning phase completed without plan or early result")

        graph = plan_to_execution_graph(plan)
        task.runtime.orchestration.graph_id = graph.graph_id
        task.sync_metadata()

        phase = await self._graph_runner.run(
            task,
            plan=plan,
            graph=graph,
            lifecycle=lifecycle,
            trace_emitter=trace_emitter,
        )
        if phase.early_result is not None:
            return phase.early_result
        assert phase.executions is not None
        assert phase.retry_records is not None
        assert phase.graph is not None
        assert phase.plan is not None
        assert phase.final_validation is not None

        return await self._finish_task(
            task,
            trace_emitter,
            answer=self._composer.compose_summary(phase.executions),
            executions=phase.executions,
            validation=phase.final_validation,
            plan=phase.plan,
            retry_records=phase.retry_records,
            graph_id=phase.graph.graph_id,
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
                extra=_finalization_hook_extra(task, answer=answer),
            )
        except NexusLifecycleHookError as exc:
            await publish_nexus_lifecycle_hook_failure(
                self._publish_runtime_event,
                task=task,
                point=HookPoint.BEFORE_FINALIZATION,
                phase=ExecutionPhase.COMPLETION,
                error=exc,
            )
            resolution = self._commit_durable_terminal_authority(task)
            if resolution.should_publish_terminal_event:
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

        from intergrax.runtime.policy.pre_output_policy_bridge import apply_pre_output_policy

        answer, _pre_output_decision = apply_pre_output_policy(
            self._policy_engine, task, answer=answer
        )

        resolution = self._commit_durable_terminal_authority(task)
        if resolution.should_publish_terminal_event:
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
                extra=_finalization_hook_extra(task, answer=answer),
            )
        except NexusLifecycleHookError as exc:
            await publish_nexus_lifecycle_hook_failure(
                self._publish_runtime_event,
                task=task,
                point=HookPoint.AFTER_FINALIZATION,
                phase=ExecutionPhase.COMPLETION,
                error=exc,
                non_critical=True,
            )
        self._maybe_record_adaptive_outcome_signal(task, result)
        self._maybe_record_multi_agent_evaluation(executions, task_id=task.task_id)
        from intergrax.runtime.governance.post_run_governance_bridge import (
            invoke_post_run_governance,
        )

        active_run_id, _ = require_active_execution_identity()
        invoke_post_run_governance(
            self._governance_service,
            run_id=active_run_id,
            agent_id=task.agent_id or "",
        )
        return result

    def _maybe_record_multi_agent_evaluation(
        self,
        executions: List[AgentExecutionResult],
        *,
        task_id: str,
    ) -> None:
        if self._evaluation_registry is None or len(executions) < 2:
            return
        active_run_id, _ = require_active_execution_identity()
        passed = all(item.status == AgentExecutionStatus.COMPLETED for item in executions)
        self._evaluation_registry.append(
            OnlineEvaluationObservation(
                observation_id=f"obs_{task_id}_multi_agent",
                run_id=active_run_id,
                agent_id=",".join(item.agent_id for item in executions if item.agent_id),
                mode=OnlineEvaluationMode.SHADOW,
                scenario_id="multi_agent_fan_in",
                passed=passed,
                score=1.0 if passed else 0.0,
            )
        )

    def _maybe_record_adaptive_outcome_signal(self, task: Task, result: TaskResult) -> None:
        if self._signal_collector is None:
            return
        record_task_outcome_signal(
            self._signal_collector,
            task=task,
            result=result,
            evaluation_registry=self._evaluation_registry,
            run_budget=self._run_budget,
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
        result = build_nexus_task_result(
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
            run_id=require_active_execution_identity()[0],
        )
        return result

    async def _maybe_restore_long_running(self, task: Task) -> None:
        active_run_id, _ = require_active_execution_identity()
        await maybe_restore_long_running(
            task,
            checkpoint_store=self._checkpoint_store,
            publish=self._publish_runtime_event,
            notification_adapter=self._notification_adapter,
            run_id=active_run_id,
            execution_terminal=self._execution_terminal,
        )

    async def _maybe_checkpoint_long_running(
        self,
        task: Task,
        *,
        progress_message: str,
        plan: Optional[NexusPlan] = None,
        graph: Optional[ExecutionGraph] = None,
        last_execution: Optional[AgentExecutionResult] = None,
    ) -> None:
        run_id, attempt_id = self._execution_identity.require()
        await maybe_checkpoint_long_running(
            task,
            checkpoint_store=self._checkpoint_store,
            publish=self._publish_runtime_event,
            notification_adapter=self._notification_adapter,
            progress_message=progress_message,
            run_id=run_id,
            attempt_id=attempt_id,
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
            await self._events.publish(event, task=task)

    def attach_terminal_diagnostic_trigger(
        self,
        trigger: TerminalExecutionDiagnosticTriggerProtocol,
    ) -> None:
        """Attach platform terminal diagnostic trigger after host composition."""
        self._terminal_diagnostic_trigger = trigger

    def _commit_durable_terminal_authority(self, task: Task) -> TerminalCommitResolution:
        """Persist terminal outcome and return canonical durable authority."""
        outcome = terminal_outcome_from_task_state(task.state)
        if outcome is None:
            return TerminalCommitResolution(
                canonical_record=None,
                should_publish_terminal_event=True,
            )
        run_id, _ = require_active_execution_identity()
        try:
            record = self._execution_terminal.commit_terminal_outcome(
                tenant_id=task.tenant_id,
                task_id=task.task_id,
                run_id=run_id,
                outcome=outcome,
                reason=terminal_reason_for_task_state(task.state),
                production_mode=self._production_mode,
            )
            return TerminalCommitResolution(
                canonical_record=record,
                should_publish_terminal_event=True,
            )
        except ExecutionTerminalConflictError:
            canonical = self._execution_terminal.get_terminal_record(
                tenant_id=task.tenant_id,
                task_id=task.task_id,
            )
            if canonical is None:
                raise ExecutionTerminalError(
                    "execution terminal conflict without canonical record",
                ) from None
            validate_terminal_run_id_consistency(canonical, run_id)
            task.state = reconcile_task_state_with_terminal_outcome(
                task.state,
                canonical.outcome,
            )
            return TerminalCommitResolution(
                canonical_record=canonical,
                should_publish_terminal_event=False,
            )

    async def _publish_terminal_runtime_event(self, task: Task) -> None:
        await self._publish_terminal_runtime_event_with_active_identity(task)

    async def publish_host_task_terminal_runtime(
        self,
        task: Task,
        *,
        run_id: RunId,
        attempt_id: AttemptId,
        execution_id: ExecutionId,
    ) -> RuntimeEvent:
        """Publish terminal RuntimeEvent truth for canonical host task execution."""
        from intergrax.contracts.execution_identity import (
            bind_active_execution_identity,
            reset_active_execution_identity,
        )
        from intergrax.runtime.events.trace_bridge import runtime_event_from_task_state

        token = bind_active_execution_identity(
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
        )
        try:
            operational_completed = runtime_event_from_task_state(
                task,
                run_id=run_id,
                attempt_id=attempt_id,
                message="task state -> completed",
            )
            await self._events.publish(operational_completed, task=task)
            return await self._publish_terminal_runtime_event_with_active_identity(task)
        finally:
            reset_active_execution_identity(token)

    async def _publish_terminal_runtime_event_with_active_identity(
        self,
        task: Task,
    ) -> RuntimeEvent:
        terminal_event = await self._events.publish_terminal(task)
        if self._terminal_diagnostic_trigger is not None:
            from intergrax.runtime.diagnostics.terminal_execution_diagnostic_bridge import (
                invoke_terminal_execution_diagnostics,
            )

            from intergrax.runtime.execution.boundary import ExecutionIdentityBinding

            invoke_terminal_execution_diagnostics(
                self._terminal_diagnostic_trigger,
                tenant_id=task.tenant_id,
                task_id=task.task_id,
                run_id=terminal_event.run_id,
                observed_at=terminal_event.timestamp,
                event_bus=self._event_bus,
                execution_identity=ExecutionIdentityBinding(
                    run_id=terminal_event.run_id,
                    attempt_id=terminal_event.attempt_id,
                    execution_id=terminal_event.execution_id,
                ),
            )
        return terminal_event

    def _resolve_lifecycle(self, task: Task) -> tuple[TaskLifecycle, TaskTraceEmitter]:
        return resolve_nexus_lifecycle(
            task,
            execution_identity=self._execution_identity,
            lifecycle=self._lifecycle,
            trace_emitter=self._trace_emitter,
            trace_store=self._trace_store,
            event_bus=self._event_bus,
        )

    async def _finalize_persisting_trace(
        self,
        trace_emitter: PersistingTaskTraceEmitter,
        executions: List[AgentExecutionResult],
        *,
        task_id: str = "",
    ) -> None:
        await finalize_persisting_trace(
            trace_emitter,
            executions,
            task_id=task_id,
            middleware=self._middleware,
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


def _finalization_hook_extra(task: Task, *, answer: str) -> dict[str, str]:
    return {
        "task_state": task.state.value,
        "prompt": task.message,
        "llm_output": answer,
        "output": answer,
        "tenant_id": task.tenant_id,
    }
