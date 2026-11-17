# © Artur Czarnecki. All rights reserved.

"""Build the host orchestration backend construction spec from resolved neutral values.

Tier-3 resolves the environment profile into :class:`HostOrchestrationEnvironment`, the
planner, the classifier, the context manager and the application wiring; this module keeps
what is Execution Engine owned (retry, authority, budget, attempt lifecycle, lineage and
terminal persistence, durable checkpoint gating). It never imports Tier-3 modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from intergrax.agents.persistence.checkpoint_store import AgentCheckpointStore
from intergrax.agents.persistence.compensation_queue_store import CompensationQueueStore
from intergrax.contracts.attempt_lifecycle import AttemptLifecycleStore
from intergrax.contracts.child_execution_context_inheritance import (
    ChildExecutionContextInheritancePort,
)
from intergrax.contracts.execution_bound_declarative_tool_invocation import (
    ExecutionBoundDeclarativeToolInvoker,
)
from intergrax.contracts.execution_mode import ExecutionMode
from intergrax.contracts.host_orchestration_environment import HostOrchestrationEnvironment
from intergrax.contracts.host_orchestration_wiring_port import (
    HostOrchestrationApplicationWiringPort,
)
from intergrax.contracts.idempotency_store import IdempotencyStore
from intergrax.contracts.run_budget import RunBudget
from intergrax.contracts.run_trace_store import RunTraceWriter
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.integrations.contracts.document_store import DocumentStore
from intergrax.runtime.adaptive.signal_collector import SignalCollector
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.execution.attempt_lifecycle import (
    AttemptLifecycleService,
    resolve_attempt_lifecycle_store,
)
from intergrax.runtime.execution.authority import (
    resolve_execution_authority_policy_from_runtime_config,
)
from intergrax.runtime.execution.budget import (
    create_execution_budget_ledger_factory,
    fixed_execution_budget_ledger_factory,
    resolve_execution_budget_allocation_policy_from_runtime_config,
)
from intergrax.runtime.execution.execution_terminal import ExecutionTerminalService
from intergrax.runtime.execution.execution_terminal.wiring import (
    resolve_execution_terminal_store,
)
from intergrax.runtime.execution.host_orchestration_loop_init_spec import (
    HostOrchestrationLoopInitSpec,
)
from intergrax.runtime.execution.host_runtime_config import RuntimeConfig
from intergrax.runtime.execution.lineage.wiring import (
    resolve_execution_lineage_persistence,
)
from intergrax.runtime.governance.governance_evidence_recorder import (
    GovernanceEvidenceRecorder,
)
from intergrax.runtime.long_running.notification import NotificationAdapter
from intergrax.runtime.long_running.persistence_contract import (
    TaskCheckpointPersistence,
)
from intergrax.runtime.nexus.context.context_manager import ContextManager
from intergrax.runtime.nexus.planning.nexus_planner_protocol import NexusTaskPlannerProtocol
from intergrax.runtime.nexus.retry.retry_engine import RetryPolicy
from intergrax.runtime.nexus.task_classifier_protocol import NexusTaskClassifierProtocol
from intergrax.runtime.nexus.validation.validation_engine import NexusValidationEngine
from intergrax.runtime.sandbox.manager import SandboxSessionManager
from intergrax.runtime.task_memory.persistence_contract import TaskMemoryPersistence
from intergrax.runtime.workspace.manager import ShadowWorkspaceManager

if TYPE_CHECKING:
    from intergrax.contracts.agent_execution_result import AgentExecutionResult
    from intergrax.contracts.execution_continuation_state_store import (
        ExecutionContinuationStateStore,
    )
    from intergrax.contracts.execution_lineage import ExecutionLineagePersistence
    from intergrax.contracts.execution_terminal import ExecutionTerminalStore
    from intergrax.runtime.decision_flow import DecisionFlowGate
    from intergrax.runtime.execution.authority.policy import ExecutionAuthorityPolicy
    from intergrax.runtime.execution.budget.ledger import (
        ExecutionBudgetLedger,
        ExecutionBudgetLedgerFactory,
    )
    from intergrax.runtime.execution.budget.policy import (
        ExecutionBudgetAllocationPolicy,
    )


def build_host_orchestration_loop_init_spec(
    *,
    environment: HostOrchestrationEnvironment,
    planner: NexusTaskPlannerProtocol,
    classifier: NexusTaskClassifierProtocol,
    context_manager: ContextManager,
    application_wiring: HostOrchestrationApplicationWiringPort | None,
    decision_flow_gate: DecisionFlowGate[AgentExecutionResult] | None,
    signal_collector: SignalCollector | None,
    trace_store: RunTraceWriter | None = None,
    checkpoint_store: TaskCheckpointPersistence | None = None,
    agent_checkpoint_store: AgentCheckpointStore | None = None,
    compensation_queue_store: CompensationQueueStore | None = None,
    idempotency_store: IdempotencyStore | None = None,
    declarative_tool_invoker: ExecutionBoundDeclarativeToolInvoker | None = None,
    notification_adapter: NotificationAdapter | None = None,
    runtime_events_db_path: Path | None = None,
    task_memory_store: TaskMemoryPersistence | None = None,
    task_memory_db_path: Path | None = None,
    shadow_manager: ShadowWorkspaceManager | None = None,
    sandbox_manager: SandboxSessionManager | None = None,
    runtime_event_bus: RuntimeEventBus | None = None,
    run_budget: RunBudget | None = None,
    validation_engine: NexusValidationEngine | None = None,
    runtime_config: RuntimeConfig | None = None,
    authority_policy: ExecutionAuthorityPolicy | None = None,
    budget_allocation_policy: ExecutionBudgetAllocationPolicy | None = None,
    execution_budget_ledger_factory: ExecutionBudgetLedgerFactory | None = None,
    execution_budget_ledger: ExecutionBudgetLedger | None = None,
    attempt_lifecycle_store: AttemptLifecycleStore | None = None,
    key_value_cache: DistributedKVStore | None = None,
    document_store: DocumentStore | None = None,
    execution_terminal: ExecutionTerminalService | None = None,
    execution_terminal_store: ExecutionTerminalStore | None = None,
    execution_lineage_persistence: ExecutionLineagePersistence | None = None,
    execution_continuation_state_store: ExecutionContinuationStateStore | None = None,
    governance_evidence_recorder: GovernanceEvidenceRecorder | None = None,
    child_context_inheritance: ChildExecutionContextInheritancePort | None = None,
) -> HostOrchestrationLoopInitSpec:
    """Resolve typed Execution Engine orchestration backend inputs from neutral values."""
    retry_policy = RetryPolicy(max_retries=3)
    if environment.retry_policy_name == "strict":
        retry_policy = RetryPolicy(max_retries=1)

    resolved_authority_policy = authority_policy
    if resolved_authority_policy is None and runtime_config is not None:
        resolved_authority_policy = resolve_execution_authority_policy_from_runtime_config(
            runtime_config,
        )
    resolved_budget_policy = budget_allocation_policy
    if resolved_budget_policy is None and runtime_config is not None:
        resolved_budget_policy = resolve_execution_budget_allocation_policy_from_runtime_config(
            runtime_config,
        )
    resolved_budget_ledger_factory = execution_budget_ledger_factory
    if resolved_budget_ledger_factory is None:
        if execution_budget_ledger is not None:
            resolved_budget_ledger_factory = fixed_execution_budget_ledger_factory(
                execution_budget_ledger,
            )
        else:
            resolved_budget_ledger_factory = create_execution_budget_ledger_factory(run_budget)
    kv_store = key_value_cache
    doc_store = document_store
    durable_checkpoint_store = (
        checkpoint_store if checkpoint_store is not None and environment.long_running_scheduler_enabled else None
    )
    resolved_attempt_lifecycle_store = resolve_attempt_lifecycle_store(
        provider=environment.attempt_lifecycle_persistence_provider,
        kv_store=kv_store,
        document_store=doc_store,
        explicit_store=attempt_lifecycle_store,
    )
    resolved_attempt_lifecycle = (
        AttemptLifecycleService(resolved_attempt_lifecycle_store)
        if resolved_attempt_lifecycle_store is not None
        else None
    )
    resolved_execution_lineage = resolve_execution_lineage_persistence(
        explicit_persistence=execution_lineage_persistence,
        document_store=doc_store,
        provider=environment.execution_lineage_persistence_provider,
    )
    resolved_execution_terminal = execution_terminal
    if resolved_execution_terminal is None and (
        execution_terminal_store is not None
        or kv_store is not None
        or doc_store is not None
        or durable_checkpoint_store is not None
    ):
        resolved_execution_terminal = ExecutionTerminalService(
            resolve_execution_terminal_store(
                provider=environment.execution_terminal_persistence_provider,
                kv_store=kv_store,
                document_store=doc_store,
                checkpoint_store=durable_checkpoint_store,
                execution_terminal_store=execution_terminal_store,
            ),
        )

    return HostOrchestrationLoopInitSpec(
        classifier=classifier,
        planner=planner,
        max_parallel_nodes=environment.max_parallel_nodes,
        max_inflight_nodes=environment.max_inflight_nodes,
        max_delegation_depth=environment.max_delegation_depth,
        max_run_retries=environment.max_run_retries,
        merge_strategy=environment.merge_strategy,
        context_manager=context_manager,
        event_bus=runtime_event_bus,
        trace_store=trace_store,
        retry_policy=retry_policy,
        shadow_manager=shadow_manager,
        sandbox_manager=sandbox_manager,
        checkpoint_store=checkpoint_store if environment.long_running_scheduler_enabled else None,
        agent_checkpoint_store=agent_checkpoint_store,
        compensation_queue_store=compensation_queue_store,
        idempotency_store=idempotency_store,
        declarative_tool_invoker=declarative_tool_invoker,
        notification_adapter=notification_adapter,
        runtime_events_db_path=runtime_events_db_path,
        task_memory_store=task_memory_store,
        task_memory_db_path=task_memory_db_path,
        production_mode=environment.execution_mode is ExecutionMode.STRICT,
        signal_collector=signal_collector,
        run_budget=run_budget,
        decision_flow_gate=decision_flow_gate,
        emit_coordination_advisory=environment.emit_coordination_advisory,
        allow_dynamic_replan=environment.allow_dynamic_replan,
        denied_planner_model_ids=environment.denied_planner_model_ids,
        planner_model_id=environment.planner_model_id,
        validation_engine=validation_engine,
        authority_policy=resolved_authority_policy,
        budget_allocation_policy=resolved_budget_policy,
        execution_budget_ledger_factory=resolved_budget_ledger_factory,
        attempt_lifecycle=resolved_attempt_lifecycle,
        execution_terminal=resolved_execution_terminal,
        execution_lineage_persistence=resolved_execution_lineage,
        execution_continuation_state_store=execution_continuation_state_store,
        governance_evidence_recorder=governance_evidence_recorder,
        application_wiring=application_wiring,
        child_context_inheritance=child_context_inheritance,
    )


__all__ = ["build_host_orchestration_loop_init_spec"]
