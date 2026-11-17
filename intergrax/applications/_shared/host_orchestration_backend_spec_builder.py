# © Artur Czarnecki. All rights reserved.

"""Tier-3 host orchestration backend spec builder (resolves the environment profile).

Resolves LLM adapters, planner, classifier, context manager and application wiring from
``ApplicationEnvironmentProfile`` and hands neutral values to the Execution Engine builder
``build_host_orchestration_loop_init_spec`` (Tier-1), which never reads the profile.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from intergrax.agents.persistence.checkpoint_store import AgentCheckpointStore
from intergrax.agents.persistence.compensation_queue_store import CompensationQueueStore
from intergrax.applications._shared.adaptive_wiring import ApplicationAdaptiveWiring
from intergrax.applications._shared.application_host_orchestration_wiring import (
    build_application_host_orchestration_wiring,
)
from intergrax.applications._shared.decision_wiring import ApplicationDecisionWiring
from intergrax.applications._shared.guardrail_wiring import ApplicationGuardrailWiring
from intergrax.applications._shared.host_execution_capacity_policy import (
    validate_strict_host_execution_capacity,
)
from intergrax.applications._shared.host_orchestration_environment_resolution import (
    host_context_composition_spec,
    host_orchestration_environment,
    host_orchestration_planning_spec,
)
from intergrax.applications._shared.llm_resolver import resolve_environment_llm_adapter
from intergrax.applications._shared.orchestration_wiring import (
    orchestration_requires_llm_adapter,
)
from intergrax.applications._shared.reasoning_wiring import resolve_planner_llm_adapter
from intergrax.applications._shared.security_wiring import ApplicationSecurityWiring
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
)
from intergrax.context.protocols import ContextEngine
from intergrax.contracts.attempt_lifecycle import AttemptLifecycleStore
from intergrax.contracts.child_execution_context_inheritance import (
    ChildExecutionContextInheritancePort,
)
from intergrax.contracts.execution_bound_declarative_tool_invocation import (
    ExecutionBoundDeclarativeToolInvoker,
)
from intergrax.contracts.host_orchestration_wiring_port import (
    HostOrchestrationApplicationWiringPort,
)
from intergrax.contracts.idempotency_store import IdempotencyStore
from intergrax.contracts.run_budget import RunBudget
from intergrax.contracts.run_trace_store import RunTraceWriter
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.integrations.contracts.document_store import DocumentStore
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.execution.application_environment_context_composition import (
    ContextManager,
    resolve_context_manager_from_environment,
)
from intergrax.runtime.execution.execution_terminal import ExecutionTerminalService
from intergrax.runtime.execution.host_orchestration_environment_spec_builder import (
    build_host_orchestration_loop_init_spec,
)
from intergrax.runtime.execution.host_orchestration_loop_init_spec import (
    HostOrchestrationLoopInitSpec,
)
from intergrax.runtime.execution.host_orchestration_planner_classifier_wiring import (
    resolve_nexus_task_classifier,
    resolve_nexus_task_planner,
)
from intergrax.runtime.execution.host_runtime_config import RuntimeConfig
from intergrax.runtime.execution.host_validation_composition import NexusValidationEngine
from intergrax.runtime.governance.governance_evidence_recorder import (
    GovernanceEvidenceRecorder,
)
from intergrax.runtime.long_running.notification import NotificationAdapter
from intergrax.runtime.long_running.persistence_contract import (
    TaskCheckpointPersistence,
)
from intergrax.runtime.registry.agent_registry_read import AgentRegistryRead
from intergrax.runtime.sandbox.manager import SandboxSessionManager
from intergrax.runtime.task_memory.persistence_contract import TaskMemoryPersistence
from intergrax.runtime.workspace.manager import ShadowWorkspaceManager

if TYPE_CHECKING:
    from intergrax.contracts.execution_continuation_state_store import (
        ExecutionContinuationStateStore,
    )
    from intergrax.contracts.execution_lineage import ExecutionLineagePersistence
    from intergrax.contracts.execution_terminal import ExecutionTerminalStore
    from intergrax.runtime.execution.authority.policy import ExecutionAuthorityPolicy
    from intergrax.runtime.execution.budget.ledger import (
        ExecutionBudgetLedger,
        ExecutionBudgetLedgerFactory,
    )
    from intergrax.runtime.execution.budget.policy import (
        ExecutionBudgetAllocationPolicy,
    )


def build_host_orchestration_loop_init_spec_from_environment(
    registry: AgentRegistryRead,
    *,
    env: ApplicationEnvironmentProfile,
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
    llm_adapter: LLMAdapter | None = None,
    runtime_event_bus: RuntimeEventBus | None = None,
    context_manager: ContextManager | None = None,
    context_engine: ContextEngine | None = None,
    security_wiring: ApplicationSecurityWiring | None = None,
    guardrail_wiring: ApplicationGuardrailWiring | None = None,
    decision_wiring: ApplicationDecisionWiring | None = None,
    adaptive_wiring: ApplicationAdaptiveWiring | None = None,
    application_wiring: HostOrchestrationApplicationWiringPort | None = None,
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
    """Resolve typed Execution Engine orchestration backend inputs from the environment."""
    validate_strict_host_execution_capacity(env)

    producer_llm = llm_adapter
    if producer_llm is None and orchestration_requires_llm_adapter(env):
        producer_llm = resolve_environment_llm_adapter(env)
    planner_llm = resolve_planner_llm_adapter(env, producer_adapter=producer_llm) if producer_llm is not None else None
    planning_spec = host_orchestration_planning_spec(
        env,
        llm_adapter=producer_llm,
        planner_llm_adapter=planner_llm,
    )
    planner = resolve_nexus_task_planner(planning_spec)
    classifier = resolve_nexus_task_classifier(registry, planning_spec)
    resolved_context_manager = context_manager or resolve_context_manager_from_environment(
        host_context_composition_spec(env),
        event_bus=runtime_event_bus,
        llm_adapter=producer_llm,
        context_engine=context_engine,
    )
    resolved_wiring = application_wiring or build_application_host_orchestration_wiring(
        env,
        security_wiring=security_wiring,
        guardrail_wiring=guardrail_wiring,
        decision_wiring=decision_wiring,
    )

    return build_host_orchestration_loop_init_spec(
        environment=host_orchestration_environment(env),
        planner=planner,
        classifier=classifier,
        context_manager=resolved_context_manager,
        application_wiring=resolved_wiring,
        decision_flow_gate=decision_wiring.gate if decision_wiring else None,
        signal_collector=adaptive_wiring.signal_collector if adaptive_wiring else None,
        trace_store=trace_store,
        checkpoint_store=checkpoint_store,
        agent_checkpoint_store=agent_checkpoint_store,
        compensation_queue_store=compensation_queue_store,
        idempotency_store=idempotency_store,
        declarative_tool_invoker=declarative_tool_invoker,
        notification_adapter=notification_adapter,
        runtime_events_db_path=runtime_events_db_path,
        task_memory_store=task_memory_store,
        task_memory_db_path=task_memory_db_path,
        shadow_manager=shadow_manager,
        sandbox_manager=sandbox_manager,
        runtime_event_bus=runtime_event_bus,
        run_budget=run_budget,
        validation_engine=validation_engine,
        runtime_config=runtime_config,
        authority_policy=authority_policy,
        budget_allocation_policy=budget_allocation_policy,
        execution_budget_ledger_factory=execution_budget_ledger_factory,
        execution_budget_ledger=execution_budget_ledger,
        attempt_lifecycle_store=attempt_lifecycle_store,
        key_value_cache=key_value_cache,
        document_store=document_store,
        execution_terminal=execution_terminal,
        execution_terminal_store=execution_terminal_store,
        execution_lineage_persistence=execution_lineage_persistence,
        execution_continuation_state_store=execution_continuation_state_store,
        governance_evidence_recorder=governance_evidence_recorder,
        child_context_inheritance=child_context_inheritance,
    )


__all__ = ["build_host_orchestration_loop_init_spec_from_environment"]
