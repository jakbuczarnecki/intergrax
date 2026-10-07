# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed NexusLoop construction inputs — Execution Engine owner zone only."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from intergrax.agents.persistence.checkpoint_store import AgentCheckpointStore
from intergrax.agents.persistence.compensation_queue_store import CompensationQueueStore
from intergrax.contracts.child_execution_context_inheritance import (
    ChildExecutionContextInheritancePort,
)
from intergrax.runtime.execution.host_orchestration_wiring_bundle import (
    HostOrchestrationApplicationWiringBundle,
)
from intergrax.contracts.execution_continuation_state_store import (
    ExecutionContinuationStateStore,
)
from intergrax.contracts.idempotency_store import IdempotencyStore
from intergrax.contracts.execution_bound_declarative_tool_invocation import (
    ExecutionBoundDeclarativeToolInvoker,
)
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.long_running.notification import NotificationAdapter
from intergrax.runtime.long_running.persistence_contract import TaskCheckpointPersistence
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.nexus.context.context_manager import ContextManager
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.retry.retry_engine import RetryPolicy
from intergrax.runtime.nexus.tracing.persistence_models import RunTraceWriter
from intergrax.runtime.adaptive.signal_collector import SignalCollector
from intergrax.runtime.nexus.validation.validation_engine import NexusValidationEngine
from intergrax.runtime.registry.agent_registry_read import AgentRegistryRead
from intergrax.runtime.sandbox.manager import SandboxSessionManager
from intergrax.runtime.task_memory.persistence_contract import TaskMemoryPersistence
from intergrax.runtime.workspace.manager import ShadowWorkspaceManager

if TYPE_CHECKING:
    from intergrax.runtime.decision_flow import DecisionFlowGate
    from intergrax.contracts.agent_execution_result import AgentExecutionResult
    from intergrax.runtime.execution.attempt_lifecycle import AttemptLifecycleService
    from intergrax.runtime.execution.authority.policy import ExecutionAuthorityPolicy
    from intergrax.runtime.execution.budget.ledger import ExecutionBudgetLedgerFactory
    from intergrax.runtime.execution.budget.policy import ExecutionBudgetAllocationPolicy
    from intergrax.runtime.execution.execution_terminal import ExecutionTerminalService
    from intergrax.contracts.execution_lineage import ExecutionLineagePersistence
    from intergrax.runtime.governance.governance_evidence_recorder import (
        GovernanceEvidenceRecorder,
    )
    from intergrax.runtime.nexus.planning.nexus_planner_protocol import (
        NexusTaskPlannerProtocol,
    )
    from intergrax.runtime.nexus.task_classifier_protocol import (
        NexusTaskClassifierProtocol,
    )
    from intergrax.contracts.orchestration_enums import MergeStrategy


@dataclass(frozen=True, slots=True)
class HostOrchestrationLoopInitSpec:
    """Keyword bundle for canonical NexusLoop materialization inside Execution Engine."""

    classifier: NexusTaskClassifierProtocol | None = None
    planner: NexusTaskPlannerProtocol | None = None
    max_parallel_nodes: int | None = None
    max_inflight_nodes: int | None = None
    max_delegation_depth: int | None = None
    max_run_retries: int = 0
    merge_strategy: MergeStrategy | None = None
    context_manager: ContextManager | None = None
    trace_store: RunTraceWriter | None = None
    retry_policy: RetryPolicy | None = None
    shadow_manager: ShadowWorkspaceManager | None = None
    sandbox_manager: SandboxSessionManager | None = None
    checkpoint_store: TaskCheckpointPersistence | None = None
    agent_checkpoint_store: AgentCheckpointStore | None = None
    compensation_queue_store: CompensationQueueStore | None = None
    idempotency_store: IdempotencyStore | None = None
    declarative_tool_invoker: ExecutionBoundDeclarativeToolInvoker | None = None
    notification_adapter: NotificationAdapter | None = None
    runtime_events_db_path: Path | None = None
    task_memory_store: TaskMemoryPersistence | None = None
    task_memory_db_path: Path | None = None
    production_mode: bool = False
    signal_collector: SignalCollector | None = None
    run_budget: RunBudget | None = None
    decision_flow_gate: DecisionFlowGate[AgentExecutionResult] | None = None
    emit_coordination_advisory: bool = False
    allow_dynamic_replan: bool = False
    denied_planner_model_ids: tuple[str, ...] = ()
    planner_model_id: str | None = None
    validation_engine: NexusValidationEngine | None = None
    authority_policy: ExecutionAuthorityPolicy | None = None
    budget_allocation_policy: ExecutionBudgetAllocationPolicy | None = None
    execution_budget_ledger_factory: ExecutionBudgetLedgerFactory | None = None
    attempt_lifecycle: AttemptLifecycleService | None = None
    execution_terminal: ExecutionTerminalService | None = None
    execution_lineage_persistence: ExecutionLineagePersistence | None = None
    execution_continuation_state_store: ExecutionContinuationStateStore | None = None
    governance_evidence_recorder: GovernanceEvidenceRecorder | None = None
    event_bus: RuntimeEventBus | None = None
    application_wiring: HostOrchestrationApplicationWiringBundle | None = None
    child_context_inheritance: ChildExecutionContextInheritancePort | None = None


__all__ = [
    "HostOrchestrationLoopInitSpec",
]
