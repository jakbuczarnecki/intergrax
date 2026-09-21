# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Nexus-backed host task execution composition (runtime tier)."""

from __future__ import annotations

from collections.abc import Callable

from intergrax.agents.persistence.skill_host_wiring import HostSkillCatalogWiring
from intergrax.contracts.admitted_root_governance_identity import AdmittedRootGovernanceIdentity
from intergrax.contracts.runtime_execution_admission import RootExecutionAuthorityAdmissionPort
from intergrax.runtime.execution.effective_profile_revision_admission import (
    EffectiveProfileRevisionAdmissionPort,
)
from intergrax.runtime.execution.failure_evidence.runtime_event_recorder import (
    RuntimeEventExecutionFailureEvidenceRecorder,
)
from intergrax.runtime.execution.budget.persistence import RunBudgetPersistence
from intergrax.runtime.execution.deadline_authority import ExecutionDeadlineAuthorityResolver
from intergrax.runtime.execution.host_task import HostTaskExecution
from intergrax.runtime.execution.nexus_host_task_terminal import (
    build_nexus_host_task_terminal_publisher,
)
from intergrax.runtime.execution.orchestration import OrchestrationExecutor
from intergrax.runtime.nexus.agent_router import AgentRouter
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.task.task import Task
from intergrax.contracts.execution_continuation_state_store import (
    ExecutionContinuationStateStore,
)


def build_host_task_execution(
    nexus_loop: NexusLoop,
    *,
    orchestration_triggers: frozenset[str],
    root_authority_admission: RootExecutionAuthorityAdmissionPort,
    admit_root_governance_identity: Callable[[Task], AdmittedRootGovernanceIdentity],
    pipeline_capability_suffix: str = ".pipeline",
    revision_admission: EffectiveProfileRevisionAdmissionPort | None = None,
    skill_host_wiring: HostSkillCatalogWiring | None = None,
    run_budget_persistence: RunBudgetPersistence | None = None,
    deadline_authority_resolver: ExecutionDeadlineAuthorityResolver | None = None,
    continuation_state_store: ExecutionContinuationStateStore | None = None,
) -> HostTaskExecution:
    """Internal composition builder: extract canonical execution dependencies from Nexus."""
    resolved_continuation_store = continuation_state_store
    if resolved_continuation_store is None:
        resolved_continuation_store = nexus_loop.execution_continuation_state_store
    return HostTaskExecution(
        _agent_engine=nexus_loop.agent_engine,
        _agent_router=AgentRouter(
            nexus_loop.registry,
            event_bus=nexus_loop.event_bus,
        ),
        _orchestration_executor=OrchestrationExecutor(nexus_loop),
        _orchestration_triggers=orchestration_triggers,
        _pipeline_capability_suffix=pipeline_capability_suffix,
        _ledger_factory=nexus_loop.execution_budget_ledger_factory,
        _run_budget=nexus_loop.run_budget,
        _terminal_publisher=build_nexus_host_task_terminal_publisher(nexus_loop),
        _revision_admission=revision_admission,
        _execution_lineage_persistence=nexus_loop.execution_lineage_persistence,
        _failure_evidence_recorder=RuntimeEventExecutionFailureEvidenceRecorder(
            nexus_loop.event_bus,
        ),
        _root_authority_admission=root_authority_admission,
        _continuation_state_store=resolved_continuation_store,
        _declarative_tool_invoker=nexus_loop.declarative_tool_invoker,
        _skill_host_wiring=skill_host_wiring,
        _admit_root_governance_identity=admit_root_governance_identity,
        _run_budget_persistence=run_budget_persistence,
        _deadline_authority_resolver=deadline_authority_resolver,
    )


__all__ = ["build_host_task_execution", "build_nexus_host_task_terminal_publisher"]
