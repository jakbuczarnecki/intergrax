# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Worker-side host task execution materialization (Execution Engine owner zone)."""

from __future__ import annotations

from collections.abc import Callable

from intergrax.contracts.admitted_root_governance_identity import (
    AdmittedRootGovernanceIdentity,
)
from intergrax.contracts.execution_continuation_state_store import (
    ExecutionContinuationStateStore,
)
from intergrax.contracts.runtime_execution_admission import (
    RootExecutionAuthorityAdmissionPort,
)
from intergrax.runtime.execution.budget.ledger import ExecutionBudgetLedgerFactory
from intergrax.runtime.execution.budget.persistence import (
    RunBudgetPersistence,
    create_durable_run_budget_ledger_factory,
)
from intergrax.runtime.execution.deadline_authority import ExecutionDeadlineAuthorityResolver
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.execution.host_orchestration_loop_init_spec import (
    HostOrchestrationLoopInitSpec,
)
from intergrax.runtime.execution.environment_orchestration_materialization import (
    materialize_host_orchestration_backend,
)
from intergrax.runtime.execution._orchestration_backend_access import (
    orchestration_backend_for_execution_engine,
)
from intergrax.runtime.execution.execution_terminal import ExecutionTerminalService
from intergrax.runtime.execution.nexus_host_execution import build_host_task_execution
from intergrax.runtime.long_running.persistence_contract import TaskCheckpointPersistence
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.contracts.host_orchestration_run_retry import HostOrchestrationRunRetrySpec
from intergrax.runtime.nexus.retry.retry_engine import RetryPolicy
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task


def _nexus_retry_policy(spec: HostOrchestrationRunRetrySpec) -> RetryPolicy:
    return RetryPolicy(
        max_retries=spec.max_retries,
        retry_alternate_agent=spec.retry_alternate_agent,
    )


def build_worker_host_task_execution_from_registry(
    registry: AgentRegistry,
    *,
    checkpoint_store: TaskCheckpointPersistence | None = None,
    run_budget: RunBudget | None = None,
    run_budget_persistence: RunBudgetPersistence | None = None,
    deadline_authority_resolver: ExecutionDeadlineAuthorityResolver | None = None,
    execution_budget_ledger_factory: ExecutionBudgetLedgerFactory | None = None,
    execution_terminal: ExecutionTerminalService | None = None,
    orchestration_triggers: frozenset[str] = frozenset(),
    pipeline_capability_suffix: str = ".pipeline",
    execution_continuation_state_store: ExecutionContinuationStateStore | None = None,
    production_mode: bool,
    admit_root_governance_identity: Callable[[Task], AdmittedRootGovernanceIdentity],
    root_authority_admission: RootExecutionAuthorityAdmissionPort,
    retry_policy: HostOrchestrationRunRetrySpec | None = None,
) -> HostTaskExecutionPort:
    """Materialize governed worker host execution without exposing Nexus to runtime/task."""
    resolved_factory = execution_budget_ledger_factory
    if resolved_factory is None and run_budget_persistence is not None:
        resolved_factory = create_durable_run_budget_ledger_factory(
            run_budget_persistence,
            run_budget,
        )
    if run_budget_persistence is not None and deadline_authority_resolver is None:
        raise ValueError(
            "durable run_budget_persistence requires deadline_authority_resolver",
        )
    spec = HostOrchestrationLoopInitSpec(
        checkpoint_store=checkpoint_store,
        run_budget=run_budget,
        execution_budget_ledger_factory=resolved_factory,
        execution_terminal=execution_terminal,
        execution_continuation_state_store=execution_continuation_state_store,
        production_mode=production_mode,
        max_run_retries=0,
        retry_policy=_nexus_retry_policy(
            retry_policy or HostOrchestrationRunRetrySpec(max_retries=0),
        ),
    )
    materialization = materialize_host_orchestration_backend(registry, spec)
    return build_host_task_execution(
        orchestration_backend_for_execution_engine(materialization),
        orchestration_triggers=orchestration_triggers,
        pipeline_capability_suffix=pipeline_capability_suffix,
        root_authority_admission=root_authority_admission,
        admit_root_governance_identity=admit_root_governance_identity,
        run_budget_persistence=run_budget_persistence,
        deadline_authority_resolver=deadline_authority_resolver,
    )


__all__ = ["build_worker_host_task_execution_from_registry"]
