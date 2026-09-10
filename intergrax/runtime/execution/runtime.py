# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical root execution lifecycle (UE-10R1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
)
from intergrax.contracts.execution_lineage import ExecutionLineagePersistence
from intergrax.runtime.execution.active_decision_checkpoint_persistence import (
    bind_active_decision_checkpoint_persistence,
    reset_active_decision_checkpoint_persistence,
)
from intergrax.runtime.execution.active_decision_finalization_persistence import (
    bind_active_decision_finalization_persistence,
    reset_active_decision_finalization_persistence,
)
from intergrax.runtime.execution.active_decision_lifecycle_host import (
    bind_active_decision_lifecycle_host,
    reset_active_decision_lifecycle_host,
)
from intergrax.runtime.execution.active_execution_work_port import (
    ActiveExecutionWorkPortBinding,
    bind_active_execution_work_port,
    reset_active_execution_work_port,
)
from intergrax.runtime.execution.decision_checkpoint_persistence import (
    DecisionCheckpointPersistence,
)
from intergrax.runtime.execution.decision_finalization_persistence import (
    DecisionFinalizationPersistence,
)
from intergrax.runtime.execution.active_execution_budget import (
    bind_root_execution_budget,
    reset_active_execution_budget,
)
from intergrax.runtime.execution.boundary import (
    ExecutionAdmissionHook,
    ExecutionBoundary,
    ExecutionDelegate,
    ExecutionIdentityBinding,
)
from intergrax.runtime.execution.budget.ledger import (
    ExecutionBudgetLedgerFactory,
    RunBudgetExecutionBudgetLedgerFactory,
)
from intergrax.runtime.execution.decision_lifecycle_host import DecisionLifecycleHost
from intergrax.runtime.execution.lineage.root_activation import (
    activate_root_execution_lineage,
    build_root_lineage_admission_hook,
    deactivate_root_execution_lineage,
    merge_lineage_root_admission_hooks,
    validate_root_lineage_inputs,
)
from intergrax.runtime.execution.identity_authority import (
    BackgroundTransportIdentity,
    RootTaskIdentity,
    mint_background_transport_identity,
    mint_child_execution_id,
    mint_retry_attempt_id,
    mint_root_execution_identity,
)
from intergrax.runtime.nexus.budget.budget_models import RunBudget

RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")
CheckpointPayloadT = TypeVar("CheckpointPayloadT")
WorkInputT = TypeVar("WorkInputT")
WorkOutputT = TypeVar("WorkOutputT")
WorkResultT = TypeVar("WorkResultT")


@dataclass(frozen=True, slots=True)
class RootExecutionContext:
    """Typed root lifecycle inputs bound before strategy routing."""

    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId
    authority: ParentExecutionAuthority
    tenant_id: str | None = None
    task_id: TaskId | None = None
    segment_predecessor_root_execution_id: ExecutionId | None = None


@dataclass(frozen=True, slots=True)
class RootExecutionOptions:
    """Optional inputs for resolving a root execution context."""

    authority: ParentExecutionAuthority
    run_id: RunId | None = None
    attempt_id: AttemptId | None = None
    execution_id: ExecutionId | None = None
    tenant_id: str | None = None
    task_id: TaskId | None = None
    segment_predecessor_root_execution_id: ExecutionId | None = None


def resolve_root_execution_context(options: RootExecutionOptions) -> RootExecutionContext:
    """Resolve typed root context; mints RunId and AttemptId when omitted."""
    identity = mint_root_execution_identity(
        run_id=options.run_id,
        attempt_id=options.attempt_id,
        execution_id=options.execution_id,
    )
    return RootExecutionContext(
        run_id=identity.run_id,
        attempt_id=identity.attempt_id,
        execution_id=identity.execution_id,
        authority=options.authority,
        tenant_id=options.tenant_id,
        task_id=options.task_id,
        segment_predecessor_root_execution_id=options.segment_predecessor_root_execution_id,
    )


class ExecutionRuntime(Generic[RequestT, ResultT]):
    """
    Canonical root execution lifecycle owner (UE-10R1).

    Resolves root identity, binds authority and budget, routes through
    :class:`ExecutionBoundary` and :class:`StrategyExecutionRouter`.
    """

    __slots__ = (
        "_delegate",
        "_ledger_factory",
        "_run_budget",
        "_admission_hooks",
        "_decision_lifecycle_host",
        "_decision_checkpoint_persistence",
        "_decision_finalization_persistence",
        "_execution_work_port_binding",
        "_execution_lineage_persistence",
    )

    def __init__(
        self,
        delegate: ExecutionDelegate[RequestT, ResultT],
        *,
        ledger_factory: ExecutionBudgetLedgerFactory | None = None,
        run_budget: RunBudget | None = None,
        admission_hooks: tuple[ExecutionAdmissionHook[RequestT], ...] = (),
        execution_lineage_persistence: ExecutionLineagePersistence | None = None,
        decision_lifecycle_host: DecisionLifecycleHost | None = None,
        decision_checkpoint_persistence: (
            DecisionCheckpointPersistence[CheckpointPayloadT] | None
        ) = None,
        decision_finalization_persistence: (
            DecisionFinalizationPersistence[CheckpointPayloadT] | None
        ) = None,
        execution_work_port_binding: (
            ActiveExecutionWorkPortBinding[WorkInputT, WorkOutputT, WorkResultT] | None
        ) = None,
    ) -> None:
        self._delegate = delegate
        self._ledger_factory = (
            ledger_factory
            if ledger_factory is not None
            else RunBudgetExecutionBudgetLedgerFactory(default_run_budget=run_budget)
        )
        self._run_budget = run_budget
        self._admission_hooks = admission_hooks
        self._decision_lifecycle_host = decision_lifecycle_host
        self._decision_checkpoint_persistence = decision_checkpoint_persistence
        self._decision_finalization_persistence = decision_finalization_persistence
        self._execution_work_port_binding = execution_work_port_binding
        self._execution_lineage_persistence = execution_lineage_persistence

    async def execute(
        self,
        request: RequestT,
        root_context: RootExecutionContext,
    ) -> ResultT:
        execution_id = root_context.execution_id
        ledger = self._ledger_factory.create_ledger(
            self._run_budget,
            tenant_id=root_context.tenant_id,
            run_id=root_context.run_id,
            attempt_id=root_context.attempt_id,
        )
        binding = ExecutionIdentityBinding(
            run_id=root_context.run_id,
            attempt_id=root_context.attempt_id,
            execution_id=execution_id,
        )
        admission_hooks = self._admission_hooks
        lineage_token = None
        degradation_token = None
        if self._execution_lineage_persistence is not None:
            lineage_scope = validate_root_lineage_inputs(
                tenant_id=root_context.tenant_id,
                task_id=root_context.task_id,
                run_id=root_context.run_id,
                attempt_id=root_context.attempt_id,
                execution_id=execution_id,
            )
            _, lineage_token, degradation_token = activate_root_execution_lineage(
                persistence=self._execution_lineage_persistence,
                scope=lineage_scope,
                root_execution_id=execution_id,
                predecessor_root_execution_id=root_context.segment_predecessor_root_execution_id,
            )
            lineage_hook = build_root_lineage_admission_hook(
                persistence=self._execution_lineage_persistence,
                scope=lineage_scope,
                segment_root_execution_id=execution_id,
                execution_id=execution_id,
            )
            admission_hooks = merge_lineage_root_admission_hooks(
                lineage_hook,
                self._admission_hooks,
            )
        boundary = ExecutionBoundary[RequestT, ResultT](
            self._delegate,
            admission_hooks=admission_hooks,
            identity=binding,
            authority=root_context.authority,
        )
        budget_token = bind_root_execution_budget(
            execution_id=execution_id,
            ledger=ledger,
            run_budget=self._run_budget,
        )
        host_token = None
        persistence_token = None
        finalization_token = None
        work_port_token = None
        try:
            if self._decision_lifecycle_host is not None:
                host_token = bind_active_decision_lifecycle_host(
                    self._decision_lifecycle_host,
                )
            if self._decision_checkpoint_persistence is not None:
                persistence_token = bind_active_decision_checkpoint_persistence(
                    self._decision_checkpoint_persistence,
                )
            if self._decision_finalization_persistence is not None:
                finalization_token = bind_active_decision_finalization_persistence(
                    self._decision_finalization_persistence,
                )
            if self._execution_work_port_binding is not None:
                work_port_token = bind_active_execution_work_port(
                    self._execution_work_port_binding,
                )
            return await boundary.execute(request)
        finally:
            if lineage_token is not None and degradation_token is not None:
                deactivate_root_execution_lineage(lineage_token, degradation_token)
            if work_port_token is not None:
                reset_active_execution_work_port(work_port_token)
            if persistence_token is not None:
                reset_active_decision_checkpoint_persistence(persistence_token)
            if finalization_token is not None:
                reset_active_decision_finalization_persistence(finalization_token)
            if host_token is not None:
                reset_active_decision_lifecycle_host(host_token)
            reset_active_execution_budget(budget_token)
