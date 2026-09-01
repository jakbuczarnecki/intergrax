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
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
)
from intergrax.runtime.execution.active_decision_checkpoint_persistence import (
    bind_active_decision_checkpoint_persistence,
    reset_active_decision_checkpoint_persistence,
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
from intergrax.runtime.nexus.budget.budget_models import RunBudget

RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")
CheckpointPayloadT = TypeVar("CheckpointPayloadT")
WorkInputT = TypeVar("WorkInputT")
WorkOutputT = TypeVar("WorkOutputT")
WorkResultT = TypeVar("WorkResultT")


@dataclass(frozen=True, slots=True)
class RootTaskIdentity:
    """Resolved root Run and Attempt identifiers plus minted root ExecutionId."""

    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId


@dataclass(frozen=True, slots=True)
class RootExecutionContext:
    """Typed root lifecycle inputs bound before strategy routing."""

    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId
    authority: ParentExecutionAuthority
    tenant_id: str | None = None


@dataclass(frozen=True, slots=True)
class RootExecutionOptions:
    """Optional inputs for resolving a root execution context."""

    authority: ParentExecutionAuthority
    run_id: RunId | None = None
    attempt_id: AttemptId | None = None
    tenant_id: str | None = None


def mint_root_execution_identity(
    *,
    run_id: RunId | None = None,
    attempt_id: AttemptId | None = None,
) -> RootTaskIdentity:
    """Mint canonical generic root identity for one root execution invocation."""
    return RootTaskIdentity(
        run_id=run_id or mint_run_id(),
        attempt_id=attempt_id or mint_attempt_id(),
        execution_id=mint_execution_id(),
    )


def resolve_root_execution_context(options: RootExecutionOptions) -> RootExecutionContext:
    """Resolve typed root context; mints RunId and AttemptId when omitted."""
    identity = mint_root_execution_identity(
        run_id=options.run_id,
        attempt_id=options.attempt_id,
    )
    return RootExecutionContext(
        run_id=identity.run_id,
        attempt_id=identity.attempt_id,
        execution_id=identity.execution_id,
        authority=options.authority,
        tenant_id=options.tenant_id,
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
        "_execution_work_port_binding",
    )

    def __init__(
        self,
        delegate: ExecutionDelegate[RequestT, ResultT],
        *,
        ledger_factory: ExecutionBudgetLedgerFactory | None = None,
        run_budget: RunBudget | None = None,
        admission_hooks: tuple[ExecutionAdmissionHook[RequestT], ...] = (),
        decision_lifecycle_host: DecisionLifecycleHost | None = None,
        decision_checkpoint_persistence: (
            DecisionCheckpointPersistence[CheckpointPayloadT] | None
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
        self._execution_work_port_binding = execution_work_port_binding

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
        boundary = ExecutionBoundary[RequestT, ResultT](
            self._delegate,
            admission_hooks=self._admission_hooks,
            identity=binding,
            authority=root_context.authority,
        )
        budget_token = bind_root_execution_budget(
            execution_id=execution_id,
            ledger=ledger,
        )
        host_token = None
        persistence_token = None
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
            if self._execution_work_port_binding is not None:
                work_port_token = bind_active_execution_work_port(
                    self._execution_work_port_binding,
                )
            return await boundary.execute(request)
        finally:
            if work_port_token is not None:
                reset_active_execution_work_port(work_port_token)
            if persistence_token is not None:
                reset_active_decision_checkpoint_persistence(persistence_token)
            if host_token is not None:
                reset_active_decision_lifecycle_host(host_token)
            reset_active_execution_budget(budget_token)
