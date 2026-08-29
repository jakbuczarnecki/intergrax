# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical child execution lineage (UE-7A)."""

from __future__ import annotations

from typing import Generic, TypeVar

from intergrax.contracts.execution_identity import (
    mint_execution_id,
    require_active_execution_id,
    require_active_execution_identity,
)
from intergrax.runtime.execution.active_execution_budget import (
    ActiveExecutionBudgetState,
    bind_active_execution_budget,
    peek_active_execution_budget,
    reset_active_execution_budget,
)
from intergrax.runtime.execution.authority.policy import (
    ChildAuthorityContext,
    DefaultStrictAuthorityPolicy,
    ExecutionAuthorityPolicy,
)
from intergrax.runtime.execution.boundary import (
    ExecutionAdmissionHook,
    ExecutionBoundary,
    ExecutionDelegate,
    ExecutionIdentityBinding,
)
from intergrax.runtime.execution.budget.ledger import ExecutionBudgetLedger
from intergrax.runtime.execution.budget.models import (
    ChildBudgetAllocationContext,
    ExecutionBudgetAllocationMode,
)
from intergrax.runtime.execution.budget.policy import (
    DefaultSharedPoolBudgetPolicy,
    ExecutionBudgetAllocationPolicy,
)
from intergrax.runtime.governance.active_execution_authority import (
    require_active_execution_authority,
)
from intergrax.runtime.nexus.budget.budget_models import RunBudget

RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")


class ChildExecutionRunner(Generic[RequestT, ResultT]):
    """
    Mint a child ExecutionId under the active parent Execution and route work
    through :class:`ExecutionBoundary`.
    """

    __slots__ = ("_authority_policy", "_budget_policy", "_ledger")

    def __init__(
        self,
        authority_policy: ExecutionAuthorityPolicy | None = None,
        budget_policy: ExecutionBudgetAllocationPolicy | None = None,
        ledger: ExecutionBudgetLedger | None = None,
    ) -> None:
        self._authority_policy = (
            authority_policy
            if authority_policy is not None
            else DefaultStrictAuthorityPolicy()
        )
        self._budget_policy = (
            budget_policy
            if budget_policy is not None
            else DefaultSharedPoolBudgetPolicy()
        )
        self._ledger = ledger

    async def execute(
        self,
        *,
        request: RequestT,
        delegate: ExecutionDelegate[RequestT, ResultT],
        admission_hooks: tuple[ExecutionAdmissionHook[RequestT], ...] = (),
        requested_permission_scopes: tuple[str, ...] | None = None,
        requested_budget: RunBudget | None = None,
    ) -> ResultT:
        parent_run_id, parent_attempt_id = require_active_execution_identity()
        parent_execution_id = require_active_execution_id()
        parent_authority = require_active_execution_authority()

        resolution = self._authority_policy.resolve_child_authority(
            ChildAuthorityContext(
                parent_authority=parent_authority,
                requested_permission_scopes=requested_permission_scopes,
            )
        )
        child_authority = resolution.authority
        effective = resolution.effective_delegation

        parent_budget_state = peek_active_execution_budget()
        if parent_budget_state is None:
            parent_mode = ExecutionBudgetAllocationMode.SHARED
            parent_remaining = None
        else:
            parent_mode = parent_budget_state.mode
            parent_remaining = None
            if parent_mode is ExecutionBudgetAllocationMode.RESERVED:
                parent_remaining = parent_budget_state.ledger.snapshot_reservation_remaining(
                    parent_execution_id,
                )

        budget_decision = self._budget_policy.resolve_child_budget(
            ChildBudgetAllocationContext(
                parent_execution_id=parent_execution_id,
                parent_allocation_mode=parent_mode,
                parent_reservation_remaining=parent_remaining,
                requested_budget=requested_budget,
            )
        )

        child_execution_id = mint_execution_id()
        if self._ledger is None:
            raise RuntimeError("execution budget ledger required for child execution")

        grant = self._ledger.grant_child_budget(
            execution_id=child_execution_id,
            parent_execution_id=parent_execution_id,
            decision=budget_decision,
        )
        active_budget = ActiveExecutionBudgetState(
            execution_id=child_execution_id,
            mode=grant.mode,
            ledger=self._ledger,
            reservation_allowance=grant.reservation_allowance,
        )

        identity = ExecutionIdentityBinding(
            run_id=parent_run_id,
            attempt_id=parent_attempt_id,
            execution_id=child_execution_id,
            parent_execution_id=parent_execution_id,
        )
        boundary = ExecutionBoundary[RequestT, ResultT](
            delegate,
            admission_hooks=admission_hooks,
            identity=identity,
            authority=child_authority,
            effective_delegation=effective,
        )
        budget_token = bind_active_execution_budget(active_budget)
        try:
            return await boundary.execute(request)
        finally:
            reset_active_execution_budget(budget_token)
            self._ledger.release_child_budget(child_execution_id)
