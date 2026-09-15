# © Artur Czarnecki. All rights reserved.

"""Production adoption path for DelegatedExecutionProvider via child execution (P2.1-S2A)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, runtime_checkable

from pydantic import ValidationError

from intergrax.contracts.delegated_execution_invocation_binding import (
    assert_provider_outcome_has_no_invocation_binding,
    delegated_provider_outcome_contract_mismatch_failure,
    enrich_delegated_outcome_with_platform_invocation_binding,
)
from intergrax.contracts.delegated_execution_provider import (
    DelegatedExecutionContractError,
    DelegatedExecutionOperationMetadata,
    DelegatedExecutionOutcome,
    DelegatedExecutionProvider,
    DelegatedExecutionRequest,
    digest_delegated_execution_payload,
)
from intergrax.contracts.execution_identity import (
    require_active_execution_id,
    require_active_execution_identity,
    peek_active_parent_execution_id,
)
from intergrax.runtime.execution.authority.policy import (
    DefaultStrictAuthorityPolicy,
    ExecutionAuthorityPolicy,
)
from intergrax.runtime.execution.boundary import (
    ExecutionAdmissionHook,
    ExecutionDelegate,
    ExecutionIdentityBinding,
)
from intergrax.runtime.execution.budget.ledger import ExecutionBudgetLedger
from intergrax.runtime.execution.budget.models import ExecutionBudgetReservationGrant
from intergrax.runtime.execution.budget.policy import (
    DefaultSharedPoolBudgetPolicy,
    ExecutionBudgetAllocationPolicy,
)
from intergrax.runtime.execution.child import ChildExecutionRunner
from intergrax.runtime.execution.delegated_execution.context_projection import (
    project_delegated_execution_context,
)
from intergrax.runtime.execution.active_execution_budget import require_active_execution_budget
from intergrax.runtime.governance.active_execution_authority import (
    peek_active_effective_delegation,
    require_active_execution_authority,
)
from intergrax.runtime.nexus.budget.budget_models import RunBudget

RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True, slots=True)
class DelegatedExecutionWorkUnit(Generic[RequestT]):
    """Typed child-work carrier admitted before provider dispatch."""

    payload: RequestT
    operation: DelegatedExecutionOperationMetadata


@runtime_checkable
class DelegatedExecutionPort(Protocol[RequestT, ResultT]):
    """Platform port for delegated provider work through canonical child execution."""

    async def execute_delegated(
        self,
        *,
        payload: RequestT,
        operation: DelegatedExecutionOperationMetadata,
        requested_permission_scopes: tuple[str, ...] | None = None,
        requested_budget: RunBudget | None = None,
        admission_hooks: tuple[
            ExecutionAdmissionHook[DelegatedExecutionWorkUnit[RequestT]],
            ...,
        ] = (),
    ) -> DelegatedExecutionOutcome[ResultT]:
        ...


class _DelegatedProviderDispatchDelegate(
    Generic[RequestT, ResultT],
):
    """Routes admitted child execution state into provider-neutral dispatch."""

    __slots__ = ("_provider",)

    def __init__(
        self,
        provider: DelegatedExecutionProvider[RequestT, ResultT],
    ) -> None:
        self._provider = provider

    async def execute(
        self,
        work: DelegatedExecutionWorkUnit[RequestT],
    ) -> DelegatedExecutionOutcome[ResultT]:
        run_id, attempt_id = require_active_execution_identity()
        execution_id = require_active_execution_id()
        parent_execution_id = peek_active_parent_execution_id()
        if parent_execution_id is None:
            raise RuntimeError(
                "parent_execution_id required for delegated provider dispatch",
            )
        identity = ExecutionIdentityBinding(
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
            parent_execution_id=parent_execution_id,
        )
        budget_state = require_active_execution_budget()
        grant = ExecutionBudgetReservationGrant(
            execution_id=budget_state.execution_id,
            parent_execution_id=parent_execution_id,
            mode=budget_state.mode,
            reservation_allowance=budget_state.reservation_allowance,
        )
        context = project_delegated_execution_context(
            identity=identity,
            authority=require_active_execution_authority(),
            effective_delegation=peek_active_effective_delegation(),
            budget_grant=grant,
            correlation_id=work.operation.correlation_id,
        )
        provider_request = DelegatedExecutionRequest(
            context=context,
            payload=work.payload,
            operation=work.operation,
        )
        outcome = await self._provider.execute(provider_request)
        try:
            assert_provider_outcome_has_no_invocation_binding(outcome)
        except DelegatedExecutionContractError:
            return delegated_provider_outcome_contract_mismatch_failure(
                failure_message=(
                    "provider outcome contains platform-owned invocation binding"
                ),
            )
        payload_digest = digest_delegated_execution_payload(work.payload)
        try:
            return enrich_delegated_outcome_with_platform_invocation_binding(
                outcome=outcome,
                context=context,
                operation=work.operation,
                payload_digest=payload_digest,
            )
        except (ValidationError, DelegatedExecutionContractError, ValueError):
            return delegated_provider_outcome_contract_mismatch_failure(
                failure_message=(
                    "provider outcome failed platform invocation correlation checks"
                ),
            )


class DelegatedExecutionService(Generic[RequestT, ResultT]):
    """
    Execution-owned adoption service: child admission precedes provider dispatch.

    Depends on :class:`DelegatedExecutionProvider` at composition root — never on
    a concrete provider implementation.
    """

    __slots__ = ("_child_runner", "_dispatch_delegate")

    def __init__(
        self,
        provider: DelegatedExecutionProvider[RequestT, ResultT],
        *,
        ledger: ExecutionBudgetLedger | None = None,
        authority_policy: ExecutionAuthorityPolicy | None = None,
        budget_policy: ExecutionBudgetAllocationPolicy | None = None,
    ) -> None:
        self._dispatch_delegate = _DelegatedProviderDispatchDelegate(provider)
        self._child_runner = ChildExecutionRunner[
            DelegatedExecutionWorkUnit[RequestT],
            DelegatedExecutionOutcome[ResultT],
        ](
            authority_policy=(
                authority_policy
                if authority_policy is not None
                else DefaultStrictAuthorityPolicy()
            ),
            budget_policy=(
                budget_policy
                if budget_policy is not None
                else DefaultSharedPoolBudgetPolicy()
            ),
            ledger=ledger,
        )

    async def execute_delegated(
        self,
        *,
        payload: RequestT,
        operation: DelegatedExecutionOperationMetadata,
        requested_permission_scopes: tuple[str, ...] | None = None,
        requested_budget: RunBudget | None = None,
        admission_hooks: tuple[
            ExecutionAdmissionHook[DelegatedExecutionWorkUnit[RequestT]],
            ...,
        ] = (),
    ) -> DelegatedExecutionOutcome[ResultT]:
        work = DelegatedExecutionWorkUnit(payload=payload, operation=operation)
        return await self._child_runner.execute(
            request=work,
            delegate=self._dispatch_delegate,
            admission_hooks=admission_hooks,
            requested_permission_scopes=requested_permission_scopes,
            requested_budget=requested_budget,
        )


def delegated_execution_service(
    provider: DelegatedExecutionProvider[RequestT, ResultT],
    *,
    ledger: ExecutionBudgetLedger | None = None,
    authority_policy: ExecutionAuthorityPolicy | None = None,
    budget_policy: ExecutionBudgetAllocationPolicy | None = None,
) -> DelegatedExecutionService[RequestT, ResultT]:
    """Composition helper for the canonical delegated execution adoption path."""
    return DelegatedExecutionService(
        provider,
        ledger=ledger,
        authority_policy=authority_policy,
        budget_policy=budget_policy,
    )


__all__ = [
    "DelegatedExecutionPort",
    "DelegatedExecutionService",
    "DelegatedExecutionWorkUnit",
    "delegated_execution_service",
]
