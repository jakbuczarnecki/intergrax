# © Artur Czarnecki. All rights reserved.

"""Project admitted UER child execution into provider-neutral context."""

from __future__ import annotations

from intergrax.contracts.delegated_execution_provider import (
    DelegatedExecutionBudgetBounds,
    DelegatedExecutionBudgetMode,
    DelegatedExecutionBudgetProjection,
    DelegatedExecutionContext,
)
from intergrax.contracts.delegation_authority import (
    EffectiveDelegationAuthority,
    ParentExecutionAuthority,
)
from intergrax.contracts.execution_identity import ExecutionId, RunId, AttemptId
from intergrax.runtime.execution.boundary import ExecutionIdentityBinding
from intergrax.runtime.execution.budget.models import ExecutionBudgetAllocationMode
from intergrax.runtime.execution.budget.models import ExecutionBudgetReservationGrant
from intergrax.runtime.nexus.budget.budget_models import RunBudget


def _map_budget_bounds(budget: RunBudget | None) -> DelegatedExecutionBudgetBounds | None:
    if budget is None:
        return None
    return DelegatedExecutionBudgetBounds(
        max_input_tokens=budget.max_input_tokens,
        max_output_tokens=budget.max_output_tokens,
        max_total_tokens=budget.max_total_tokens,
        max_llm_calls=budget.max_llm_calls,
        max_tool_calls=budget.max_tool_calls,
        max_rag_invocations=budget.max_rag_invocations,
        max_websearch_invocations=budget.max_websearch_invocations,
        max_wall_time_seconds=budget.max_wall_time_seconds,
        max_planner_iterations=budget.max_planner_iterations,
        max_replans=budget.max_replans,
    )


def _map_allocation_mode(
    mode: ExecutionBudgetAllocationMode,
) -> DelegatedExecutionBudgetMode:
    if mode is ExecutionBudgetAllocationMode.SHARED:
        return DelegatedExecutionBudgetMode.SHARED
    return DelegatedExecutionBudgetMode.RESERVED


def project_delegated_execution_context(
    *,
    identity: ExecutionIdentityBinding,
    authority: ParentExecutionAuthority,
    effective_delegation: EffectiveDelegationAuthority | None,
    budget_grant: ExecutionBudgetReservationGrant,
    correlation_id: str | None = None,
) -> DelegatedExecutionContext:
    """Build readonly provider context from UER-admitted child execution state."""
    if identity.parent_execution_id is None:
        raise ValueError("parent_execution_id required for delegated child context")
    return DelegatedExecutionContext(
        execution_id=identity.execution_id,
        parent_execution_id=identity.parent_execution_id,
        run_id=identity.run_id,
        attempt_id=identity.attempt_id,
        authority=authority,
        effective_delegation=effective_delegation,
        budget=DelegatedExecutionBudgetProjection(
            allocation_mode=_map_allocation_mode(budget_grant.mode),
            reservation_allowance=_map_budget_bounds(budget_grant.reservation_allowance),
        ),
        correlation_id=correlation_id,
    )


def project_delegated_execution_context_from_parts(
    *,
    execution_id: ExecutionId,
    parent_execution_id: ExecutionId,
    run_id: RunId,
    attempt_id: AttemptId,
    authority: ParentExecutionAuthority,
    effective_delegation: EffectiveDelegationAuthority | None,
    budget_grant: ExecutionBudgetReservationGrant,
    correlation_id: str | None = None,
) -> DelegatedExecutionContext:
    """Convenience projection when an ``ExecutionIdentityBinding`` is unavailable."""
    return project_delegated_execution_context(
        identity=ExecutionIdentityBinding(
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
            parent_execution_id=parent_execution_id,
        ),
        authority=authority,
        effective_delegation=effective_delegation,
        budget_grant=budget_grant,
        correlation_id=correlation_id,
    )
