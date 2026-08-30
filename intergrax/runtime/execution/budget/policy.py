# © Artur Czarnecki. All rights reserved.

"""Execution budget allocation policy contract and platform default (UE-8B1)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.runtime.execution.budget.models import (
    ChildBudgetAllocationContext,
    ChildBudgetAllocationDecision,
    ExecutionBudgetAllocationMode,
)


@runtime_checkable
class ExecutionBudgetAllocationPolicy(Protocol):
    """Pluggable strategy for child execution budget allocation."""

    def resolve_child_budget(
        self,
        context: ChildBudgetAllocationContext,
    ) -> ChildBudgetAllocationDecision:
        """Resolve budget allocation intent for a child under ``context``."""


class DefaultSharedPoolBudgetPolicy:
    """
    Platform default: shared Run pool unless an explicit child budget is requested.

    No explicit request participates in the shared pool without reserving the
  remaining Run capacity. An explicit request asks the ledger for a bounded
    exclusive reservation backed by canonical accounting.
    """

    def resolve_child_budget(
        self,
        context: ChildBudgetAllocationContext,
    ) -> ChildBudgetAllocationDecision:
        if context.requested_budget is None:
            return ChildBudgetAllocationDecision(
                mode=ExecutionBudgetAllocationMode.SHARED,
            )
        return ChildBudgetAllocationDecision(
            mode=ExecutionBudgetAllocationMode.RESERVED,
            reservation_request=context.requested_budget,
        )
