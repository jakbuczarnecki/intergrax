# © Artur Czarnecki. All rights reserved.

"""Neutral execution budget ledger ports for host orchestration wiring."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from intergrax.contracts.budget_usage_totals import BudgetUsageTotals
from intergrax.contracts.execution_identity import AttemptId, ExecutionId, RunId
from intergrax.contracts.run_budget import RunBudget


class ExecutionBudgetAllocationMode(Enum):
    """How a child execution participates in hierarchical budget accounting."""

    SHARED = "shared"
    RESERVED = "reserved"


@dataclass(frozen=True, slots=True)
class ChildBudgetAllocationDecision:
    """Policy intent for child budget participation (validated by the ledger)."""

    mode: ExecutionBudgetAllocationMode
    reservation_request: RunBudget | None = None


@dataclass(frozen=True, slots=True)
class ExecutionBudgetReservationGrant:
    """Ledger-validated grant for a child execution."""

    execution_id: ExecutionId
    parent_execution_id: ExecutionId
    mode: ExecutionBudgetAllocationMode
    reservation_allowance: RunBudget | None


class ExecutionBudgetLedgerPort(Protocol):
    """Canonical hierarchical budget accounting and enforcement."""

    def grant_child_budget(
        self,
        *,
        execution_id: ExecutionId,
        parent_execution_id: ExecutionId,
        decision: ChildBudgetAllocationDecision,
    ) -> ExecutionBudgetReservationGrant:
        """Validate and record child budget participation."""

    def release_child_budget(self, execution_id: ExecutionId) -> None:
        """Release unused exclusive reservation for ``execution_id``."""

    def ensure_shared_participant(
        self,
        execution_id: ExecutionId,
        *,
        parent_execution_id: ExecutionId,
    ) -> None:
        """Register a shared participant when absent (idempotent)."""

    def consume_budget(
        self,
        execution_id: ExecutionId,
        amounts: BudgetUsageTotals,
    ) -> None:
        """Consume budget against the effective grant for ``execution_id``."""

    def snapshot_root_available(self) -> RunBudget:
        """Return remaining capacity at the Run root pool."""


class ExecutionBudgetLedgerFactoryPort(Protocol):
    """Create one mutable ledger instance per Run lifecycle."""

    def create_ledger(
        self,
        run_budget: RunBudget | None = None,
        *,
        tenant_id: str | None = None,
        run_id: RunId | None = None,
        attempt_id: AttemptId | None = None,
    ) -> ExecutionBudgetLedgerPort:
        """Return a fresh ledger for the active Run."""


__all__ = [
    "ChildBudgetAllocationDecision",
    "ExecutionBudgetAllocationMode",
    "ExecutionBudgetLedgerFactoryPort",
    "ExecutionBudgetLedgerPort",
    "ExecutionBudgetReservationGrant",
]
