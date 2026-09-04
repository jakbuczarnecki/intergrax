# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Provider-neutral worker budget accounting ports (AW-5B)."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from intergrax.contracts.autonomous_work.profile_reference import BudgetProfileRef
from intergrax.contracts.autonomous_work.worker_budget_accounting import (
    WorkerAccountingState,
    WorkerBudgetAdmissionResult,
    WorkerBudgetPolicy,
    WorkerBudgetReserveRequest,
    WorkerExecutionReservation,
    WorkerProactiveEvaluationAccountingRequest,
    BudgetUsageTotals,
)
from intergrax.contracts.execution_identity import ExecutionId


class WorkerBudgetProfileResolutionError(Exception):
    """Budget profile reference could not be resolved fail-closed."""


@runtime_checkable
class WorkerBudgetProfileResolver(Protocol):
    """Resolve ``BudgetProfileRef`` into immutable ``WorkerBudgetPolicy``."""

    def resolve(self, profile_ref: BudgetProfileRef) -> WorkerBudgetPolicy:
        """Return policy for ``profile_ref`` or raise ``WorkerBudgetProfileResolutionError``."""


@runtime_checkable
class ExecutionUsageProvider(Protocol):
    """Canonical execution usage evidence — AW does not inspect runtime internals."""

    def get_final_usage(self, execution_id: ExecutionId) -> BudgetUsageTotals | None:
        """Return final usage totals for ``execution_id`` when terminal evidence exists."""


class WorkerAccountingConflict(Exception):
    """Conflicting accounting mutation for the same canonical identity."""


class WorkerAccountingNotFound(Exception):
    """Accounting entity was not found for the requested identity."""


@runtime_checkable
class WorkerAccountingRepository(Protocol):
    """Atomic worker accounting persistence port."""

    def reserve(self, request: WorkerBudgetReserveRequest) -> WorkerBudgetAdmissionResult:
        """Atomically check quotas and reserve one concurrency slot."""

    def bind_execution(
        self,
        *,
        logical_dispatch: object,
        execution_id: ExecutionId,
        bound_at: datetime,
    ) -> WorkerExecutionReservation:
        """Bind a reserved slot to canonical ``ExecutionId`` and charge execution quotas."""

    def release_reservation(
        self,
        *,
        logical_dispatch: object,
        released_at: datetime,
    ) -> WorkerExecutionReservation:
        """Release an unbound reservation without counting as started execution."""

    def release_execution(
        self,
        *,
        worker_instance_id: str,
        execution_id: ExecutionId,
        released_at: datetime,
    ) -> WorkerExecutionReservation:
        """Idempotent terminal concurrency release for one bound execution."""

    def record_consumption(
        self,
        *,
        worker_instance_id: str,
        execution_id: ExecutionId,
        usage: BudgetUsageTotals,
        recorded_at: datetime,
    ) -> None:
        """Idempotent usage recording per canonical ``ExecutionId``."""

    def record_proactive_evaluation(
        self,
        request: WorkerProactiveEvaluationAccountingRequest,
    ) -> WorkerBudgetAdmissionResult:
        """Record one proactive evaluation attempt with durable idempotency."""

    def get_window_state(
        self,
        *,
        window: object,
    ) -> WorkerAccountingState | None:
        """Return persisted state for one immutable window identity."""
