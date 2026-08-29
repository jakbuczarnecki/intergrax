# © Artur Czarnecki. All rights reserved.

"""Canonical execution budget ledger (UE-8B1)."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Protocol

from intergrax.contracts.execution_identity import ExecutionId
from intergrax.runtime.execution.budget.models import (
    BudgetUsageTotals,
    ChildBudgetAllocationDecision,
    ExecutionBudgetAllocationMode,
    ExecutionBudgetError,
    ExecutionBudgetReservationError,
    ExecutionBudgetReservationGrant,
    run_budget_to_usage_totals,
    usage_totals_to_run_budget,
)
from intergrax.runtime.nexus.budget.budget_models import RunBudget


@dataclass
class _ReservationRecord:
    execution_id: ExecutionId
    parent_execution_id: ExecutionId
    mode: ExecutionBudgetAllocationMode
    allowance: BudgetUsageTotals
    consumed: BudgetUsageTotals
    child_reserved: BudgetUsageTotals
    released: bool = False


class ExecutionBudgetLedger(Protocol):
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

    def consume_budget(
        self,
        execution_id: ExecutionId,
        amounts: BudgetUsageTotals,
    ) -> None:
        """Consume budget against the effective grant for ``execution_id``."""

    def snapshot_root_available(self) -> RunBudget:
        """Return remaining capacity at the Run root pool."""


def create_execution_budget_ledger(run_budget: RunBudget | None) -> InMemoryExecutionBudgetLedger:
    """Create one canonical in-memory ledger for a Run from ``RunBudget`` root limits."""
    limits = run_budget if run_budget is not None else RunBudget()
    return InMemoryExecutionBudgetLedger(root_limits=limits)


class ExecutionBudgetLedgerFactory(Protocol):
    """Create one mutable ledger instance per Run lifecycle."""

    def create_ledger(self, run_budget: RunBudget | None = None) -> ExecutionBudgetLedger:
        """Return a fresh ledger for the active Run."""


@dataclass(frozen=True, slots=True)
class RunBudgetExecutionBudgetLedgerFactory:
    """Default factory that instantiates in-memory ledgers from ``RunBudget``."""

    default_run_budget: RunBudget | None = None

    def create_ledger(self, run_budget: RunBudget | None = None) -> ExecutionBudgetLedger:
        resolved = run_budget if run_budget is not None else self.default_run_budget
        return create_execution_budget_ledger(resolved)


def create_execution_budget_ledger_factory(
    run_budget: RunBudget | None = None,
) -> RunBudgetExecutionBudgetLedgerFactory:
    """Build the default per-Run ledger factory for composition."""
    return RunBudgetExecutionBudgetLedgerFactory(default_run_budget=run_budget)


@dataclass(frozen=True, slots=True)
class FixedExecutionBudgetLedgerFactory:
    """Explicit test/provider factory that always returns the same ledger instance."""

    ledger: ExecutionBudgetLedger

    def create_ledger(self, run_budget: RunBudget | None = None) -> ExecutionBudgetLedger:
        del run_budget
        return self.ledger


def fixed_execution_budget_ledger_factory(
    ledger: ExecutionBudgetLedger,
) -> FixedExecutionBudgetLedgerFactory:
    """Wrap a pre-built ledger for narrowly scoped test injection."""
    return FixedExecutionBudgetLedgerFactory(ledger=ledger)


class InMemoryExecutionBudgetLedger:
    """Thread-safe in-process canonical execution budget ledger."""

    __slots__ = ("_lock", "_root_limits", "_records", "_root_shared_consumed", "_root_permanent_consumed")

    def __init__(self, *, root_limits: RunBudget) -> None:
        self._lock = threading.Lock()
        self._root_limits = root_limits
        self._records: dict[ExecutionId, _ReservationRecord] = {}
        self._root_shared_consumed = BudgetUsageTotals()
        self._root_permanent_consumed = BudgetUsageTotals()

    def grant_child_budget(
        self,
        *,
        execution_id: ExecutionId,
        parent_execution_id: ExecutionId,
        decision: ChildBudgetAllocationDecision,
    ) -> ExecutionBudgetReservationGrant:
        with self._lock:
            if execution_id in self._records:
                raise ExecutionBudgetReservationError(
                    f"budget already granted for execution {execution_id!r}"
                )
            if decision.mode is ExecutionBudgetAllocationMode.SHARED:
                record = _ReservationRecord(
                    execution_id=execution_id,
                    parent_execution_id=parent_execution_id,
                    mode=ExecutionBudgetAllocationMode.SHARED,
                    allowance=BudgetUsageTotals(),
                    consumed=BudgetUsageTotals(),
                    child_reserved=BudgetUsageTotals(),
                )
                self._records[execution_id] = record
                return ExecutionBudgetReservationGrant(
                    execution_id=execution_id,
                    parent_execution_id=parent_execution_id,
                    mode=ExecutionBudgetAllocationMode.SHARED,
                    reservation_allowance=None,
                )

            if decision.reservation_request is None:
                raise ExecutionBudgetReservationError(
                    "reserved allocation requires reservation_request"
                )
            requested = run_budget_to_usage_totals(decision.reservation_request)
            self._validate_positive_request(requested)
            backing_id = self._resolve_backing_execution_id_unlocked(parent_execution_id)
            self._validate_reservation_fits_backing_unlocked(requested, backing_id)
            self._apply_child_reservation_to_parent(parent_execution_id, requested)
            record = _ReservationRecord(
                execution_id=execution_id,
                parent_execution_id=parent_execution_id,
                mode=ExecutionBudgetAllocationMode.RESERVED,
                allowance=requested,
                consumed=BudgetUsageTotals(),
                child_reserved=BudgetUsageTotals(),
            )
            self._records[execution_id] = record
            return ExecutionBudgetReservationGrant(
                execution_id=execution_id,
                parent_execution_id=parent_execution_id,
                mode=ExecutionBudgetAllocationMode.RESERVED,
                reservation_allowance=usage_totals_to_run_budget(requested),
            )

    def release_child_budget(self, execution_id: ExecutionId) -> None:
        with self._lock:
            record = self._records.get(execution_id)
            if record is None:
                raise ExecutionBudgetReservationError(
                    f"unknown execution budget reservation {execution_id!r}"
                )
            if record.released:
                raise ExecutionBudgetReservationError(
                    f"budget reservation already released for execution {execution_id!r}"
                )
            if record.mode is ExecutionBudgetAllocationMode.RESERVED:
                self._release_child_reservation_from_parent(
                    record.parent_execution_id,
                    record.allowance,
                )
                self._commit_consumed_to_backing_unlocked(record)
            record.released = True

    def consume_budget(
        self,
        execution_id: ExecutionId,
        amounts: BudgetUsageTotals,
    ) -> None:
        with self._lock:
            record = self._require_active_record_unlocked(execution_id)
            self._validate_positive_request(amounts)
            remaining = self._effective_remaining_for_record_unlocked(record)
            if not self._usage_fits(amounts, remaining):
                raise ExecutionBudgetError(
                    f"consumption exceeds effective grant for execution {execution_id!r}"
                )
            record.consumed = record.consumed.add(amounts)
            if record.mode is ExecutionBudgetAllocationMode.SHARED:
                backing_id = self._resolve_backing_execution_id_unlocked(
                    record.parent_execution_id
                )
                if backing_id is None:
                    self._root_shared_consumed = self._root_shared_consumed.add(amounts)
                else:
                    backing_record = self._records.get(backing_id)
                    if backing_record is not None:
                        backing_record.consumed = backing_record.consumed.add(amounts)

    def snapshot_root_available(self) -> RunBudget:
        with self._lock:
            return self._available_at_root_pool_unlocked()

    def snapshot_reservation_remaining(self, execution_id: ExecutionId) -> RunBudget:
        with self._lock:
            record = self._require_active_record_unlocked(execution_id)
            return usage_totals_to_run_budget(
                self._effective_remaining_for_record_unlocked(record)
            )

    def _require_active_record_unlocked(self, execution_id: ExecutionId) -> _ReservationRecord:
        record = self._records.get(execution_id)
        if record is None or record.released:
            raise ExecutionBudgetReservationError(
                f"unknown execution budget reservation {execution_id!r}"
            )
        return record

    def _resolve_backing_execution_id_unlocked(
        self,
        parent_execution_id: ExecutionId,
    ) -> ExecutionId | None:
        parent_record = self._records.get(parent_execution_id)
        if parent_record is None:
            return None
        if parent_record.mode is ExecutionBudgetAllocationMode.RESERVED:
            return parent_execution_id
        return self._resolve_backing_execution_id_unlocked(parent_record.parent_execution_id)

    def _root_consumed_unlocked(self) -> BudgetUsageTotals:
        return self._root_shared_consumed.add(self._root_permanent_consumed)

    def _available_at_root_pool_unlocked(self) -> RunBudget:
        root_consumed = self._root_consumed_unlocked()
        reserved = self._top_level_reserved_totals_unlocked()
        return RunBudget(
            max_input_tokens=self._available_dimension(
                self._root_limits.max_input_tokens,
                root_consumed.input_tokens,
                reserved.input_tokens,
            ),
            max_output_tokens=self._available_dimension(
                self._root_limits.max_output_tokens,
                root_consumed.output_tokens,
                reserved.output_tokens,
            ),
            max_total_tokens=self._available_dimension(
                self._root_limits.max_total_tokens,
                root_consumed.total_tokens,
                reserved.total_tokens,
            ),
            max_llm_calls=self._available_dimension(
                self._root_limits.max_llm_calls,
                root_consumed.llm_calls,
                reserved.llm_calls,
            ),
            max_tool_calls=self._available_dimension(
                self._root_limits.max_tool_calls,
                root_consumed.tool_calls,
                reserved.tool_calls,
            ),
            max_rag_invocations=self._available_dimension(
                self._root_limits.max_rag_invocations,
                root_consumed.rag_invocations,
                reserved.rag_invocations,
            ),
            max_websearch_invocations=self._available_dimension(
                self._root_limits.max_websearch_invocations,
                root_consumed.websearch_invocations,
                reserved.websearch_invocations,
            ),
            max_wall_time_seconds=self._available_dimension_float(
                self._root_limits.max_wall_time_seconds,
                root_consumed.wall_time_seconds,
                reserved.wall_time_seconds,
            ),
            max_planner_iterations=self._available_dimension(
                self._root_limits.max_planner_iterations,
                root_consumed.planner_iterations,
                reserved.planner_iterations,
            ),
            max_replans=self._available_dimension(
                self._root_limits.max_replans,
                root_consumed.replans,
                reserved.replans,
            ),
        )

    def _top_level_reserved_totals_unlocked(self) -> BudgetUsageTotals:
        totals = BudgetUsageTotals()
        for record in self._records.values():
            if record.released:
                continue
            if record.mode is not ExecutionBudgetAllocationMode.RESERVED:
                continue
            if self._resolve_backing_execution_id_unlocked(record.parent_execution_id) is not None:
                continue
            totals = totals.add(
                record.allowance.subtract(record.consumed).subtract(record.child_reserved)
            )
        return totals

    def _available_at_backing_unlocked(
        self,
        backing_id: ExecutionId | None,
    ) -> BudgetUsageTotals:
        if backing_id is None:
            return run_budget_to_usage_totals(self._available_at_root_pool_unlocked())
        record = self._require_active_record_unlocked(backing_id)
        return self._effective_remaining_for_record_unlocked(record)

    def _effective_remaining_for_record_unlocked(
        self,
        record: _ReservationRecord,
    ) -> BudgetUsageTotals:
        if record.mode is ExecutionBudgetAllocationMode.RESERVED:
            return record.allowance.subtract(record.consumed).subtract(record.child_reserved)
        backing_id = self._resolve_backing_execution_id_unlocked(record.parent_execution_id)
        return self._available_at_backing_unlocked(backing_id)

    def _validate_reservation_fits_backing_unlocked(
        self,
        requested: BudgetUsageTotals,
        backing_id: ExecutionId | None,
    ) -> None:
        available = self._available_at_backing_unlocked(backing_id)
        if not self._usage_fits(requested, available):
            raise ExecutionBudgetReservationError(
                "reservation exceeds available budget at backing level"
            )

    def _apply_child_reservation_to_parent(
        self,
        parent_execution_id: ExecutionId,
        requested: BudgetUsageTotals,
    ) -> None:
        parent_record = self._records.get(parent_execution_id)
        if parent_record is None:
            return
        parent_record.child_reserved = parent_record.child_reserved.add(requested)

    def _commit_consumed_to_backing_unlocked(self, record: _ReservationRecord) -> None:
        backing_id = self._resolve_backing_execution_id_unlocked(record.parent_execution_id)
        if backing_id is None:
            self._root_permanent_consumed = self._root_permanent_consumed.add(record.consumed)
            return
        parent_record = self._records.get(backing_id)
        if parent_record is None:
            self._root_permanent_consumed = self._root_permanent_consumed.add(record.consumed)
            return
        parent_record.consumed = parent_record.consumed.add(record.consumed)

    def _release_child_reservation_from_parent(
        self,
        parent_execution_id: ExecutionId,
        released_allowance: BudgetUsageTotals,
    ) -> None:
        parent_record = self._records.get(parent_execution_id)
        if parent_record is None:
            return
        parent_record.child_reserved = parent_record.child_reserved.subtract(released_allowance)

    @staticmethod
    def _validate_positive_request(amounts: BudgetUsageTotals) -> None:
        if (
            amounts.input_tokens < 0
            or amounts.output_tokens < 0
            or amounts.total_tokens < 0
            or amounts.llm_calls < 0
            or amounts.tool_calls < 0
            or amounts.rag_invocations < 0
            or amounts.websearch_invocations < 0
            or amounts.wall_time_seconds < 0
            or amounts.planner_iterations < 0
            or amounts.replans < 0
        ):
            raise ExecutionBudgetError("budget amounts must be non-negative")

    @staticmethod
    def _usage_fits(requested: BudgetUsageTotals, available: BudgetUsageTotals) -> bool:
        return (
            requested.input_tokens <= available.input_tokens
            and requested.output_tokens <= available.output_tokens
            and requested.total_tokens <= available.total_tokens
            and requested.llm_calls <= available.llm_calls
            and requested.tool_calls <= available.tool_calls
            and requested.rag_invocations <= available.rag_invocations
            and requested.websearch_invocations <= available.websearch_invocations
            and requested.wall_time_seconds <= available.wall_time_seconds
            and requested.planner_iterations <= available.planner_iterations
            and requested.replans <= available.replans
        )

    @staticmethod
    def _available_dimension(
        limit: int | None,
        consumed: int,
        reserved: int,
    ) -> int | None:
        if limit is None:
            return None
        return max(0, limit - consumed - reserved)

    @staticmethod
    def _available_dimension_float(
        limit: float | None,
        consumed: float,
        reserved: float,
    ) -> float | None:
        if limit is None:
            return None
        return max(0.0, limit - consumed - reserved)
