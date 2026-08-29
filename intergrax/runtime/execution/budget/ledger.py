# © Artur Czarnecki. All rights reserved.

"""Canonical execution budget ledger (UE-8B1)."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Protocol

from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    validate_execution_id,
)
from intergrax.runtime.execution.budget.snapshot import (
    PersistedBudgetRecord,
    RunBudgetLedgerSnapshot,
)
from intergrax.runtime.execution.budget.models import (
    BudgetReservationScope,
    BudgetUsageTotals,
    ChildBudgetAllocationDecision,
    ExecutionBudgetAllocationMode,
    ExecutionBudgetError,
    ExecutionBudgetReservationError,
    ExecutionBudgetReservationGrant,
    parse_reservation_request,
    reservation_remaining_to_run_budget,
    run_budget_to_usage_totals,
)
from intergrax.runtime.nexus.budget.budget_models import RunBudget

ROOT_BUDGET_POOL_PARENT: ExecutionId = validate_execution_id(
    "exec_00000000000000000000000000000001"
)


@dataclass
class _ReservationRecord:
    execution_id: ExecutionId
    parent_execution_id: ExecutionId
    mode: ExecutionBudgetAllocationMode
    allowance: BudgetUsageTotals
    reserved_scope: BudgetReservationScope
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


def create_execution_budget_ledger(run_budget: RunBudget | None) -> InMemoryExecutionBudgetLedger:
    """Create one canonical in-memory ledger for a Run from ``RunBudget`` root limits."""
    limits = run_budget if run_budget is not None else RunBudget()
    return InMemoryExecutionBudgetLedger(root_limits=limits)


class ExecutionBudgetLedgerFactory(Protocol):
    """Create one mutable ledger instance per Run lifecycle."""

    def create_ledger(
        self,
        run_budget: RunBudget | None = None,
        *,
        tenant_id: str | None = None,
        run_id: RunId | None = None,
        attempt_id: AttemptId | None = None,
    ) -> ExecutionBudgetLedger:
        """Return a fresh ledger for the active Run."""


@dataclass(frozen=True, slots=True)
class RunBudgetExecutionBudgetLedgerFactory:
    """Default factory that instantiates in-memory ledgers from ``RunBudget``."""

    default_run_budget: RunBudget | None = None

    def create_ledger(
        self,
        run_budget: RunBudget | None = None,
        *,
        tenant_id: str | None = None,
        run_id: RunId | None = None,
        attempt_id: AttemptId | None = None,
    ) -> ExecutionBudgetLedger:
        del tenant_id, run_id, attempt_id
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

    def create_ledger(
        self,
        run_budget: RunBudget | None = None,
        *,
        tenant_id: str | None = None,
        run_id: RunId | None = None,
        attempt_id: AttemptId | None = None,
    ) -> ExecutionBudgetLedger:
        del run_budget, tenant_id, run_id, attempt_id
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
                    reserved_scope=BudgetReservationScope(),
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
            parsed = parse_reservation_request(decision.reservation_request)
            if parsed.scope.is_empty:
                raise ExecutionBudgetReservationError(
                    "reserved allocation requires at least one explicit dimension"
                )
            requested = parsed.allowance
            reserved_scope = parsed.scope
            self._validate_positive_request(requested)
            self._validate_reservation_fits_backing_unlocked(
                requested,
                reserved_scope,
                parent_execution_id,
            )
            self._apply_child_reservation_to_parent(
                parent_execution_id,
                requested,
                reserved_scope,
            )
            record = _ReservationRecord(
                execution_id=execution_id,
                parent_execution_id=parent_execution_id,
                mode=ExecutionBudgetAllocationMode.RESERVED,
                allowance=requested,
                reserved_scope=reserved_scope,
                consumed=BudgetUsageTotals(),
                child_reserved=BudgetUsageTotals(),
            )
            self._records[execution_id] = record
            return ExecutionBudgetReservationGrant(
                execution_id=execution_id,
                parent_execution_id=parent_execution_id,
                mode=ExecutionBudgetAllocationMode.RESERVED,
                reservation_allowance=reservation_remaining_to_run_budget(
                    requested,
                    reserved_scope,
                ),
            )

    def ensure_shared_participant(
        self,
        execution_id: ExecutionId,
        *,
        parent_execution_id: ExecutionId,
    ) -> None:
        with self._lock:
            if execution_id in self._records:
                return
            record = _ReservationRecord(
                execution_id=execution_id,
                parent_execution_id=parent_execution_id,
                mode=ExecutionBudgetAllocationMode.SHARED,
                allowance=BudgetUsageTotals(),
                reserved_scope=BudgetReservationScope(),
                consumed=BudgetUsageTotals(),
                child_reserved=BudgetUsageTotals(),
            )
            self._records[execution_id] = record

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
                    record.reserved_scope,
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
            own_reserved, ancestor_debits, root_shared = self._partition_consumption_unlocked(
                record,
                amounts,
            )
            if not self._partitioned_consumption_fits_unlocked(
                record,
                own_reserved,
                ancestor_debits,
                root_shared,
            ):
                raise ExecutionBudgetError(
                    f"consumption exceeds effective grant for execution {execution_id!r}"
                )
            if record.mode is ExecutionBudgetAllocationMode.RESERVED:
                record.consumed = record.consumed.add(own_reserved)
            else:
                record.consumed = record.consumed.add(amounts)
            for ancestor_id, ancestor_amounts in ancestor_debits.items():
                ancestor_record = self._require_active_record_unlocked(ancestor_id)
                ancestor_record.consumed = ancestor_record.consumed.add(ancestor_amounts)
            if not self._is_zero_usage(root_shared):
                self._root_shared_consumed = self._root_shared_consumed.add(root_shared)

    def snapshot_root_available(self) -> RunBudget:
        with self._lock:
            return self._available_at_root_pool_unlocked()

    def snapshot_reservation_remaining(self, execution_id: ExecutionId) -> RunBudget:
        with self._lock:
            record = self._require_active_record_unlocked(execution_id)
            return reservation_remaining_to_run_budget(
                self._effective_remaining_for_record_unlocked(record),
                record.reserved_scope,
            )

    def export_snapshot(self, attempt_id: AttemptId) -> RunBudgetLedgerSnapshot:
        with self._lock:
            records = tuple(
                PersistedBudgetRecord(
                    execution_id=record.execution_id,
                    parent_execution_id=record.parent_execution_id,
                    mode=record.mode,
                    allowance=record.allowance,
                    reserved_scope=record.reserved_scope,
                    consumed=record.consumed,
                    child_reserved=record.child_reserved,
                    released=record.released,
                )
                for record in self._records.values()
            )
            return RunBudgetLedgerSnapshot(
                schema_version=1,
                attempt_id=attempt_id,
                root_limits=self._root_limits,
                root_shared_consumed=self._root_shared_consumed,
                root_permanent_consumed=self._root_permanent_consumed,
                records=records,
            )

    def restore_snapshot(self, snapshot: RunBudgetLedgerSnapshot) -> None:
        with self._lock:
            self._root_limits = snapshot.root_limits
            self._root_shared_consumed = snapshot.root_shared_consumed
            self._root_permanent_consumed = snapshot.root_permanent_consumed
            self._records = {
                record.execution_id: _ReservationRecord(
                    execution_id=record.execution_id,
                    parent_execution_id=record.parent_execution_id,
                    mode=record.mode,
                    allowance=record.allowance,
                    reserved_scope=record.reserved_scope,
                    consumed=record.consumed,
                    child_reserved=record.child_reserved,
                    released=record.released,
                )
                for record in snapshot.records
            }

    def settle_unreleased_reservations(self) -> None:
        """Commit consumed amounts and release unreleased reservation holds."""
        with self._lock:
            self._settle_unreleased_reservations_unlocked()

    def prepare_for_attempt_redelivery(self) -> None:
        """Finalize prior-attempt reservation state before a new Attempt starts."""
        with self._lock:
            self._settle_unreleased_reservations_unlocked()

    def _settle_unreleased_reservations_unlocked(self) -> None:
        for execution_id in list(self._records.keys()):
            record = self._records[execution_id]
            if record.released:
                continue
            if record.mode is ExecutionBudgetAllocationMode.RESERVED:
                self._release_child_reservation_from_parent(
                    record.parent_execution_id,
                    record.allowance,
                    record.reserved_scope,
                )
                self._commit_consumed_to_backing_unlocked(record)
            record.released = True
        self._records.clear()

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
            remaining = record.allowance.subtract(record.consumed).subtract(
                record.child_reserved
            )
            totals = totals.add(record.reserved_scope.select(remaining))
        return totals

    def _effective_remaining_for_record_unlocked(
        self,
        record: _ReservationRecord,
    ) -> BudgetUsageTotals:
        if record.mode is ExecutionBudgetAllocationMode.RESERVED:
            return record.allowance.subtract(record.consumed).subtract(record.child_reserved)
        return self._shared_pool_available_unlocked(record)

    def _shared_pool_available_unlocked(
        self,
        record: _ReservationRecord,
    ) -> BudgetUsageTotals:
        available = run_budget_to_usage_totals(self._available_at_root_pool_unlocked())
        current_parent = record.parent_execution_id
        while True:
            parent_record = self._records.get(current_parent)
            if parent_record is None or parent_record.released:
                break
            if parent_record.mode is ExecutionBudgetAllocationMode.RESERVED:
                parent_remaining = parent_record.allowance.subtract(
                    parent_record.consumed
                ).subtract(parent_record.child_reserved)
                available = self._overlay_scoped_dimensions(
                    available,
                    parent_remaining,
                    parent_record.reserved_scope,
                )
            current_parent = parent_record.parent_execution_id
        return available

    def _available_for_new_reservation_unlocked(
        self,
        parent_execution_id: ExecutionId,
        scope: BudgetReservationScope,
    ) -> BudgetUsageTotals:
        root_available = run_budget_to_usage_totals(self._available_at_root_pool_unlocked())
        available = root_available
        current_parent = parent_execution_id
        while True:
            parent_record = self._records.get(current_parent)
            if parent_record is None or parent_record.released:
                break
            if parent_record.mode is ExecutionBudgetAllocationMode.RESERVED:
                parent_remaining = parent_record.allowance.subtract(
                    parent_record.consumed
                ).subtract(parent_record.child_reserved)
                available = self._overlay_scoped_dimensions(
                    available,
                    parent_remaining,
                    parent_record.reserved_scope,
                )
            current_parent = parent_record.parent_execution_id
        return scope.select(available)

    def _validate_reservation_fits_backing_unlocked(
        self,
        requested: BudgetUsageTotals,
        scope: BudgetReservationScope,
        parent_execution_id: ExecutionId,
    ) -> None:
        available = self._available_for_new_reservation_unlocked(parent_execution_id, scope)
        scoped_requested = scope.select(requested)
        if not self._usage_fits(scoped_requested, available):
            raise ExecutionBudgetReservationError(
                "reservation exceeds available budget at backing level"
            )

    def _apply_child_reservation_to_parent(
        self,
        parent_execution_id: ExecutionId,
        requested: BudgetUsageTotals,
        scope: BudgetReservationScope,
    ) -> None:
        parent_record = self._records.get(parent_execution_id)
        if parent_record is None:
            return
        parent_record.child_reserved = parent_record.child_reserved.add(
            scope.select(requested)
        )

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
        scope: BudgetReservationScope,
    ) -> None:
        parent_record = self._records.get(parent_execution_id)
        if parent_record is None:
            return
        parent_record.child_reserved = parent_record.child_reserved.subtract(
            scope.select(released_allowance)
        )

    def _partition_consumption_unlocked(
        self,
        record: _ReservationRecord,
        amounts: BudgetUsageTotals,
    ) -> tuple[BudgetUsageTotals, dict[ExecutionId, BudgetUsageTotals], BudgetUsageTotals]:
        own_reserved = BudgetUsageTotals()
        if record.mode is ExecutionBudgetAllocationMode.RESERVED:
            own_reserved = record.reserved_scope.select(amounts)
            pending = record.reserved_scope.complement_select(amounts)
        else:
            pending = amounts

        ancestor_debits: dict[ExecutionId, BudgetUsageTotals] = {}
        root_shared = BudgetUsageTotals()
        current_parent = record.parent_execution_id

        while not self._is_zero_usage(pending):
            ancestor_id = self._find_reserved_ancestor_unlocked(current_parent)
            if ancestor_id is None:
                root_shared = root_shared.add(pending)
                break
            ancestor_record = self._records[ancestor_id]
            scoped = ancestor_record.reserved_scope.select(pending)
            pending = ancestor_record.reserved_scope.complement_select(pending)
            if not self._is_zero_usage(scoped):
                existing = ancestor_debits.get(ancestor_id)
                ancestor_debits[ancestor_id] = (
                    scoped if existing is None else existing.add(scoped)
                )
            current_parent = ancestor_record.parent_execution_id

        return own_reserved, ancestor_debits, root_shared

    def _partitioned_consumption_fits_unlocked(
        self,
        record: _ReservationRecord,
        own_reserved: BudgetUsageTotals,
        ancestor_debits: dict[ExecutionId, BudgetUsageTotals],
        root_shared: BudgetUsageTotals,
    ) -> bool:
        if record.mode is ExecutionBudgetAllocationMode.RESERVED:
            remaining = self._effective_remaining_for_record_unlocked(record)
            if not self._usage_fits(own_reserved, remaining):
                return False
        for ancestor_id, ancestor_amounts in ancestor_debits.items():
            ancestor_record = self._require_active_record_unlocked(ancestor_id)
            remaining = self._effective_remaining_for_record_unlocked(ancestor_record)
            scoped_amounts = ancestor_record.reserved_scope.select(ancestor_amounts)
            if not self._usage_fits(scoped_amounts, remaining):
                return False
        if not self._is_zero_usage(root_shared):
            root_available = run_budget_to_usage_totals(self._available_at_root_pool_unlocked())
            if not self._usage_fits(root_shared, root_available):
                return False
        return True

    def _find_reserved_ancestor_unlocked(
        self,
        parent_execution_id: ExecutionId,
    ) -> ExecutionId | None:
        current_parent = parent_execution_id
        while True:
            parent_record = self._records.get(current_parent)
            if parent_record is None or parent_record.released:
                return None
            if parent_record.mode is ExecutionBudgetAllocationMode.RESERVED:
                return current_parent
            current_parent = parent_record.parent_execution_id

    @staticmethod
    def _overlay_scoped_dimensions(
        base: BudgetUsageTotals,
        overlay: BudgetUsageTotals,
        scope: BudgetReservationScope,
    ) -> BudgetUsageTotals:
        return BudgetUsageTotals(
            input_tokens=overlay.input_tokens if scope.input_tokens else base.input_tokens,
            output_tokens=overlay.output_tokens if scope.output_tokens else base.output_tokens,
            total_tokens=overlay.total_tokens if scope.total_tokens else base.total_tokens,
            llm_calls=overlay.llm_calls if scope.llm_calls else base.llm_calls,
            tool_calls=overlay.tool_calls if scope.tool_calls else base.tool_calls,
            rag_invocations=overlay.rag_invocations if scope.rag_invocations else base.rag_invocations,
            websearch_invocations=overlay.websearch_invocations
            if scope.websearch_invocations
            else base.websearch_invocations,
            wall_time_seconds=overlay.wall_time_seconds
            if scope.wall_time_seconds
            else base.wall_time_seconds,
            planner_iterations=overlay.planner_iterations
            if scope.planner_iterations
            else base.planner_iterations,
            replans=overlay.replans if scope.replans else base.replans,
        )

    @staticmethod
    def _is_zero_usage(amounts: BudgetUsageTotals) -> bool:
        return (
            amounts.input_tokens == 0
            and amounts.output_tokens == 0
            and amounts.total_tokens == 0
            and amounts.llm_calls == 0
            and amounts.tool_calls == 0
            and amounts.rag_invocations == 0
            and amounts.websearch_invocations == 0
            and amounts.wall_time_seconds == 0.0
            and amounts.planner_iterations == 0
            and amounts.replans == 0
        )

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
