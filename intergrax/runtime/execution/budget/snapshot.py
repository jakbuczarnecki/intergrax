# © Artur Czarnecki. All rights reserved.

"""Serializable execution budget ledger snapshots (UE-9AR1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.execution_identity import AttemptId, ExecutionId, validate_attempt_id
from intergrax.runtime.execution.budget.models import (
    BudgetReservationScope,
    BudgetUsageTotals,
    ExecutionBudgetAllocationMode,
)
from intergrax.runtime.nexus.budget.budget_models import RunBudget

_SNAPSHOT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class PersistedBudgetRecord:
    """One reservation record captured for durable restore."""

    execution_id: ExecutionId
    parent_execution_id: ExecutionId
    mode: ExecutionBudgetAllocationMode
    allowance: BudgetUsageTotals
    reserved_scope: BudgetReservationScope
    consumed: BudgetUsageTotals
    child_reserved: BudgetUsageTotals
    released: bool


@dataclass(frozen=True, slots=True)
class RunBudgetLedgerSnapshot:
    """Durable per-Run budget state keyed by RunId."""

    schema_version: int
    attempt_id: AttemptId
    root_limits: RunBudget
    root_shared_consumed: BudgetUsageTotals
    root_permanent_consumed: BudgetUsageTotals
    records: tuple[PersistedBudgetRecord, ...]

    @staticmethod
    def empty(
        *,
        attempt_id: AttemptId,
        root_limits: RunBudget,
    ) -> RunBudgetLedgerSnapshot:
        return RunBudgetLedgerSnapshot(
            schema_version=_SNAPSHOT_SCHEMA_VERSION,
            attempt_id=attempt_id,
            root_limits=root_limits,
            root_shared_consumed=BudgetUsageTotals(),
            root_permanent_consumed=BudgetUsageTotals(),
            records=(),
        )


class RunBudgetLedgerSnapshotError(RuntimeError):
    """Raised when durable budget state is missing, corrupt, or inconsistent."""


def validate_snapshot_schema_version(schema_version: int) -> None:
    if schema_version != _SNAPSHOT_SCHEMA_VERSION:
        raise RunBudgetLedgerSnapshotError(
            f"unsupported run budget snapshot schema version: {schema_version!r}",
        )


def validate_snapshot_attempt_id(raw: object) -> AttemptId:
    if not isinstance(raw, str):
        raise RunBudgetLedgerSnapshotError("run budget snapshot attempt_id must be a string")
    return validate_attempt_id(raw)
