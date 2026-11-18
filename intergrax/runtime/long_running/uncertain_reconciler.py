# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Reconciler for UNCERTAIN scheduled resumes (PDC-D7-R015).

A lease-expired ``RUNNING`` schedule, a quarantined malformed ``PENDING`` row and a lease-expired
ledger action end as ``UNCERTAIN``: nobody knows whether the resume executed. This module decides,
per entry and from existing authorities only, what is proven:

- ``EXECUTION_PROVEN``: the terminal authority holds a terminal record for the task. The row is
  closed; it is never redriven.
- ``NOT_STARTED_PROVEN``: no attempt and no terminal record exist. The row may be redriven once.
- ``UNPROVEN``: evidence is missing or contradictory. The row stays ``UNCERTAIN`` and the recorded
  result says so; an operator decides. This is the fail-closed default.

Every entry also gets a double-execution check from the observed attempt count (at most one).
The resolve is a fenced transition of the source, so a second pass over the same row finds it
already resolved and records nothing for it.

Liveness: each entry is reconciled in isolation (a failing read or resolve is recorded with
``failed=True`` and the pass goes on; a failed entry is tried again on the next pass). An entry
recorded ``UNPROVEN`` is remembered with its fence for the life of the process and not recorded
again until its fence changes; the listing is widened by the remembered entries, so rows behind
them are still reached. The memory is process-local and pruned to the rows the source still lists.

Limit: the SQLite and PostgreSQL schedule stores do not implement ``UncertainScheduleSource`` yet
(no listing by status, no resolve transition). Until a store implements the port the reconciler
reaches only sources that do; for product hosts PDC-D7-R015 is not closed by this module alone.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol, runtime_checkable

from intergrax.runtime.long_running.scheduled_resume import ScheduledResume

__all__ = [
    "UNCERTAIN_RECONCILED_LOG_CODE",
    "DoubleExecutionCheck",
    "ReconciliationRecord",
    "UncertainExecutionEvidence",
    "UncertainOutcome",
    "UncertainReconciler",
    "UncertainScheduleSource",
]

logger = logging.getLogger(__name__)

UNCERTAIN_RECONCILED_LOG_CODE = "scheduled_resume_uncertain_reconciled"
DEFAULT_RECONCILE_BATCH_LIMIT = 100


class UncertainOutcome(StrEnum):
    EXECUTION_PROVEN = "execution_proven"
    NOT_STARTED_PROVEN = "not_started_proven"
    UNPROVEN = "unproven"


class DoubleExecutionCheck(StrEnum):
    AT_MOST_ONCE = "at_most_once"
    VIOLATED = "violated"
    NOT_OBSERVABLE = "not_observable"


@runtime_checkable
class UncertainScheduleSource(Protocol):
    """Schedule rows in ``UNCERTAIN`` and their fenced resolution."""

    def list_uncertain(self, limit: int) -> Sequence[ScheduledResume]:
        """At most ``limit`` rows whose status is ``UNCERTAIN``, in a stable order (oldest first)."""
        ...

    def resolve_uncertain(self, schedule_id: str, *, expected_fence: int, outcome: UncertainOutcome) -> bool:
        """Fenced transition out of ``UNCERTAIN``; False when the race was lost and nothing changed.

        ``EXECUTION_PROVEN`` closes the row as completed; ``NOT_STARTED_PROVEN`` returns it to
        ``PENDING`` for one redrive. ``UNPROVEN`` is never passed.
        """
        ...


@runtime_checkable
class UncertainExecutionEvidence(Protocol):
    """Reads over the existing terminal authority and attempt lifecycle; ``None`` means unavailable."""

    def terminal_record_present(self, *, tenant_id: str, task_id: str) -> bool | None: ...

    def attempts_observed(self, *, tenant_id: str, task_id: str) -> int | None: ...


@dataclass(frozen=True, slots=True)
class ReconciliationRecord:
    """Recorded result of one UNCERTAIN entry in one pass."""

    schedule_id: str
    tenant_id: str
    task_id: str
    outcome: UncertainOutcome
    resolve_applied: bool
    redrive_eligible: bool
    attempts_observed: int | None
    double_execution: DoubleExecutionCheck
    reconciled_at: datetime
    failed: bool = False


class UncertainReconciler:
    """Resolve UNCERTAIN schedules exactly once with a recorded double-execution check."""

    def __init__(
        self,
        source: UncertainScheduleSource,
        evidence: UncertainExecutionEvidence,
        *,
        batch_limit: int = DEFAULT_RECONCILE_BATCH_LIMIT,
    ) -> None:
        if batch_limit < 1:
            raise ValueError("batch_limit must be at least 1")
        self._source = source
        self._evidence = evidence
        self._batch_limit = batch_limit
        # schedule_id -> fence of entries already recorded UNPROVEN (process-local, pruned per pass).
        self._recorded_unproven: dict[str, int] = {}

    def reconcile_once(self, now: datetime) -> tuple[ReconciliationRecord, ...]:
        listed = self._source.list_uncertain(self._batch_limit + len(self._recorded_unproven))
        listed_ids = {entry.schedule_id for entry in listed}
        self._recorded_unproven = {
            schedule_id: fence for schedule_id, fence in self._recorded_unproven.items() if schedule_id in listed_ids
        }
        records: list[ReconciliationRecord] = []
        for entry in listed:
            if len(records) >= self._batch_limit:
                break
            if self._recorded_unproven.get(entry.schedule_id) == entry.fence:
                continue
            record = self._reconcile_isolated(entry, now)
            records.append(record)
            if record.outcome is UncertainOutcome.UNPROVEN and not record.failed:
                self._recorded_unproven[entry.schedule_id] = entry.fence
            log = logger.error if record.failed else logger.warning
            log(
                "%s schedule_id=%s outcome=%s resolve_applied=%s attempts=%s double_execution=%s failed=%s",
                UNCERTAIN_RECONCILED_LOG_CODE,
                record.schedule_id,
                record.outcome.value,
                record.resolve_applied,
                record.attempts_observed,
                record.double_execution.value,
                record.failed,
            )
        return tuple(records)

    def _reconcile_isolated(self, entry: ScheduledResume, now: datetime) -> ReconciliationRecord:
        """One entry; a failure is recorded (fail closed: UNPROVEN, nothing resolved) instead of raised."""
        try:
            return self._reconcile_entry(entry, now)
        except Exception:
            logger.exception(
                "%s schedule_id=%s reconciliation failed", UNCERTAIN_RECONCILED_LOG_CODE, entry.schedule_id
            )
            return ReconciliationRecord(
                schedule_id=entry.schedule_id,
                tenant_id=entry.tenant_id,
                task_id=entry.task_id,
                outcome=UncertainOutcome.UNPROVEN,
                resolve_applied=False,
                redrive_eligible=False,
                attempts_observed=None,
                double_execution=DoubleExecutionCheck.NOT_OBSERVABLE,
                reconciled_at=now,
                failed=True,
            )

    def _reconcile_entry(self, entry: ScheduledResume, now: datetime) -> ReconciliationRecord:
        terminal = self._evidence.terminal_record_present(tenant_id=entry.tenant_id, task_id=entry.task_id)
        attempts = self._evidence.attempts_observed(tenant_id=entry.tenant_id, task_id=entry.task_id)
        double_execution = _double_execution_check(attempts)
        outcome = _classify(terminal=terminal, attempts=attempts, double_execution=double_execution)
        applied = False
        if outcome is not UncertainOutcome.UNPROVEN:
            applied = self._source.resolve_uncertain(
                entry.schedule_id,
                expected_fence=entry.fence,
                outcome=outcome,
            )
        return ReconciliationRecord(
            schedule_id=entry.schedule_id,
            tenant_id=entry.tenant_id,
            task_id=entry.task_id,
            outcome=outcome,
            resolve_applied=applied,
            redrive_eligible=applied and outcome is UncertainOutcome.NOT_STARTED_PROVEN,
            attempts_observed=attempts,
            double_execution=double_execution,
            reconciled_at=now,
        )


def _double_execution_check(attempts: int | None) -> DoubleExecutionCheck:
    if attempts is None:
        return DoubleExecutionCheck.NOT_OBSERVABLE
    if attempts > 1:
        return DoubleExecutionCheck.VIOLATED
    return DoubleExecutionCheck.AT_MOST_ONCE


def _classify(
    *,
    terminal: bool | None,
    attempts: int | None,
    double_execution: DoubleExecutionCheck,
) -> UncertainOutcome:
    """Fail closed: anything short of proof, and any observed double execution, stays UNPROVEN."""
    if double_execution is not DoubleExecutionCheck.AT_MOST_ONCE:
        return UncertainOutcome.UNPROVEN
    if terminal is True:
        return UncertainOutcome.EXECUTION_PROVEN
    if terminal is False and attempts == 0:
        return UncertainOutcome.NOT_STARTED_PROVEN
    return UncertainOutcome.UNPROVEN
