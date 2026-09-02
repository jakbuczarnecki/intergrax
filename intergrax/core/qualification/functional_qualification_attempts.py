# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Bounded attempt orchestration for prerequisite-conditioned qualification cases."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeVar

_T = TypeVar("_T")


class QualificationPreconditionStatus(StrEnum):
    SATISFIED = "satisfied"
    NOT_SATISFIED = "not_satisfied"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class QualificationAttemptPolicy:
    max_attempts: int = 1


@dataclass(frozen=True, slots=True)
class QualificationAttemptRecord:
    attempt_index: int
    precondition_status: QualificationPreconditionStatus
    task_id: str
    run_id: str
    summary: str | None = None


@dataclass(frozen=True, slots=True)
class QualificationBoundedAttemptOutcome:
  authoritative_result: _T | None
  authoritative_attempt_index: int | None
  attempt_records: tuple[QualificationAttemptRecord, ...]
  exhausted: bool = False
  blocked_reason: str | None = None


def execute_bounded_attempts(
    policy: QualificationAttemptPolicy,
    execute_attempt: Callable[[int], tuple[_T, QualificationPreconditionStatus, str, str]],
    summarize_miss: Callable[[_T], str | None],
) -> QualificationBoundedAttemptOutcome[_T]:
    """Run bounded real attempts; first SATISFIED attempt is authoritative.

    execute_attempt returns (result, precondition_status, task_id, run_id).
    On NOT_SATISFIED the attempt is recorded and the next attempt runs.
    On SATISFIED the result is compared authoritatively and no further attempts run.
    On BLOCKED the sequence stops immediately.
  """
    if policy.max_attempts < 1:
        raise ValueError("qualification_attempt_policy_invalid:max_attempts")

    attempt_records: list[QualificationAttemptRecord] = []
    for attempt_index in range(1, policy.max_attempts + 1):
        result, precondition_status, task_id, run_id = execute_attempt(attempt_index)
        attempt_records.append(
            QualificationAttemptRecord(
                attempt_index=attempt_index,
                precondition_status=precondition_status,
                task_id=task_id,
                run_id=run_id,
                summary=summarize_miss(result) if precondition_status is QualificationPreconditionStatus.NOT_SATISFIED else None,
            ),
        )
        if precondition_status is QualificationPreconditionStatus.BLOCKED:
            return QualificationBoundedAttemptOutcome(
                authoritative_result=None,
                authoritative_attempt_index=None,
                attempt_records=tuple(attempt_records),
                exhausted=False,
                blocked_reason="prerequisite_blocked",
            )
        if precondition_status is QualificationPreconditionStatus.SATISFIED:
            return QualificationBoundedAttemptOutcome(
                authoritative_result=result,
                authoritative_attempt_index=attempt_index,
                attempt_records=tuple(attempt_records),
            )

    return QualificationBoundedAttemptOutcome(
        authoritative_result=None,
        authoritative_attempt_index=None,
        attempt_records=tuple(attempt_records),
        exhausted=True,
        blocked_reason="prerequisite_not_reached",
    )


def compute_attempt_metrics(
    attempt_records: tuple[QualificationAttemptRecord, ...],
) -> tuple[int, int, int]:
    """Return (total_attempts, prerequisite_misses, prerequisite_exhaustions marker)."""
    total = len(attempt_records)
    misses = sum(
        1 for item in attempt_records if item.precondition_status is QualificationPreconditionStatus.NOT_SATISFIED
    )
    exhaustions = 1 if total > 0 and all(
        item.precondition_status is QualificationPreconditionStatus.NOT_SATISFIED for item in attempt_records
    ) else 0
    return total, misses, exhaustions


__all__ = [
    "QualificationAttemptPolicy",
    "QualificationAttemptRecord",
    "QualificationBoundedAttemptOutcome",
    "QualificationPreconditionStatus",
    "compute_attempt_metrics",
    "execute_bounded_attempts",
]
