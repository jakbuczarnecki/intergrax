# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Normalized case results for cross-domain qualification (DIAG-FUNCTIONAL-Q5)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.core.qualification.functional_diagnostic_expectation import (
    QualificationCaseComparison,
    QualificationFunctionalOutcome,
)
from intergrax.core.qualification.functional_qualification_attempts import QualificationAttemptRecord
from intergrax.runtime.diagnostics.functional_operator_projection import FunctionalOperatorOutcomeStatus


@dataclass(frozen=True, slots=True)
class QualificationNormalizedCaseResult:
    case_id: str
    task_id: str
    run_id: str
    attempt_id: str | None
    tenant_id: str | None
    comparison: QualificationCaseComparison
    functional_outcome: QualificationFunctionalOutcome
    diag_first_failed_check: str | None
    operator_outcome: str | None
    repeat_group: str | None
    identity_fidelity_match: bool
    is_healthy_case: bool
    expects_functional_failure: bool
    expects_inconclusive: bool
    expected_first_failed_check: str | None
    authoritative_attempt_index: int | None = None
    attempt_history: tuple[QualificationAttemptRecord, ...] = ()
    attempt_count: int = 1
    prerequisite_miss_count: int = 0
    prerequisite_exhausted: bool = False
    blocked_reason: str | None = None


@dataclass(frozen=True, slots=True)
class QualificationRepeatabilityGroup:
    group_id: str
    signatures: tuple[tuple[str, ...], ...]


def stage_matches_case(
    *,
    expected_first_failed_check: str | None,
    actual_first_failed_check: str | None,
) -> bool:
    return str(expected_first_failed_check or "") == str(actual_first_failed_check or "")


def functional_failure_detected(
    *,
    functional_outcome: QualificationFunctionalOutcome,
    operator_outcome: str | None,
) -> bool:
    return (
        functional_outcome is QualificationFunctionalOutcome.FAILED
        and operator_outcome == FunctionalOperatorOutcomeStatus.PROVEN_FUNCTIONAL_FAILURE.value
    )


def repeatability_passes(groups: tuple[QualificationRepeatabilityGroup, ...]) -> bool:
    for group in groups:
        if not group.signatures:
            return False
        if len(set(group.signatures)) != 1:
            return False
    return True


__all__ = [
    "QualificationNormalizedCaseResult",
    "QualificationRepeatabilityGroup",
    "functional_failure_detected",
    "repeatability_passes",
    "stage_matches_case",
]
