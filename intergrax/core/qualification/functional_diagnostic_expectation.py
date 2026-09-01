# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Typed qualification expectations for functional diagnostic ground truth (DIAG-FUNCTIONAL-Q1)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.runtime.diagnostics.functional_diagnostic_identity import FunctionalDiagnosticCheckId
from intergrax.runtime.diagnostics.functional_diagnostic_analysis import FunctionalDiagnosticCheckStatus
from intergrax.runtime.diagnostics.functional_operator_projection import FunctionalOperatorOutcomeStatus


class QualificationExecutionOutcome(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"


class QualificationFunctionalOutcome(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, slots=True)
class QualificationCheckExpectation:
    check_id: FunctionalDiagnosticCheckId
    expected_status: FunctionalDiagnosticCheckStatus


@dataclass(frozen=True, slots=True)
class QualificationCaseExpectation:
    case_id: str
    expected_execution_outcome: QualificationExecutionOutcome
    expected_functional_outcome: QualificationFunctionalOutcome
    expected_check_results: tuple[QualificationCheckExpectation, ...]
    expected_first_proven_failed_check: FunctionalDiagnosticCheckId | None = None
    expected_operator_outcome: FunctionalOperatorOutcomeStatus | None = None
    include_output_relation: bool = False
    include_validation: bool = True


class QualificationComparisonResult(StrEnum):
    MATCH = "match"
    MISMATCH = "mismatch"


@dataclass(frozen=True, slots=True)
class QualificationCheckMismatch:
    check_id: FunctionalDiagnosticCheckId
    expected_status: FunctionalDiagnosticCheckStatus
    actual_status: FunctionalDiagnosticCheckStatus


@dataclass(frozen=True, slots=True)
class QualificationComparisonMismatch:
    field: str
    expected: str
    actual: str


@dataclass(frozen=True, slots=True)
class QualificationCaseComparison:
    case_id: str
    result: QualificationComparisonResult
    check_mismatches: tuple[QualificationCheckMismatch, ...] = ()
    field_mismatches: tuple[QualificationComparisonMismatch, ...] = ()


__all__ = [
    "QualificationCaseComparison",
    "QualificationCaseExpectation",
    "QualificationCheckExpectation",
    "QualificationCheckMismatch",
    "QualificationComparisonMismatch",
    "QualificationComparisonResult",
    "QualificationExecutionOutcome",
    "QualificationFunctionalOutcome",
]
