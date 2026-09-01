# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Deterministic ground-truth comparator for functional diagnostic qualification (DIAG-FUNCTIONAL-Q1)."""

from __future__ import annotations

from intergrax.core.qualification.functional_diagnostic_expectation import (
    QualificationCaseComparison,
    QualificationCaseExpectation,
    QualificationCheckMismatch,
    QualificationComparisonMismatch,
    QualificationComparisonResult,
    QualificationExecutionOutcome,
    QualificationFunctionalOutcome,
)
from intergrax.runtime.diagnostics.diagnostic_assessment_composer import OperatorDiagnosticAssessment
from intergrax.runtime.diagnostics.functional_diagnostic_analysis import (
    FunctionalDiagnosticAnalysis,
    FunctionalDiagnosticCheckStatus,
)
from intergrax.runtime.diagnostics.functional_operator_projection import FunctionalOperatorOutcomeStatus


def compare_qualification_case(
    expectation: QualificationCaseExpectation,
    *,
    actual_execution_outcome: QualificationExecutionOutcome,
    actual_functional_outcome: QualificationFunctionalOutcome,
    analysis: FunctionalDiagnosticAnalysis | None,
    operator_assessment: OperatorDiagnosticAssessment | None,
) -> QualificationCaseComparison:
    field_mismatches: list[QualificationComparisonMismatch] = []
    if actual_execution_outcome is not expectation.expected_execution_outcome:
        field_mismatches.append(
            QualificationComparisonMismatch(
                field="execution_outcome",
                expected=expectation.expected_execution_outcome.value,
                actual=actual_execution_outcome.value,
            ),
        )
    if expectation.compare_functional_outcome:
        if actual_functional_outcome is not expectation.expected_functional_outcome:
            field_mismatches.append(
                QualificationComparisonMismatch(
                    field="functional_outcome",
                    expected=expectation.expected_functional_outcome.value,
                    actual=actual_functional_outcome.value,
                ),
            )

    check_mismatches: list[QualificationCheckMismatch] = []
    if analysis is None:
        for item in expectation.expected_check_results:
            check_mismatches.append(
                QualificationCheckMismatch(
                    check_id=item.check_id,
                    expected_status=item.expected_status,
                    actual_status=FunctionalDiagnosticCheckStatus.NOT_EVALUATED,
                ),
            )
    else:
        actual_by_id = {item.check_id: item.status for item in analysis.check_results}
        for item in expectation.expected_check_results:
            actual_status = actual_by_id.get(item.check_id, FunctionalDiagnosticCheckStatus.NOT_EVALUATED)
            if actual_status is not item.expected_status:
                check_mismatches.append(
                    QualificationCheckMismatch(
                        check_id=item.check_id,
                        expected_status=item.expected_status,
                        actual_status=actual_status,
                    ),
                )
        expected_first = expectation.expected_first_proven_failed_check
        if expected_first != analysis.first_proven_failure:
            field_mismatches.append(
                QualificationComparisonMismatch(
                    field="first_proven_failed_check",
                    expected=str(expected_first or ""),
                    actual=str(analysis.first_proven_failure or ""),
                ),
            )

    if expectation.expected_operator_outcome is not None:
        actual_operator = (
            operator_assessment.functional_projection.outcome_status
            if operator_assessment is not None and operator_assessment.functional_projection is not None
            else FunctionalOperatorOutcomeStatus.INCONCLUSIVE
        )
        if actual_operator is not expectation.expected_operator_outcome:
            field_mismatches.append(
                QualificationComparisonMismatch(
                    field="operator_outcome_status",
                    expected=expectation.expected_operator_outcome.value,
                    actual=actual_operator.value,
                ),
            )

    result = (
        QualificationComparisonResult.MATCH
        if not check_mismatches and not field_mismatches
        else QualificationComparisonResult.MISMATCH
    )
    return QualificationCaseComparison(
        case_id=expectation.case_id,
        result=result,
        check_mismatches=tuple(check_mismatches),
        field_mismatches=tuple(field_mismatches),
    )


__all__ = ["compare_qualification_case"]
