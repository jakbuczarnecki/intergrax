# © Artur Czarnecki. All rights reserved.

"""Unit tests for functional diagnostic qualification comparator."""

from __future__ import annotations

import pytest

from intergrax.core.qualification.functional_diagnostic_comparator import compare_qualification_case
from intergrax.core.qualification.functional_diagnostic_expectation import (
    QualificationCaseExpectation,
    QualificationCheckExpectation,
    QualificationComparisonResult,
    QualificationExecutionOutcome,
    QualificationFunctionalOutcome,
)
from intergrax.runtime.diagnostics.diagnostic_assessment import DiagnosticAssessment
from intergrax.runtime.diagnostics.diagnostic_assessment_composer import DiagnosticAssessmentComposer
from intergrax.runtime.diagnostics.functional_diagnostic_analysis import (
    FunctionalDiagnosticAnalysis,
    FunctionalDiagnosticCheckResult,
    FunctionalDiagnosticCheckStatus,
)
from intergrax.runtime.diagnostics.functional_diagnostic_identity import (
    FunctionalDiagnosticCheckId,
    FunctionalDiagnosticSpecificationId,
)
from intergrax.runtime.diagnostics.functional_diagnostic_specification import (
    FunctionalDiagnosticCheck,
    FunctionalDiagnosticRequirement,
    FunctionalDiagnosticRequirementKind,
    FunctionalDiagnosticSpecification,
    OperationOutcomeStatusRequirement,
)
from intergrax.runtime.diagnostics.functional_evidence import PipelineOperationStatus
from intergrax.runtime.diagnostics.functional_operator_projection import FunctionalOperatorOutcomeStatus
from intergrax.contracts.execution_identity import mint_run_id, mint_task_id

pytestmark = pytest.mark.unit

_CHECK = FunctionalDiagnosticCheckId("fdcheck_00000000000000000000000000000001")
_SPEC = FunctionalDiagnosticSpecificationId("fdspec_000000000000000000000000c1aa0001")


def _analysis(*statuses: FunctionalDiagnosticCheckStatus) -> FunctionalDiagnosticAnalysis:
    results = tuple(
        FunctionalDiagnosticCheckResult(
            check_id=_CHECK,
            status=status,
            factual_claim="claim",
            supporting_evidence_refs=(),
            limitations=(),
        )
        for status in statuses
    )
    return FunctionalDiagnosticAnalysis(
        tenant_id="tenant-a",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=None,
        specification_id=_SPEC,
        specification_version=1,
        check_results=results,
        first_proven_failure=_CHECK if statuses and statuses[0] is FunctionalDiagnosticCheckStatus.PROVEN_FAIL else None,
        limitations=(),
    )


def test_comparator_reports_match_when_expectations_align() -> None:
    expectation = QualificationCaseExpectation(
        case_id="unit-healthy",
        expected_execution_outcome=QualificationExecutionOutcome.COMPLETED,
        expected_functional_outcome=QualificationFunctionalOutcome.PASSED,
        expected_check_results=(
            QualificationCheckExpectation(_CHECK, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        ),
        expected_operator_outcome=FunctionalOperatorOutcomeStatus.PROVEN_FUNCTIONAL_SUCCESS,
    )
    analysis = _analysis(FunctionalDiagnosticCheckStatus.PROVEN_PASS)
    lifecycle = DiagnosticAssessment(
        tenant_id="tenant-a",
        task_id=analysis.task_id,
        run_id=analysis.run_id,
        findings=(),
        limitations=(),
    )
    operator = DiagnosticAssessmentComposer().compose(lifecycle, analysis)
    result = compare_qualification_case(
        expectation,
        actual_execution_outcome=QualificationExecutionOutcome.COMPLETED,
        actual_functional_outcome=QualificationFunctionalOutcome.PASSED,
        analysis=analysis,
        operator_assessment=operator,
    )
    assert result.result is QualificationComparisonResult.MATCH


def test_comparator_reports_mismatch_on_failed_check() -> None:
    expectation = QualificationCaseExpectation(
        case_id="unit-failure",
        expected_execution_outcome=QualificationExecutionOutcome.COMPLETED,
        expected_functional_outcome=QualificationFunctionalOutcome.FAILED,
        expected_check_results=(
            QualificationCheckExpectation(_CHECK, FunctionalDiagnosticCheckStatus.PROVEN_FAIL),
        ),
        expected_first_proven_failed_check=_CHECK,
        expected_operator_outcome=FunctionalOperatorOutcomeStatus.PROVEN_FUNCTIONAL_FAILURE,
    )
    analysis = _analysis(FunctionalDiagnosticCheckStatus.PROVEN_PASS)
    result = compare_qualification_case(
        expectation,
        actual_execution_outcome=QualificationExecutionOutcome.COMPLETED,
        actual_functional_outcome=QualificationFunctionalOutcome.FAILED,
        analysis=analysis,
        operator_assessment=None,
    )
    assert result.result is QualificationComparisonResult.MISMATCH
    assert result.check_mismatches


def test_comparator_skips_functional_outcome_when_disabled() -> None:
    expectation = QualificationCaseExpectation(
        case_id="unit-inconclusive",
        expected_execution_outcome=QualificationExecutionOutcome.COMPLETED,
        expected_functional_outcome=QualificationFunctionalOutcome.PASSED,
        expected_check_results=(
            QualificationCheckExpectation(_CHECK, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        ),
        compare_functional_outcome=False,
    )
    analysis = _analysis(FunctionalDiagnosticCheckStatus.PROVEN_PASS)
    result = compare_qualification_case(
        expectation,
        actual_execution_outcome=QualificationExecutionOutcome.COMPLETED,
        actual_functional_outcome=QualificationFunctionalOutcome.FAILED,
        analysis=analysis,
        operator_assessment=None,
    )
    assert result.result is QualificationComparisonResult.MATCH
