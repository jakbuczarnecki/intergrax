# © Artur Czarnecki. All rights reserved.

"""Q2 qualification comparator and metrics semantics (DIAG-FUNCTIONAL-Q2-R1)."""

from __future__ import annotations

import pytest

from intergrax.core.qualification.functional_diagnostic_comparator import compare_qualification_case
from intergrax.core.qualification.functional_diagnostic_expectation import (
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
from intergrax.runtime.diagnostics.functional_diagnostic_identity import FunctionalDiagnosticSpecificationId
from intergrax.runtime.diagnostics.functional_operator_projection import FunctionalOperatorOutcomeStatus
from intergrax.runtime.diagnostics.specifications.q2_tool_selection_functional_diagnostic_specification import (
    CHECK_Q2_CANDIDATES,
    CHECK_Q2_INVOCATION,
    CHECK_Q2_SELECTION,
    CHECK_Q2_VALIDATION,
    Q2_TOOL_SPECIFICATION_ID,
)
from tests.system.functional_diagnostics_q2.cases import (
    Q2_B_WRONG_TOOL,
    Q2_C_INVOKE_FAILURE,
    Q2_D_VALIDATION_FAILURE,
    Q2_E_MISSING_EVIDENCE,
)
from tests.system.functional_diagnostics_q2.runner import (
    _functional_failure_detected,
    _stage_matches,
    QualificationRunRecord,
    QualificationReport,
)
from intergrax.contracts.execution_identity import mint_run_id, mint_task_id

pytestmark = pytest.mark.unit

_SPEC = Q2_TOOL_SPECIFICATION_ID


def _q2_analysis(
  *,
  candidate: FunctionalDiagnosticCheckStatus,
  selection: FunctionalDiagnosticCheckStatus,
  invocation: FunctionalDiagnosticCheckStatus,
  validation: FunctionalDiagnosticCheckStatus,
  first_failure: str | None,
) -> FunctionalDiagnosticAnalysis:
    task_id = mint_task_id()
    run_id = mint_run_id()
    results = (
        FunctionalDiagnosticCheckResult(
            check_id=CHECK_Q2_CANDIDATES,
            status=candidate,
            factual_claim="c",
            supporting_evidence_refs=(),
            limitations=(),
        ),
        FunctionalDiagnosticCheckResult(
            check_id=CHECK_Q2_SELECTION,
            status=selection,
            factual_claim="s",
            supporting_evidence_refs=(),
            limitations=(),
        ),
        FunctionalDiagnosticCheckResult(
            check_id=CHECK_Q2_INVOCATION,
            status=invocation,
            factual_claim="i",
            supporting_evidence_refs=(),
            limitations=(),
        ),
        FunctionalDiagnosticCheckResult(
            check_id=CHECK_Q2_VALIDATION,
            status=validation,
            factual_claim="v",
            supporting_evidence_refs=(),
            limitations=(),
        ),
    )
    return FunctionalDiagnosticAnalysis(
        tenant_id="tenant-q2",
        task_id=task_id,
        run_id=run_id,
        attempt_id=None,
        specification_id=_SPEC,
        specification_version=1,
        check_results=results,
        first_proven_failure=first_failure,
        limitations=(),
    )


def _operator(analysis: FunctionalDiagnosticAnalysis) -> object:
    lifecycle = DiagnosticAssessment(
        tenant_id=analysis.tenant_id,
        task_id=analysis.task_id,
        run_id=analysis.run_id,
        findings=(),
        limitations=(),
    )
    return DiagnosticAssessmentComposer().compose(lifecycle_assessment=lifecycle, functional_analysis=analysis)


def test_wrong_tool_vector_matches_when_invocation_succeeds() -> None:
    analysis = _q2_analysis(
        candidate=FunctionalDiagnosticCheckStatus.PROVEN_PASS,
        selection=FunctionalDiagnosticCheckStatus.PROVEN_FAIL,
        invocation=FunctionalDiagnosticCheckStatus.PROVEN_PASS,
        validation=FunctionalDiagnosticCheckStatus.PROVEN_FAIL,
        first_failure=CHECK_Q2_SELECTION,
    )
    operator = _operator(analysis)
    result = compare_qualification_case(
        Q2_B_WRONG_TOOL,
        actual_execution_outcome=QualificationExecutionOutcome.COMPLETED,
        actual_functional_outcome=QualificationFunctionalOutcome.FAILED,
        analysis=analysis,
        operator_assessment=operator,
    )
    assert result.result is QualificationComparisonResult.MATCH


def test_missing_evidence_matches_without_functional_outcome_comparison() -> None:
    analysis = _q2_analysis(
        candidate=FunctionalDiagnosticCheckStatus.PROVEN_PASS,
        selection=FunctionalDiagnosticCheckStatus.INSUFFICIENT_EVIDENCE,
        invocation=FunctionalDiagnosticCheckStatus.PROVEN_PASS,
        validation=FunctionalDiagnosticCheckStatus.NOT_EVALUATED,
        first_failure=None,
    )
    operator = _operator(analysis)
    result = compare_qualification_case(
        Q2_E_MISSING_EVIDENCE,
        actual_execution_outcome=QualificationExecutionOutcome.COMPLETED,
        actual_functional_outcome=QualificationFunctionalOutcome.FAILED,
        analysis=analysis,
        operator_assessment=operator,
    )
    assert result.result is QualificationComparisonResult.MATCH


def test_q2_c_invoke_failure_baseline_preserved() -> None:
    analysis = _q2_analysis(
        candidate=FunctionalDiagnosticCheckStatus.PROVEN_PASS,
        selection=FunctionalDiagnosticCheckStatus.PROVEN_PASS,
        invocation=FunctionalDiagnosticCheckStatus.PROVEN_FAIL,
        validation=FunctionalDiagnosticCheckStatus.PROVEN_FAIL,
        first_failure=CHECK_Q2_INVOCATION,
    )
    operator = _operator(analysis)
    result = compare_qualification_case(
        Q2_C_INVOKE_FAILURE,
        actual_execution_outcome=QualificationExecutionOutcome.COMPLETED,
        actual_functional_outcome=QualificationFunctionalOutcome.FAILED,
        analysis=analysis,
        operator_assessment=operator,
    )
    assert result.result is QualificationComparisonResult.MATCH


def test_q2_d_validation_failure_baseline_preserved() -> None:
    analysis = _q2_analysis(
        candidate=FunctionalDiagnosticCheckStatus.PROVEN_PASS,
        selection=FunctionalDiagnosticCheckStatus.PROVEN_PASS,
        invocation=FunctionalDiagnosticCheckStatus.PROVEN_PASS,
        validation=FunctionalDiagnosticCheckStatus.PROVEN_FAIL,
        first_failure=CHECK_Q2_VALIDATION,
    )
    operator = _operator(analysis)
    result = compare_qualification_case(
        Q2_D_VALIDATION_FAILURE,
        actual_execution_outcome=QualificationExecutionOutcome.COMPLETED,
        actual_functional_outcome=QualificationFunctionalOutcome.FAILED,
        analysis=analysis,
        operator_assessment=operator,
    )
    assert result.result is QualificationComparisonResult.MATCH


def _minimal_record(
    *,
    case_id: str,
    comparison_result: QualificationComparisonResult,
    functional_outcome: QualificationFunctionalOutcome,
    operator_outcome: str | None,
    diag_first_failed_check: str | None,
    repeat_group: str | None = None,
) -> QualificationRunRecord:
    from intergrax.core.qualification.functional_diagnostic_expectation import QualificationCaseComparison
    from tests.system.functional_diagnostics_q2.runner import EvidenceFidelitySnapshot

    fidelity = EvidenceFidelitySnapshot(
        provider_candidate_refs=(),
        actual_selected_tool=None,
        emitted_selected_tool=None,
        actual_invoke_succeeded=None,
        emitted_invoke_succeeded=None,
        candidate_fidelity_match=True,
        selection_fidelity_match=True,
        invocation_fidelity_match=True,
        validation_fidelity_match=True,
        identity_fidelity_match=True,
        failure_injection_layer=None,
    )
    return QualificationRunRecord(
        case_id=case_id,
        task_id="task_test",
        run_id="run_test",
        execution_outcome=QualificationExecutionOutcome.COMPLETED,
        functional_outcome=functional_outcome,
        comparison=QualificationCaseComparison(case_id=case_id, result=comparison_result),
        evidence_fidelity=fidelity,
        diag_first_failed_check=diag_first_failed_check,
        operator_outcome=operator_outcome,
        available_tools=(),
        expected_tool="tool:workspace.search",
        actual_tool=None,
        invocation_status=None,
        repeat_group=repeat_group,
    )


def test_stage_match_uses_first_proven_failure_not_full_case_match() -> None:
    record = _minimal_record(
        case_id="Q2-B",
        comparison_result=QualificationComparisonResult.MISMATCH,
        functional_outcome=QualificationFunctionalOutcome.FAILED,
        operator_outcome=FunctionalOperatorOutcomeStatus.PROVEN_FUNCTIONAL_FAILURE.value,
        diag_first_failed_check=str(CHECK_Q2_SELECTION),
    )
    assert _stage_matches(record, Q2_B_WRONG_TOOL)


def test_false_negative_requires_missing_functional_failure_detection() -> None:
    detected = _minimal_record(
        case_id="Q2-B",
        comparison_result=QualificationComparisonResult.MISMATCH,
        functional_outcome=QualificationFunctionalOutcome.FAILED,
        operator_outcome=FunctionalOperatorOutcomeStatus.PROVEN_FUNCTIONAL_FAILURE.value,
        diag_first_failed_check=str(CHECK_Q2_SELECTION),
    )
    missed = _minimal_record(
        case_id="Q2-B",
        comparison_result=QualificationComparisonResult.MISMATCH,
        functional_outcome=QualificationFunctionalOutcome.FAILED,
        operator_outcome=FunctionalOperatorOutcomeStatus.INCONCLUSIVE.value,
        diag_first_failed_check=None,
    )
    assert _functional_failure_detected(detected)
    assert not _functional_failure_detected(missed)


def test_qualification_report_accepts_metric_fields() -> None:
    report = QualificationReport(
        verdict="FAILED",
        total_cases=1,
        matched_cases=0,
        mismatched_cases=1,
        false_positive_cases=0,
        false_negative_cases=0,
        inconclusive_correct_cases=0,
        stage_accuracy_percent=100.0,
        inconclusive_accuracy_percent=0.0,
        repeatability_pass=True,
        records=(),
        stage_matched_cases=1,
        functional_failure_detected_cases=1,
        functional_failure_ground_truth_cases=1,
    )
    assert report.stage_matched_cases == 1
