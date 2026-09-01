# © Artur Czarnecki. All rights reserved.

"""Ground-truth expectations for DIAG-FUNCTIONAL-Q1 mandatory matrix."""

from __future__ import annotations

from intergrax.core.qualification.functional_diagnostic_expectation import (
    QualificationCaseExpectation,
    QualificationCheckExpectation,
    QualificationExecutionOutcome,
    QualificationFunctionalOutcome,
)
from intergrax.runtime.diagnostics.functional_diagnostic_analysis import FunctionalDiagnosticCheckStatus
from intergrax.runtime.diagnostics.functional_operator_projection import FunctionalOperatorOutcomeStatus
from intergrax.runtime.diagnostics.specifications.c1_rag_functional_diagnostic_specification import (
    CHECK_C1_CANDIDATES,
    CHECK_C1_OUTPUT_RELATION,
    CHECK_C1_RETRIEVAL_OPERATION,
    CHECK_C1_SELECTION,
    CHECK_C1_VALIDATION,
)

HEALTHY_QUERY = "When did Incident Orion occur?"
_SELECTION_FAILURE_QUERY = (
    "placeholder decoy date qualification selection failure injection outdated operations note"
)
_SYNTHESIS_FAILURE_DRAFT = "# Draft\n\nIncident Orion occurred on 2025-01-01.\n"


def _healthy_checks() -> tuple[QualificationCheckExpectation, ...]:
    return (
        QualificationCheckExpectation(CHECK_C1_RETRIEVAL_OPERATION, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_C1_CANDIDATES, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_C1_SELECTION, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_C1_VALIDATION, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
    )


def _selection_failure_checks() -> tuple[QualificationCheckExpectation, ...]:
    return (
        QualificationCheckExpectation(CHECK_C1_RETRIEVAL_OPERATION, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_C1_CANDIDATES, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_C1_SELECTION, FunctionalDiagnosticCheckStatus.PROVEN_FAIL),
        QualificationCheckExpectation(CHECK_C1_VALIDATION, FunctionalDiagnosticCheckStatus.PROVEN_FAIL),
    )


def _synthesis_failure_checks() -> tuple[QualificationCheckExpectation, ...]:
    return (
        QualificationCheckExpectation(CHECK_C1_RETRIEVAL_OPERATION, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_C1_CANDIDATES, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_C1_SELECTION, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_C1_OUTPUT_RELATION, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_C1_VALIDATION, FunctionalDiagnosticCheckStatus.PROVEN_FAIL),
    )


def _missing_evidence_checks() -> tuple[QualificationCheckExpectation, ...]:
    return (
        QualificationCheckExpectation(CHECK_C1_RETRIEVAL_OPERATION, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_C1_CANDIDATES, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_C1_SELECTION, FunctionalDiagnosticCheckStatus.INSUFFICIENT_EVIDENCE),
    )


Q1_A_HEALTHY = QualificationCaseExpectation(
    case_id="Q1-A",
    expected_execution_outcome=QualificationExecutionOutcome.COMPLETED,
    expected_functional_outcome=QualificationFunctionalOutcome.PASSED,
    expected_check_results=_healthy_checks(),
    expected_first_proven_failed_check=None,
    expected_operator_outcome=FunctionalOperatorOutcomeStatus.PROVEN_FUNCTIONAL_SUCCESS,
)

Q1_B_SELECTION_FAILURE = QualificationCaseExpectation(
    case_id="Q1-B",
    expected_execution_outcome=QualificationExecutionOutcome.COMPLETED,
    expected_functional_outcome=QualificationFunctionalOutcome.FAILED,
    expected_check_results=_selection_failure_checks(),
    expected_first_proven_failed_check=CHECK_C1_SELECTION,
    expected_operator_outcome=FunctionalOperatorOutcomeStatus.PROVEN_FUNCTIONAL_FAILURE,
)

Q1_C_SYNTHESIS_FAILURE = QualificationCaseExpectation(
    case_id="Q1-C",
    expected_execution_outcome=QualificationExecutionOutcome.COMPLETED,
    expected_functional_outcome=QualificationFunctionalOutcome.FAILED,
    expected_check_results=_synthesis_failure_checks(),
    expected_first_proven_failed_check=CHECK_C1_VALIDATION,
    expected_operator_outcome=FunctionalOperatorOutcomeStatus.PROVEN_FUNCTIONAL_FAILURE,
    include_output_relation=True,
)

Q1_D_MISSING_EVIDENCE = QualificationCaseExpectation(
    case_id="Q1-D",
    expected_execution_outcome=QualificationExecutionOutcome.COMPLETED,
    expected_functional_outcome=QualificationFunctionalOutcome.PASSED,
    expected_check_results=_missing_evidence_checks(),
    expected_first_proven_failed_check=None,
    expected_operator_outcome=FunctionalOperatorOutcomeStatus.INCONCLUSIVE,
    include_validation=False,
)

Q1_E_HEALTHY = QualificationCaseExpectation(
    case_id="Q1-E-A",
    expected_execution_outcome=QualificationExecutionOutcome.COMPLETED,
    expected_functional_outcome=QualificationFunctionalOutcome.PASSED,
    expected_check_results=_healthy_checks(),
    expected_first_proven_failed_check=None,
    expected_operator_outcome=FunctionalOperatorOutcomeStatus.PROVEN_FUNCTIONAL_SUCCESS,
)

Q1_E_FAILURE = QualificationCaseExpectation(
    case_id="Q1-E-B",
    expected_execution_outcome=QualificationExecutionOutcome.COMPLETED,
    expected_functional_outcome=QualificationFunctionalOutcome.FAILED,
    expected_check_results=_selection_failure_checks(),
    expected_first_proven_failed_check=CHECK_C1_SELECTION,
    expected_operator_outcome=FunctionalOperatorOutcomeStatus.PROVEN_FUNCTIONAL_FAILURE,
)

Q1_H_HISTORICAL_WRONG_DATE = QualificationCaseExpectation(
    case_id="Q1-H",
    expected_execution_outcome=QualificationExecutionOutcome.COMPLETED,
    expected_functional_outcome=QualificationFunctionalOutcome.PASSED,
    expected_check_results=_healthy_checks(),
    expected_first_proven_failed_check=None,
    expected_operator_outcome=FunctionalOperatorOutcomeStatus.PROVEN_FUNCTIONAL_SUCCESS,
)

MANDATORY_CASES: tuple[QualificationCaseExpectation, ...] = (
    Q1_A_HEALTHY,
    Q1_B_SELECTION_FAILURE,
    Q1_C_SYNTHESIS_FAILURE,
    Q1_D_MISSING_EVIDENCE,
)


def case_metadata(case: QualificationCaseExpectation) -> dict[str, object]:
    base: dict[str, object] = {
        "qualification_case_id": case.case_id,
        "query": HEALTHY_QUERY,
    }
    if case.case_id in {"Q1-B", "Q1-E-B", "Q1-F"}:
        base["query"] = _SELECTION_FAILURE_QUERY
        base["qualification_failure_injection_layer"] = "retrieval_ranking_query"
    if case.case_id == "Q1-D":
        base["qualification_suppress_functional_evidence_kinds"] = ["selection"]
    if case.case_id == "Q1-C":
        base["draft"] = _SYNTHESIS_FAILURE_DRAFT
        base["qualification_failure_injection_layer"] = "synthesis_input_draft"
        base["shadow_workspace"] = True
        base["output_name"] = "q1-synthesis-draft.md"
    return base


__all__ = [
    "HEALTHY_QUERY",
    "MANDATORY_CASES",
    "Q1_A_HEALTHY",
    "Q1_B_SELECTION_FAILURE",
    "Q1_C_SYNTHESIS_FAILURE",
    "Q1_D_MISSING_EVIDENCE",
    "Q1_E_FAILURE",
    "Q1_E_HEALTHY",
    "Q1_H_HISTORICAL_WRONG_DATE",
    "case_metadata",
]
