# © Artur Czarnecki. All rights reserved.

"""Ground-truth expectations for DIAG-FUNCTIONAL-Q3 mandatory matrix."""

from __future__ import annotations

from intergrax.core.qualification.functional_diagnostic_expectation import (
    QualificationCaseExpectation,
    QualificationCheckExpectation,
    QualificationExecutionOutcome,
    QualificationFunctionalOutcome,
)
from intergrax.runtime.diagnostics.functional_diagnostic_analysis import FunctionalDiagnosticCheckStatus
from intergrax.runtime.diagnostics.functional_operator_projection import FunctionalOperatorOutcomeStatus
from intergrax.runtime.diagnostics.specifications.q3_web_search_functional_diagnostic_specification import (
    CHECK_Q3_CANDIDATES,
    CHECK_Q3_EXTRACTION_RELATION,
    CHECK_Q3_EXTRACTION_VALIDATION,
    CHECK_Q3_FINAL,
    CHECK_Q3_QUERY,
    CHECK_Q3_SEARCH,
    CHECK_Q3_SELECTION,
)
from tests.system.functional_diagnostics_q3.oracle import HEALTHY_TASK

_REPEAT_CASE_ID = "Q3-H"


def _healthy_checks() -> tuple[QualificationCheckExpectation, ...]:
    return (
        QualificationCheckExpectation(CHECK_Q3_QUERY, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_Q3_SEARCH, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_Q3_CANDIDATES, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_Q3_SELECTION, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(
            CHECK_Q3_EXTRACTION_RELATION,
            FunctionalDiagnosticCheckStatus.PROVEN_PASS,
        ),
        QualificationCheckExpectation(
            CHECK_Q3_EXTRACTION_VALIDATION,
            FunctionalDiagnosticCheckStatus.PROVEN_PASS,
        ),
        QualificationCheckExpectation(CHECK_Q3_FINAL, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
    )


def _bad_query_checks() -> tuple[QualificationCheckExpectation, ...]:
    """Wrong query localizes at QUERY; downstream varies with provider result mix."""
    return (
        QualificationCheckExpectation(CHECK_Q3_QUERY, FunctionalDiagnosticCheckStatus.PROVEN_FAIL),
        QualificationCheckExpectation(CHECK_Q3_SEARCH, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_Q3_CANDIDATES, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
    )


def _wrong_source_checks() -> tuple[QualificationCheckExpectation, ...]:
    """Wrong source localizes at SELECTION; extraction validation is source-agnostic."""
    return (
        QualificationCheckExpectation(CHECK_Q3_QUERY, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_Q3_SEARCH, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_Q3_CANDIDATES, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_Q3_SELECTION, FunctionalDiagnosticCheckStatus.PROVEN_FAIL),
        QualificationCheckExpectation(
            CHECK_Q3_EXTRACTION_RELATION,
            FunctionalDiagnosticCheckStatus.PROVEN_PASS,
        ),
    )


def _bad_extraction_checks() -> tuple[QualificationCheckExpectation, ...]:
    return (
        QualificationCheckExpectation(CHECK_Q3_QUERY, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_Q3_SEARCH, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_Q3_CANDIDATES, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_Q3_SELECTION, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(
            CHECK_Q3_EXTRACTION_RELATION,
            FunctionalDiagnosticCheckStatus.PROVEN_PASS,
        ),
        QualificationCheckExpectation(
            CHECK_Q3_EXTRACTION_VALIDATION,
            FunctionalDiagnosticCheckStatus.PROVEN_FAIL,
        ),
        QualificationCheckExpectation(CHECK_Q3_FINAL, FunctionalDiagnosticCheckStatus.PROVEN_FAIL),
    )


def _bad_synthesis_checks() -> tuple[QualificationCheckExpectation, ...]:
    return (
        QualificationCheckExpectation(CHECK_Q3_QUERY, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_Q3_SEARCH, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_Q3_CANDIDATES, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_Q3_SELECTION, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(
            CHECK_Q3_EXTRACTION_RELATION,
            FunctionalDiagnosticCheckStatus.PROVEN_PASS,
        ),
        QualificationCheckExpectation(
            CHECK_Q3_EXTRACTION_VALIDATION,
            FunctionalDiagnosticCheckStatus.PROVEN_PASS,
        ),
        QualificationCheckExpectation(CHECK_Q3_FINAL, FunctionalDiagnosticCheckStatus.PROVEN_FAIL),
    )


def _missing_selection_checks() -> tuple[QualificationCheckExpectation, ...]:
    return (
        QualificationCheckExpectation(CHECK_Q3_QUERY, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_Q3_SEARCH, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_Q3_CANDIDATES, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_Q3_SELECTION, FunctionalDiagnosticCheckStatus.INSUFFICIENT_EVIDENCE),
        QualificationCheckExpectation(
            CHECK_Q3_EXTRACTION_RELATION,
            FunctionalDiagnosticCheckStatus.PROVEN_PASS,
        ),
        QualificationCheckExpectation(
            CHECK_Q3_EXTRACTION_VALIDATION,
            FunctionalDiagnosticCheckStatus.PROVEN_PASS,
        ),
    )


Q3_A_HEALTHY = QualificationCaseExpectation(
    case_id="Q3-A",
    expected_execution_outcome=QualificationExecutionOutcome.COMPLETED,
    expected_functional_outcome=QualificationFunctionalOutcome.PASSED,
    expected_check_results=_healthy_checks(),
    expected_first_proven_failed_check=None,
    expected_operator_outcome=FunctionalOperatorOutcomeStatus.PROVEN_FUNCTIONAL_SUCCESS,
)

Q3_B_BAD_QUERY = QualificationCaseExpectation(
    case_id="Q3-B",
    expected_execution_outcome=QualificationExecutionOutcome.COMPLETED,
    expected_functional_outcome=QualificationFunctionalOutcome.FAILED,
    expected_check_results=_bad_query_checks(),
    expected_first_proven_failed_check=CHECK_Q3_QUERY,
    expected_operator_outcome=FunctionalOperatorOutcomeStatus.PROVEN_FUNCTIONAL_FAILURE,
)

Q3_C_WRONG_SOURCE = QualificationCaseExpectation(
    case_id="Q3-C",
    expected_execution_outcome=QualificationExecutionOutcome.COMPLETED,
    expected_functional_outcome=QualificationFunctionalOutcome.FAILED,
    expected_check_results=_wrong_source_checks(),
    expected_first_proven_failed_check=CHECK_Q3_SELECTION,
    expected_operator_outcome=FunctionalOperatorOutcomeStatus.PROVEN_FUNCTIONAL_FAILURE,
)

Q3_D_BAD_EXTRACTION = QualificationCaseExpectation(
    case_id="Q3-D",
    expected_execution_outcome=QualificationExecutionOutcome.COMPLETED,
    expected_functional_outcome=QualificationFunctionalOutcome.FAILED,
    expected_check_results=_bad_extraction_checks(),
    expected_first_proven_failed_check=CHECK_Q3_EXTRACTION_VALIDATION,
    expected_operator_outcome=FunctionalOperatorOutcomeStatus.PROVEN_FUNCTIONAL_FAILURE,
)

Q3_E_BAD_SYNTHESIS = QualificationCaseExpectation(
    case_id="Q3-E",
    expected_execution_outcome=QualificationExecutionOutcome.COMPLETED,
    expected_functional_outcome=QualificationFunctionalOutcome.FAILED,
    expected_check_results=_bad_synthesis_checks(),
    expected_first_proven_failed_check=CHECK_Q3_FINAL,
    expected_operator_outcome=FunctionalOperatorOutcomeStatus.PROVEN_FUNCTIONAL_FAILURE,
)

Q3_F_MISSING_SELECTION = QualificationCaseExpectation(
    case_id="Q3-F",
    expected_execution_outcome=QualificationExecutionOutcome.COMPLETED,
    expected_functional_outcome=QualificationFunctionalOutcome.FAILED,
    expected_check_results=_missing_selection_checks(),
    expected_first_proven_failed_check=None,
    expected_operator_outcome=FunctionalOperatorOutcomeStatus.INCONCLUSIVE,
    include_validation=False,
    compare_functional_outcome=False,
)

Q3_G_HEALTHY = QualificationCaseExpectation(
    case_id="Q3-G-A",
    expected_execution_outcome=QualificationExecutionOutcome.COMPLETED,
    expected_functional_outcome=QualificationFunctionalOutcome.PASSED,
    expected_check_results=_healthy_checks(),
    expected_first_proven_failed_check=None,
    expected_operator_outcome=FunctionalOperatorOutcomeStatus.PROVEN_FUNCTIONAL_SUCCESS,
)

Q3_G_WRONG_SOURCE = QualificationCaseExpectation(
    case_id="Q3-G-B",
    expected_execution_outcome=QualificationExecutionOutcome.COMPLETED,
    expected_functional_outcome=QualificationFunctionalOutcome.FAILED,
    expected_check_results=_wrong_source_checks(),
    expected_first_proven_failed_check=CHECK_Q3_SELECTION,
    expected_operator_outcome=FunctionalOperatorOutcomeStatus.PROVEN_FUNCTIONAL_FAILURE,
)

MANDATORY_CASES: tuple[QualificationCaseExpectation, ...] = (
    Q3_A_HEALTHY,
    Q3_B_BAD_QUERY,
    Q3_C_WRONG_SOURCE,
    Q3_D_BAD_EXTRACTION,
    Q3_E_BAD_SYNTHESIS,
    Q3_F_MISSING_SELECTION,
)


def case_metadata(case: QualificationCaseExpectation) -> dict[str, object]:
    base: dict[str, object] = {
        "qualification_case_id": case.case_id,
        "qualification_task_message": HEALTHY_TASK,
        "qualification_search_limit": 8,
    }
    if case.case_id == "Q3-B":
        base["qualification_failure_injection_layer"] = "query_construction_bias"
    if case.case_id in {"Q3-C", "Q3-G-B", _REPEAT_CASE_ID}:
        base["qualification_failure_injection_layer"] = "source_selection_bias"
    if case.case_id == "Q3-D":
        base["qualification_failure_injection_layer"] = "extraction_bias"
    if case.case_id == "Q3-E":
        base["qualification_failure_injection_layer"] = "synthesis_bias"
    if case.case_id == "Q3-F":
        base["qualification_suppress_functional_evidence_kinds"] = ["selection"]
    return base


__all__ = [
    "HEALTHY_TASK",
    "MANDATORY_CASES",
    "Q3_A_HEALTHY",
    "Q3_B_BAD_QUERY",
    "Q3_C_WRONG_SOURCE",
    "Q3_D_BAD_EXTRACTION",
    "Q3_E_BAD_SYNTHESIS",
    "Q3_F_MISSING_SELECTION",
    "Q3_G_HEALTHY",
    "Q3_G_WRONG_SOURCE",
    "_REPEAT_CASE_ID",
    "case_metadata",
]
