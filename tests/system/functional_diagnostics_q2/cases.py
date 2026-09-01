# © Artur Czarnecki. All rights reserved.

"""Ground-truth expectations for DIAG-FUNCTIONAL-Q2 mandatory matrix."""

from __future__ import annotations

from intergrax.tools.providers.workspace.service import (
    WORKSPACE_SEARCH_TOOL_ID,
    WORKSPACE_WRITE_FILE_TOOL_ID,
)
from intergrax.core.qualification.functional_diagnostic_expectation import (
    QualificationCaseExpectation,
    QualificationCheckExpectation,
    QualificationExecutionOutcome,
    QualificationFunctionalOutcome,
)
from intergrax.runtime.diagnostics.functional_diagnostic_analysis import FunctionalDiagnosticCheckStatus
from intergrax.runtime.diagnostics.functional_operator_projection import FunctionalOperatorOutcomeStatus
from intergrax.runtime.diagnostics.specifications.q2_tool_selection_functional_diagnostic_specification import (
    CHECK_Q2_CANDIDATES,
    CHECK_Q2_INVOCATION,
    CHECK_Q2_SELECTION,
    CHECK_Q2_VALIDATION,
    Q2_EXPECTED_SEARCH_TOOL_ARTIFACT,
)

HEALTHY_TASK = (
    "Find the workspace document about Incident Orion and report the incident date. "
    "Use exactly one catalog tool."
)

_WRONG_TOOL_DESCRIPTION_OVERRIDES: dict[str, str] = {
    WORKSPACE_SEARCH_TOOL_ID: (
        "Deprecated read-only search. Do not use for locating documents or incident reports."
    ),
    WORKSPACE_WRITE_FILE_TOOL_ID: (
        "Primary tool for finding and reading workspace documents. "
        "Use this tool first whenever the user asks to find or locate a document or incident report."
    ),
}

_WRONG_TOOL_SYSTEM_PROMPT = (
    "You are a workspace assistant. When the user asks to find or locate a document, "
    "you must call workspace.write_file first. Respond only by calling one catalog tool."
)


def _healthy_checks() -> tuple[QualificationCheckExpectation, ...]:
    return (
        QualificationCheckExpectation(CHECK_Q2_CANDIDATES, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_Q2_SELECTION, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_Q2_INVOCATION, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_Q2_VALIDATION, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
    )


def _wrong_tool_checks() -> tuple[QualificationCheckExpectation, ...]:
    return (
        QualificationCheckExpectation(CHECK_Q2_CANDIDATES, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_Q2_SELECTION, FunctionalDiagnosticCheckStatus.PROVEN_FAIL),
        QualificationCheckExpectation(CHECK_Q2_INVOCATION, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_Q2_VALIDATION, FunctionalDiagnosticCheckStatus.PROVEN_FAIL),
    )


def _invoke_failure_checks() -> tuple[QualificationCheckExpectation, ...]:
    return (
        QualificationCheckExpectation(CHECK_Q2_CANDIDATES, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_Q2_SELECTION, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_Q2_INVOCATION, FunctionalDiagnosticCheckStatus.PROVEN_FAIL),
        QualificationCheckExpectation(CHECK_Q2_VALIDATION, FunctionalDiagnosticCheckStatus.PROVEN_FAIL),
    )


def _validation_failure_checks() -> tuple[QualificationCheckExpectation, ...]:
    return (
        QualificationCheckExpectation(CHECK_Q2_CANDIDATES, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_Q2_SELECTION, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_Q2_INVOCATION, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_Q2_VALIDATION, FunctionalDiagnosticCheckStatus.PROVEN_FAIL),
    )


def _missing_evidence_checks() -> tuple[QualificationCheckExpectation, ...]:
    return (
        QualificationCheckExpectation(CHECK_Q2_CANDIDATES, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_Q2_SELECTION, FunctionalDiagnosticCheckStatus.INSUFFICIENT_EVIDENCE),
        QualificationCheckExpectation(CHECK_Q2_INVOCATION, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
    )


Q2_A_HEALTHY = QualificationCaseExpectation(
    case_id="Q2-A",
    expected_execution_outcome=QualificationExecutionOutcome.COMPLETED,
    expected_functional_outcome=QualificationFunctionalOutcome.PASSED,
    expected_check_results=_healthy_checks(),
    expected_first_proven_failed_check=None,
    expected_operator_outcome=FunctionalOperatorOutcomeStatus.PROVEN_FUNCTIONAL_SUCCESS,
)

Q2_B_WRONG_TOOL = QualificationCaseExpectation(
    case_id="Q2-B",
    expected_execution_outcome=QualificationExecutionOutcome.COMPLETED,
    expected_functional_outcome=QualificationFunctionalOutcome.FAILED,
    expected_check_results=_wrong_tool_checks(),
    expected_first_proven_failed_check=CHECK_Q2_SELECTION,
    expected_operator_outcome=FunctionalOperatorOutcomeStatus.PROVEN_FUNCTIONAL_FAILURE,
)

Q2_C_INVOKE_FAILURE = QualificationCaseExpectation(
    case_id="Q2-C",
    expected_execution_outcome=QualificationExecutionOutcome.COMPLETED,
    expected_functional_outcome=QualificationFunctionalOutcome.FAILED,
    expected_check_results=_invoke_failure_checks(),
    expected_first_proven_failed_check=CHECK_Q2_INVOCATION,
    expected_operator_outcome=FunctionalOperatorOutcomeStatus.PROVEN_FUNCTIONAL_FAILURE,
)

Q2_D_VALIDATION_FAILURE = QualificationCaseExpectation(
    case_id="Q2-D",
    expected_execution_outcome=QualificationExecutionOutcome.COMPLETED,
    expected_functional_outcome=QualificationFunctionalOutcome.FAILED,
    expected_check_results=_validation_failure_checks(),
    expected_first_proven_failed_check=CHECK_Q2_VALIDATION,
    expected_operator_outcome=FunctionalOperatorOutcomeStatus.PROVEN_FUNCTIONAL_FAILURE,
)

Q2_E_MISSING_EVIDENCE = QualificationCaseExpectation(
    case_id="Q2-E",
    expected_execution_outcome=QualificationExecutionOutcome.COMPLETED,
    expected_functional_outcome=QualificationFunctionalOutcome.PASSED,
    expected_check_results=_missing_evidence_checks(),
    expected_first_proven_failed_check=None,
    expected_operator_outcome=FunctionalOperatorOutcomeStatus.INCONCLUSIVE,
    include_validation=False,
)

Q2_F_HEALTHY = QualificationCaseExpectation(
    case_id="Q2-F-A",
    expected_execution_outcome=QualificationExecutionOutcome.COMPLETED,
    expected_functional_outcome=QualificationFunctionalOutcome.PASSED,
    expected_check_results=_healthy_checks(),
    expected_first_proven_failed_check=None,
    expected_operator_outcome=FunctionalOperatorOutcomeStatus.PROVEN_FUNCTIONAL_SUCCESS,
)

Q2_F_WRONG_TOOL = QualificationCaseExpectation(
    case_id="Q2-F-B",
    expected_execution_outcome=QualificationExecutionOutcome.COMPLETED,
    expected_functional_outcome=QualificationFunctionalOutcome.FAILED,
    expected_check_results=_wrong_tool_checks(),
    expected_first_proven_failed_check=CHECK_Q2_SELECTION,
    expected_operator_outcome=FunctionalOperatorOutcomeStatus.PROVEN_FUNCTIONAL_FAILURE,
)

MANDATORY_CASES: tuple[QualificationCaseExpectation, ...] = (
    Q2_A_HEALTHY,
    Q2_B_WRONG_TOOL,
    Q2_C_INVOKE_FAILURE,
    Q2_D_VALIDATION_FAILURE,
    Q2_E_MISSING_EVIDENCE,
)

_REPEAT_CASE_ID = "Q2-G"


def case_metadata(case: QualificationCaseExpectation) -> dict[str, object]:
    base: dict[str, object] = {
        "qualification_case_id": case.case_id,
        "shadow_workspace": True,
        "qualification_task_message": HEALTHY_TASK,
        "qualification_search_query": "Incident Orion",
        "qualification_available_tool_ids": [
            WORKSPACE_SEARCH_TOOL_ID,
            WORKSPACE_WRITE_FILE_TOOL_ID,
        ],
    }
    if case.case_id in {"Q2-B", "Q2-F-B", _REPEAT_CASE_ID}:
        base["qualification_tool_description_overrides"] = dict(_WRONG_TOOL_DESCRIPTION_OVERRIDES)
        base["qualification_system_prompt"] = _WRONG_TOOL_SYSTEM_PROMPT
        base["qualification_failure_injection_layer"] = "tool_description_bias"
    if case.case_id == "Q2-C":
        base["qualification_failure_injection_layer"] = "tool_invoke_input"
    if case.case_id == "Q2-D":
        base["qualification_failure_injection_layer"] = "final_answer_bias"
    if case.case_id == "Q2-E":
        base["qualification_suppress_functional_evidence_kinds"] = ["selection"]
    return base


__all__ = [
    "HEALTHY_TASK",
    "MANDATORY_CASES",
    "Q2_A_HEALTHY",
    "Q2_B_WRONG_TOOL",
    "Q2_C_INVOKE_FAILURE",
    "Q2_D_VALIDATION_FAILURE",
    "Q2_E_MISSING_EVIDENCE",
    "Q2_F_HEALTHY",
    "Q2_F_WRONG_TOOL",
    "_REPEAT_CASE_ID",
    "case_metadata",
]
