# © Artur Czarnecki. All rights reserved.

"""Ground-truth expectations for DIAG-FUNCTIONAL-Q4 mandatory matrix."""

from __future__ import annotations

from model_routing_qualifier.model_routing import (
    Q4_INVOKE_FAIL_TASK_CLASS,
    Q4_PRIMARY_TASK_CLASS,
    artifact_ref_for_profile,
    build_invoke_fail_profile,
    build_profile_a,
    build_profile_b,
)
from intergrax.core.qualification.functional_diagnostic_expectation import (
    QualificationCaseExpectation,
    QualificationCheckExpectation,
    QualificationExecutionOutcome,
    QualificationFunctionalOutcome,
)
from intergrax.runtime.diagnostics.functional_diagnostic_analysis import FunctionalDiagnosticCheckStatus
from intergrax.runtime.diagnostics.functional_operator_projection import FunctionalOperatorOutcomeStatus
from intergrax.runtime.diagnostics.specifications.q4_model_routing_functional_diagnostic_specification import (
    CHECK_Q4_CANDIDATES,
    CHECK_Q4_INVOCATION,
    CHECK_Q4_OUTPUT_RELATION,
    CHECK_Q4_SELECTION,
    CHECK_Q4_VALIDATION,
)

PROFILE_A_REF = artifact_ref_for_profile(build_profile_a())
PROFILE_B_REF = artifact_ref_for_profile(build_profile_b())
INVOKE_FAIL_PROFILE_REF = artifact_ref_for_profile(build_invoke_fail_profile())

HEALTHY_TASK = "What is 17 + 25? Reply with only the numeric result."


def _healthy_checks() -> tuple[QualificationCheckExpectation, ...]:
    return (
        QualificationCheckExpectation(CHECK_Q4_CANDIDATES, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_Q4_SELECTION, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_Q4_INVOCATION, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_Q4_OUTPUT_RELATION, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_Q4_VALIDATION, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
    )


def _wrong_route_checks() -> tuple[QualificationCheckExpectation, ...]:
    return (
        QualificationCheckExpectation(CHECK_Q4_CANDIDATES, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_Q4_SELECTION, FunctionalDiagnosticCheckStatus.PROVEN_FAIL),
        QualificationCheckExpectation(CHECK_Q4_INVOCATION, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_Q4_OUTPUT_RELATION, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_Q4_VALIDATION, FunctionalDiagnosticCheckStatus.PROVEN_FAIL),
    )


def _invoke_failure_checks() -> tuple[QualificationCheckExpectation, ...]:
    return (
        QualificationCheckExpectation(CHECK_Q4_CANDIDATES, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_Q4_SELECTION, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_Q4_INVOCATION, FunctionalDiagnosticCheckStatus.PROVEN_FAIL),
        QualificationCheckExpectation(CHECK_Q4_VALIDATION, FunctionalDiagnosticCheckStatus.PROVEN_FAIL),
    )


def _validation_failure_checks() -> tuple[QualificationCheckExpectation, ...]:
    return (
        QualificationCheckExpectation(CHECK_Q4_CANDIDATES, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_Q4_SELECTION, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_Q4_INVOCATION, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_Q4_OUTPUT_RELATION, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_Q4_VALIDATION, FunctionalDiagnosticCheckStatus.PROVEN_FAIL),
    )


def _missing_evidence_checks() -> tuple[QualificationCheckExpectation, ...]:
    return (
        QualificationCheckExpectation(CHECK_Q4_CANDIDATES, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_Q4_SELECTION, FunctionalDiagnosticCheckStatus.INSUFFICIENT_EVIDENCE),
        QualificationCheckExpectation(CHECK_Q4_INVOCATION, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        QualificationCheckExpectation(CHECK_Q4_OUTPUT_RELATION, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
    )


Q4_A_HEALTHY = QualificationCaseExpectation(
    case_id="Q4-A",
    expected_execution_outcome=QualificationExecutionOutcome.COMPLETED,
    expected_functional_outcome=QualificationFunctionalOutcome.PASSED,
    expected_check_results=_healthy_checks(),
    expected_first_proven_failed_check=None,
    expected_operator_outcome=FunctionalOperatorOutcomeStatus.PROVEN_FUNCTIONAL_SUCCESS,
)

Q4_B_WRONG_ROUTE = QualificationCaseExpectation(
    case_id="Q4-B",
    expected_execution_outcome=QualificationExecutionOutcome.COMPLETED,
    expected_functional_outcome=QualificationFunctionalOutcome.FAILED,
    expected_check_results=_wrong_route_checks(),
    expected_first_proven_failed_check=CHECK_Q4_SELECTION,
    expected_operator_outcome=FunctionalOperatorOutcomeStatus.PROVEN_FUNCTIONAL_FAILURE,
)

Q4_C_INVOKE_FAILURE = QualificationCaseExpectation(
    case_id="Q4-C",
    expected_execution_outcome=QualificationExecutionOutcome.COMPLETED,
    expected_functional_outcome=QualificationFunctionalOutcome.FAILED,
    expected_check_results=_invoke_failure_checks(),
    expected_first_proven_failed_check=CHECK_Q4_INVOCATION,
    expected_operator_outcome=FunctionalOperatorOutcomeStatus.PROVEN_FUNCTIONAL_FAILURE,
)

Q4_D_VALIDATION_FAILURE = QualificationCaseExpectation(
    case_id="Q4-D",
    expected_execution_outcome=QualificationExecutionOutcome.COMPLETED,
    expected_functional_outcome=QualificationFunctionalOutcome.FAILED,
    expected_check_results=_validation_failure_checks(),
    expected_first_proven_failed_check=CHECK_Q4_VALIDATION,
    expected_operator_outcome=FunctionalOperatorOutcomeStatus.PROVEN_FUNCTIONAL_FAILURE,
)

Q4_E_MISSING_EVIDENCE = QualificationCaseExpectation(
    case_id="Q4-E",
    expected_execution_outcome=QualificationExecutionOutcome.COMPLETED,
    expected_functional_outcome=QualificationFunctionalOutcome.FAILED,
    expected_check_results=_missing_evidence_checks(),
    expected_first_proven_failed_check=None,
    expected_operator_outcome=FunctionalOperatorOutcomeStatus.INCONCLUSIVE,
    include_validation=False,
    compare_functional_outcome=False,
)

Q4_F_HEALTHY = QualificationCaseExpectation(
    case_id="Q4-F-A",
    expected_execution_outcome=QualificationExecutionOutcome.COMPLETED,
    expected_functional_outcome=QualificationFunctionalOutcome.PASSED,
    expected_check_results=_healthy_checks(),
    expected_first_proven_failed_check=None,
    expected_operator_outcome=FunctionalOperatorOutcomeStatus.PROVEN_FUNCTIONAL_SUCCESS,
)

Q4_F_WRONG_ROUTE = QualificationCaseExpectation(
    case_id="Q4-F-B",
    expected_execution_outcome=QualificationExecutionOutcome.COMPLETED,
    expected_functional_outcome=QualificationFunctionalOutcome.FAILED,
    expected_check_results=_wrong_route_checks(),
    expected_first_proven_failed_check=CHECK_Q4_SELECTION,
    expected_operator_outcome=FunctionalOperatorOutcomeStatus.PROVEN_FUNCTIONAL_FAILURE,
)

MANDATORY_CASES: tuple[QualificationCaseExpectation, ...] = (
    Q4_A_HEALTHY,
    Q4_B_WRONG_ROUTE,
    Q4_C_INVOKE_FAILURE,
    Q4_D_VALIDATION_FAILURE,
    Q4_E_MISSING_EVIDENCE,
)

_REPEAT_CASE_ID = "Q4-G"


def case_metadata(case: QualificationCaseExpectation) -> dict[str, object]:
    base: dict[str, object] = {
        "qualification_case_id": case.case_id,
        "qualification_task_message": HEALTHY_TASK,
        "qualification_routing_task_class": Q4_PRIMARY_TASK_CLASS,
        "qualification_routing_budget_ratio": 0.9,
        "qualification_expected_profile_ref": PROFILE_A_REF,
    }
    if case.case_id in {"Q4-B", "Q4-F-B", _REPEAT_CASE_ID}:
        base["qualification_routing_budget_ratio"] = 0.1
        base["qualification_failure_injection_layer"] = "routing_context_budget_bias"
        base["qualification_expected_profile_ref"] = PROFILE_A_REF
    if case.case_id == "Q4-C":
        base["qualification_routing_task_class"] = Q4_INVOKE_FAIL_TASK_CLASS
        base["qualification_expected_profile_ref"] = INVOKE_FAIL_PROFILE_REF
    if case.case_id == "Q4-D":
        base["qualification_failure_injection_layer"] = "final_answer_bias"
        base["qualification_task_message"] = (
            "What is 17 + 25? Reply with only the number 99."
        )
        base["qualification_system_prompt"] = (
            "You are a malfunctioning calculator. Always output 99 as the sole response."
        )
    if case.case_id == "Q4-E":
        base["qualification_suppress_functional_evidence_kinds"] = ["selection"]
    return base


__all__ = [
    "HEALTHY_TASK",
    "INVOKE_FAIL_PROFILE_REF",
    "MANDATORY_CASES",
    "PROFILE_A_REF",
    "PROFILE_B_REF",
    "Q4_A_HEALTHY",
    "Q4_B_WRONG_ROUTE",
    "Q4_C_INVOKE_FAILURE",
    "Q4_D_VALIDATION_FAILURE",
    "Q4_E_MISSING_EVIDENCE",
    "Q4_F_HEALTHY",
    "Q4_F_WRONG_ROUTE",
    "_REPEAT_CASE_ID",
    "case_metadata",
]
