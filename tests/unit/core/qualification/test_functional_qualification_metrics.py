# © Artur Czarnecki. All rights reserved.

"""Unit tests for functional qualification metrics engine."""

from __future__ import annotations

import pytest

from intergrax.core.qualification.functional_diagnostic_comparator import compare_qualification_case
from intergrax.core.qualification.functional_diagnostic_expectation import (
    QualificationCaseExpectation,
    QualificationCheckExpectation,
    QualificationExecutionOutcome,
    QualificationFunctionalOutcome,
)
from intergrax.core.qualification.functional_qualification_case import (
    QualificationNormalizedCaseResult,
    QualificationRepeatabilityGroup,
)
from intergrax.core.qualification.functional_qualification_metrics import compute_qualification_metrics
from intergrax.runtime.diagnostics.functional_diagnostic_analysis import (
    FunctionalDiagnosticAnalysis,
    FunctionalDiagnosticCheckResult,
    FunctionalDiagnosticCheckStatus,
)
from intergrax.runtime.diagnostics.functional_diagnostic_identity import (
    FunctionalDiagnosticCheckId,
    FunctionalDiagnosticSpecificationId,
)
from intergrax.runtime.diagnostics.functional_operator_projection import FunctionalOperatorOutcomeStatus
from intergrax.contracts.execution_identity import mint_run_id, mint_task_id

pytestmark = pytest.mark.unit

_CHECK = FunctionalDiagnosticCheckId("fdcheck_00000000000000000000000000000011")
_SPEC = FunctionalDiagnosticSpecificationId("fdspec_000000000000000000000000a2b20001")


def _case(
    *,
    case_id: str,
    healthy: bool,
    expects_failure: bool,
    expects_inconclusive: bool,
    match: bool,
    operator: FunctionalOperatorOutcomeStatus,
) -> QualificationNormalizedCaseResult:
    expectation = QualificationCaseExpectation(
        case_id=case_id,
        expected_execution_outcome=QualificationExecutionOutcome.COMPLETED,
        expected_functional_outcome=(
            QualificationFunctionalOutcome.FAILED if expects_failure else QualificationFunctionalOutcome.PASSED
        ),
        expected_check_results=(
            QualificationCheckExpectation(_CHECK, FunctionalDiagnosticCheckStatus.PROVEN_PASS),
        ),
        expected_operator_outcome=None,
    )
    analysis = FunctionalDiagnosticAnalysis(
        tenant_id="tenant-a",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=None,
        specification_id=_SPEC,
        specification_version=1,
        check_results=(
            FunctionalDiagnosticCheckResult(
                check_id=_CHECK,
                status=FunctionalDiagnosticCheckStatus.PROVEN_FAIL if not match else FunctionalDiagnosticCheckStatus.PROVEN_PASS,
                factual_claim="claim",
                supporting_evidence_refs=(),
                limitations=(),
            ),
        ),
        first_proven_failure=_CHECK if not match else None,
        limitations=(),
    )
    comparison = compare_qualification_case(
        expectation,
        actual_execution_outcome=QualificationExecutionOutcome.COMPLETED,
        actual_functional_outcome=(
            QualificationFunctionalOutcome.FAILED if not healthy and not expects_inconclusive else QualificationFunctionalOutcome.PASSED
        ),
        analysis=analysis,
        operator_assessment=None,
    )
    return QualificationNormalizedCaseResult(
        case_id=case_id,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=None,
        tenant_id="tenant-a",
        comparison=comparison,
        functional_outcome=QualificationFunctionalOutcome.FAILED if not healthy else QualificationFunctionalOutcome.PASSED,
        diag_first_failed_check=str(_CHECK) if not match else None,
        operator_outcome=operator.value,
        repeat_group=None,
        identity_fidelity_match=True,
        is_healthy_case=healthy,
        expects_functional_failure=expects_failure,
        expects_inconclusive=expects_inconclusive,
        expected_first_failed_check=None,
    )


def test_metrics_all_pass() -> None:
    cases = (
        _case(
            case_id="healthy",
            healthy=True,
            expects_failure=False,
            expects_inconclusive=False,
            match=True,
            operator=FunctionalOperatorOutcomeStatus.PROVEN_FUNCTIONAL_SUCCESS,
        ),
    )
    metrics = compute_qualification_metrics(cases, repeatability_groups=())
    assert metrics.matched_cases == 1
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 0


def test_metrics_repeatability_fail() -> None:
    case = _case(
        case_id="repeat",
        healthy=False,
        expects_failure=True,
        expects_inconclusive=False,
        match=True,
        operator=FunctionalOperatorOutcomeStatus.PROVEN_FUNCTIONAL_FAILURE,
    )
    groups = (
        QualificationRepeatabilityGroup(
            group_id="g1",
            signatures=(("a",), ("b",)),
        ),
    )
    metrics = compute_qualification_metrics((case,), repeatability_groups=groups)
    assert metrics.repeatability_pass is False
