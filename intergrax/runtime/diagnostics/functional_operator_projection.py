# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Operator-facing functional diagnostic projection (DIAG-FUNCTIONAL-4)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.contracts.execution_identity import AttemptId, EventId, RunId, TaskId
from intergrax.contracts.functional_diagnostic_bounds import (
    MAX_FUNCTIONAL_DIAGNOSTIC_ANALYSIS_LIMITATIONS,
    MAX_FUNCTIONAL_DIAGNOSTIC_CHECKS,
    MAX_FUNCTIONAL_DIAGNOSTIC_CLAIM_LENGTH,
    MAX_FUNCTIONAL_DIAGNOSTIC_SUPPORTING_REFS,
    MAX_FUNCTIONAL_OPERATOR_FAILURES,
    MAX_FUNCTIONAL_OPERATOR_LIMITATIONS,
    MAX_FUNCTIONAL_OPERATOR_PASS_RESULTS,
)
from intergrax.runtime.diagnostics.functional_diagnostic_analysis import (
    FunctionalDiagnosticAnalysis,
    FunctionalDiagnosticCheckResult,
    FunctionalDiagnosticCheckStatus,
)
from intergrax.runtime.diagnostics.functional_diagnostic_identity import (
    FunctionalDiagnosticCheckId,
    FunctionalDiagnosticSpecificationId,
)


class FunctionalOperatorOutcomeStatus(StrEnum):
    """
    Bounded functional outcome summary for operators.

    PROVEN_FUNCTIONAL_FAILURE requires at least one PROVEN_FAIL check.
    PROVEN_FUNCTIONAL_SUCCESS requires every check to be PROVEN_PASS.
    INCONCLUSIVE covers absent proof, blocked, or unevaluated checks without
    proven failure — absence of proven failure is not proven success.
    """

    PROVEN_FUNCTIONAL_FAILURE = "proven_functional_failure"
    PROVEN_FUNCTIONAL_SUCCESS = "proven_functional_success"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, slots=True)
class FunctionalDiagnosticSummary:
    """Deterministic per-analysis check status counts."""

    checks_total: int
    passed: int
    failed: int
    insufficient: int
    blocked: int
    not_evaluated: int


@dataclass(frozen=True, slots=True)
class FunctionalDiagnosticOperatorFinding:
    """One proven functional failure — not a lifecycle finding and not root cause."""

    check_id: FunctionalDiagnosticCheckId
    factual_claim: str
    supporting_evidence_refs: tuple[EventId, ...]
    specification_id: FunctionalDiagnosticSpecificationId
    specification_version: int


@dataclass(frozen=True, slots=True)
class FunctionalDiagnosticOperatorLimitation:
    """Unresolved functional check — insufficient, blocked, or not evaluated."""

    check_id: FunctionalDiagnosticCheckId
    status: FunctionalDiagnosticCheckStatus
    factual_claim: str
    supporting_evidence_refs: tuple[EventId, ...]
    detail_limitations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FunctionalCheckPassResult:
    """Bounded pass projection — separate from problem-oriented findings."""

    check_id: FunctionalDiagnosticCheckId
    factual_claim: str


@dataclass(frozen=True, slots=True)
class FunctionalDiagnosticOperatorProjection:
    """
    Operator-facing functional diagnostic view for one analysis scope.

    NOT persisted and NOT a source of truth. Consumes ready
    ``FunctionalDiagnosticAnalysis`` without re-evaluating evidence.
    """

    tenant_id: str
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId | None
    specification_id: FunctionalDiagnosticSpecificationId
    specification_version: int
    outcome_status: FunctionalOperatorOutcomeStatus
    summary: FunctionalDiagnosticSummary
    failures: tuple[FunctionalDiagnosticOperatorFinding, ...]
    limitations: tuple[FunctionalDiagnosticOperatorLimitation, ...]
    pass_results: tuple[FunctionalCheckPassResult, ...]
    first_proven_failed_check: FunctionalDiagnosticCheckId | None
    analysis_limitations: tuple[str, ...]


class FunctionalOperatorProjector:
    """
    Deterministic projection from functional analysis to operator contracts.

    Does not query persistence, infer root cause, or use LLM.
    """

    def project(self, analysis: FunctionalDiagnosticAnalysis) -> FunctionalDiagnosticOperatorProjection:
        if type(analysis) is not FunctionalDiagnosticAnalysis:
            raise TypeError("analysis must be FunctionalDiagnosticAnalysis")

        summary = _summary_from_results(analysis.check_results)
        outcome_status = _outcome_status_from_summary(summary, analysis.check_results)

        failures: list[FunctionalDiagnosticOperatorFinding] = []
        limitations: list[FunctionalDiagnosticOperatorLimitation] = []
        pass_results: list[FunctionalCheckPassResult] = []

        for result in analysis.check_results:
            if result.status is FunctionalDiagnosticCheckStatus.PROVEN_FAIL:
                if len(failures) < MAX_FUNCTIONAL_OPERATOR_FAILURES:
                    failures.append(_failure_from_result(analysis, result))
            elif result.status is FunctionalDiagnosticCheckStatus.PROVEN_PASS:
                if len(pass_results) < MAX_FUNCTIONAL_OPERATOR_PASS_RESULTS:
                    pass_results.append(_pass_from_result(result))
            elif result.status in _LIMITATION_STATUSES:
                if len(limitations) < MAX_FUNCTIONAL_OPERATOR_LIMITATIONS:
                    limitations.append(_limitation_from_result(result))
            else:
                raise ValueError(f"unsupported functional check status: {result.status!r}")

        bounded_analysis_limitations = _bounded_analysis_limitations(analysis.limitations)

        return FunctionalDiagnosticOperatorProjection(
            tenant_id=analysis.tenant_id,
            task_id=analysis.task_id,
            run_id=analysis.run_id,
            attempt_id=analysis.attempt_id,
            specification_id=analysis.specification_id,
            specification_version=analysis.specification_version,
            outcome_status=outcome_status,
            summary=summary,
            failures=tuple(failures),
            limitations=tuple(limitations),
            pass_results=tuple(pass_results),
            first_proven_failed_check=analysis.first_proven_failure,
            analysis_limitations=bounded_analysis_limitations,
        )


_LIMITATION_STATUSES: frozenset[FunctionalDiagnosticCheckStatus] = frozenset(
    {
        FunctionalDiagnosticCheckStatus.INSUFFICIENT_EVIDENCE,
        FunctionalDiagnosticCheckStatus.BLOCKED_BY_UPSTREAM,
        FunctionalDiagnosticCheckStatus.NOT_EVALUATED,
    }
)


def _summary_from_results(
    check_results: tuple[FunctionalDiagnosticCheckResult, ...],
) -> FunctionalDiagnosticSummary:
    passed = 0
    failed = 0
    insufficient = 0
    blocked = 0
    not_evaluated = 0
    for result in check_results:
        if result.status is FunctionalDiagnosticCheckStatus.PROVEN_PASS:
            passed += 1
        elif result.status is FunctionalDiagnosticCheckStatus.PROVEN_FAIL:
            failed += 1
        elif result.status is FunctionalDiagnosticCheckStatus.INSUFFICIENT_EVIDENCE:
            insufficient += 1
        elif result.status is FunctionalDiagnosticCheckStatus.BLOCKED_BY_UPSTREAM:
            blocked += 1
        elif result.status is FunctionalDiagnosticCheckStatus.NOT_EVALUATED:
            not_evaluated += 1
    return FunctionalDiagnosticSummary(
        checks_total=len(check_results),
        passed=passed,
        failed=failed,
        insufficient=insufficient,
        blocked=blocked,
        not_evaluated=not_evaluated,
    )


def _outcome_status_from_summary(
    summary: FunctionalDiagnosticSummary,
    check_results: tuple[FunctionalDiagnosticCheckResult, ...],
) -> FunctionalOperatorOutcomeStatus:
    if summary.failed > 0:
        return FunctionalOperatorOutcomeStatus.PROVEN_FUNCTIONAL_FAILURE
    if check_results and summary.passed == summary.checks_total:
        return FunctionalOperatorOutcomeStatus.PROVEN_FUNCTIONAL_SUCCESS
    return FunctionalOperatorOutcomeStatus.INCONCLUSIVE


def _bounded_claim(claim: str) -> str:
    if len(claim) <= MAX_FUNCTIONAL_DIAGNOSTIC_CLAIM_LENGTH:
        return claim
    return claim[:MAX_FUNCTIONAL_DIAGNOSTIC_CLAIM_LENGTH]


def _bounded_refs(refs: tuple[EventId, ...]) -> tuple[EventId, ...]:
    if len(refs) <= MAX_FUNCTIONAL_DIAGNOSTIC_SUPPORTING_REFS:
        return refs
    return refs[:MAX_FUNCTIONAL_DIAGNOSTIC_SUPPORTING_REFS]


def _bounded_detail_limitations(limitations: tuple[str, ...]) -> tuple[str, ...]:
    if len(limitations) <= MAX_FUNCTIONAL_DIAGNOSTIC_ANALYSIS_LIMITATIONS:
        return limitations
    return limitations[:MAX_FUNCTIONAL_DIAGNOSTIC_ANALYSIS_LIMITATIONS]


def _bounded_analysis_limitations(limitations: tuple[str, ...]) -> tuple[str, ...]:
    if len(limitations) <= MAX_FUNCTIONAL_DIAGNOSTIC_ANALYSIS_LIMITATIONS:
        return limitations
    return limitations[:MAX_FUNCTIONAL_DIAGNOSTIC_ANALYSIS_LIMITATIONS]


def _failure_from_result(
    analysis: FunctionalDiagnosticAnalysis,
    result: FunctionalDiagnosticCheckResult,
) -> FunctionalDiagnosticOperatorFinding:
    return FunctionalDiagnosticOperatorFinding(
        check_id=result.check_id,
        factual_claim=_bounded_claim(result.factual_claim),
        supporting_evidence_refs=_bounded_refs(result.supporting_evidence_refs),
        specification_id=analysis.specification_id,
        specification_version=analysis.specification_version,
    )


def _pass_from_result(result: FunctionalDiagnosticCheckResult) -> FunctionalCheckPassResult:
    return FunctionalCheckPassResult(
        check_id=result.check_id,
        factual_claim=_bounded_claim(result.factual_claim),
    )


def _limitation_from_result(
    result: FunctionalDiagnosticCheckResult,
) -> FunctionalDiagnosticOperatorLimitation:
    return FunctionalDiagnosticOperatorLimitation(
        check_id=result.check_id,
        status=result.status,
        factual_claim=_bounded_claim(result.factual_claim),
        supporting_evidence_refs=_bounded_refs(result.supporting_evidence_refs),
        detail_limitations=_bounded_detail_limitations(result.limitations),
    )


__all__ = [
    "FunctionalCheckPassResult",
    "FunctionalDiagnosticOperatorFinding",
    "FunctionalDiagnosticOperatorLimitation",
    "FunctionalDiagnosticOperatorProjection",
    "FunctionalDiagnosticSummary",
    "FunctionalOperatorOutcomeStatus",
    "FunctionalOperatorProjector",
]
