# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Operator-facing functional diagnostic projection contracts (public platform surface)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.contracts.diagnostics.functional_diagnostic_check_status import (
    FunctionalDiagnosticCheckStatus,
)
from intergrax.contracts.diagnostics.functional_diagnostic_identity import (
    FunctionalDiagnosticCheckId,
    FunctionalDiagnosticSpecificationId,
)
from intergrax.contracts.execution_identity import AttemptId, EventId, RunId, TaskId


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

    NOT persisted and NOT a source of truth. Consumes ready functional analysis
    without re-evaluating evidence.
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


__all__ = [
    "FunctionalCheckPassResult",
    "FunctionalDiagnosticOperatorFinding",
    "FunctionalDiagnosticOperatorLimitation",
    "FunctionalDiagnosticOperatorProjection",
    "FunctionalDiagnosticSummary",
    "FunctionalOperatorOutcomeStatus",
]
