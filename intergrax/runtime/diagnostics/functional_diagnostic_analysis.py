# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Functional diagnostic analysis result contracts (DIAG-FUNCTIONAL-2)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.contracts.execution_identity import AttemptId, EventId, RunId, TaskId
from intergrax.runtime.diagnostics.functional_diagnostic_identity import (
    FunctionalDiagnosticCheckId,
    FunctionalDiagnosticSpecificationId,
)


class FunctionalDiagnosticCheckStatus(StrEnum):
    """
    Deterministic check outcome — not probabilistic confidence.

    PROVEN_PASS / PROVEN_FAIL require direct supporting evidence.
    INSUFFICIENT_EVIDENCE means absence of facts cannot be upgraded to FAIL.
    NOT_EVALUATED means the check was not reached in this analysis cycle.
    BLOCKED_BY_UPSTREAM means a dependency prevented evaluation.
    """

    PROVEN_PASS = "proven_pass"
    PROVEN_FAIL = "proven_fail"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NOT_EVALUATED = "not_evaluated"
    BLOCKED_BY_UPSTREAM = "blocked_by_upstream"


_CONTRADICTION_LIMITATION = (
    "Contradictory evidence observed for this check; result cannot be proven."
)


@dataclass(frozen=True, slots=True)
class FunctionalDiagnosticCheckResult:
    """Bounded result for one specification check."""

    check_id: FunctionalDiagnosticCheckId
    status: FunctionalDiagnosticCheckStatus
    factual_claim: str
    supporting_evidence_refs: tuple[EventId, ...]
    limitations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FunctionalDiagnosticAnalysis:
    """
    Deterministic functional diagnostic output for one execution scope.

    NOT persisted and NOT a source of truth. Future operator-facing
    ``DiagnosticAssessment`` composition may consume this projection.
    """

    tenant_id: str
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId | None
    specification_id: FunctionalDiagnosticSpecificationId
    specification_version: int
    check_results: tuple[FunctionalDiagnosticCheckResult, ...]
    first_proven_failure: FunctionalDiagnosticCheckId | None
    limitations: tuple[str, ...]


class FunctionalDiagnosticAnalysisIntegrityError(Exception):
    """Raised when analysis inputs violate scope or specification contracts."""


__all__ = [
    "FunctionalDiagnosticAnalysis",
    "FunctionalDiagnosticAnalysisIntegrityError",
    "FunctionalDiagnosticCheckResult",
    "FunctionalDiagnosticCheckStatus",
    "_CONTRADICTION_LIMITATION",
]
