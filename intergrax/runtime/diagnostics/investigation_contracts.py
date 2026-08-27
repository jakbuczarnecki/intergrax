# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""
Platform-to-investigator boundary contracts (DIAG-8B).

``IncidentInvestigationInput`` is read-only derived context from central
diagnostics — not an execution request and not canonical Problem lifecycle
state.

``InvestigationConclusion`` is a derived investigation workflow result — not
canonical Problem lifecycle state and not platform PROVEN root cause.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.contracts.evidence_claims import EvidenceClaimSet
from intergrax.runtime.diagnostics.diagnostic_read_models import (
    DiagnosticProblemDetail,
    DiagnosticProblemOccurrenceView,
    DiagnosticProblemSummary,
)
from intergrax.runtime.diagnostics.problem_grouping import ProblemGroupingSubjectRef
from intergrax.runtime.diagnostics.problem_lifecycle import ProblemId


class IncidentInvestigationIntegrityError(Exception):
    """Raised when investigation boundary input violates tenant or Problem invariants."""


class InvestigationConclusionIntegrityError(Exception):
    """Raised when an investigation conclusion violates authority boundaries."""


class InvestigationConclusionStatus(StrEnum):
    """Investigation workflow terminal state — not ProblemStatus or DiagnosticCertainty."""

    SUPPORTED = "supported"
    UNRESOLVED = "unresolved"
    NOT_ACCEPTED = "not_accepted"


@dataclass(frozen=True, slots=True)
class IncidentInvestigationProblemContext:
    """
    Bounded read-only Problem scope for one investigation starting point.

    Reuses canonical ``DiagnosticProblemSummary`` and ``DiagnosticProblemOccurrenceView``
    rather than duplicating occurrence DTOs. Occurrence views already carry typed
    subject identity, read status, optional assessment, and unavailable reason.
    """

    problem: DiagnosticProblemSummary
    occurrences: tuple[DiagnosticProblemOccurrenceView, ...]


@dataclass(frozen=True, slots=True)
class IncidentInvestigationInput:
    """
    Read-only derived context from central diagnostics for incident investigation.

    Investigators consume stable platform ``ProblemId`` identity and bounded facts
    only. This contract does not require ``TaskId`` or ``RunId`` at the top level;
    execution scope remains inside occurrence views when present.

    NOT an execution request. NOT a source of truth for Problem lifecycle.
    """

    tenant_id: str
    problem_contexts: tuple[IncidentInvestigationProblemContext, ...]


@dataclass(frozen=True, slots=True)
class InvestigationConclusion:
    """
    Derived investigation workflow conclusion — scenario/domain authority only.

    ``InvestigationConclusionStatus`` is intentionally separate from
    ``ProblemStatus`` and ``DiagnosticCertainty``. A ``SUPPORTED`` conclusion
    does not close a platform Problem or prove platform root cause.

    Typed evidence claims remain scenario authority via ``EvidenceClaimSet``.
    Richer canonical evidence navigation beyond finding/limitation refs is
    future work (DIAG-8C+).
    """

    status: InvestigationConclusionStatus
    investigated_problem_ids: tuple[ProblemId, ...]
    claim_set: EvidenceClaimSet | None = None
    summary: str | None = None


def _require_tenant_id(tenant_id: str) -> str:
    if type(tenant_id) is not str:
        raise TypeError("tenant_id must be str")
    normalized = tenant_id.strip()
    if not normalized:
        raise ValueError("tenant_id must be non-empty and not whitespace-only")
    if tenant_id != normalized:
        raise ValueError("tenant_id must not contain leading or trailing whitespace")
    return normalized


def _validate_subject_ref_tenant(
    subject_ref: ProblemGroupingSubjectRef,
    *,
    expected_tenant_id: str,
    context: str,
) -> None:
    if subject_ref.tenant_id != expected_tenant_id:
        raise IncidentInvestigationIntegrityError(
            f"{context} subject tenant_id {subject_ref.tenant_id!r} "
            f"does not match input tenant_id {expected_tenant_id!r}"
        )


def validate_incident_investigation_problem_context(
    context: IncidentInvestigationProblemContext,
    *,
    expected_tenant_id: str,
) -> IncidentInvestigationProblemContext:
    tenant_id = _require_tenant_id(expected_tenant_id)
    if context.problem.tenant_id != tenant_id:
        raise IncidentInvestigationIntegrityError(
            "problem summary tenant_id does not match input tenant_id"
        )
    for index, occurrence in enumerate(context.occurrences):
        _validate_subject_ref_tenant(
            occurrence.subject_ref,
            expected_tenant_id=tenant_id,
            context=f"occurrence[{index}]",
        )
    return context


def validate_incident_investigation_input(
    investigation_input: IncidentInvestigationInput,
) -> IncidentInvestigationInput:
    tenant_id = _require_tenant_id(investigation_input.tenant_id)
    if not investigation_input.problem_contexts:
        raise IncidentInvestigationIntegrityError(
            "incident investigation input requires at least one problem context"
        )

    seen_problem_ids: set[ProblemId] = set()
    for index, context in enumerate(investigation_input.problem_contexts):
        validate_incident_investigation_problem_context(
            context,
            expected_tenant_id=tenant_id,
        )
        problem_id = context.problem.problem_id
        if problem_id in seen_problem_ids:
            raise IncidentInvestigationIntegrityError(
                f"duplicate ProblemId {problem_id!r} in problem_contexts[{index}]"
            )
        seen_problem_ids.add(problem_id)

    return investigation_input


def incident_investigation_input_from_problem_details(
    *,
    tenant_id: str,
    details: tuple[DiagnosticProblemDetail, ...],
) -> IncidentInvestigationInput:
    """
    Map bounded ``DiagnosticReadService`` detail DTOs to investigation input.

    DIAG-8C will wire scenario entry to this helper; DIAG-8B establishes the
    contract only.
    """
    normalized_tenant_id = _require_tenant_id(tenant_id)
    if not details:
        raise IncidentInvestigationIntegrityError(
            "incident investigation input requires at least one problem detail"
        )

    contexts: list[IncidentInvestigationProblemContext] = []
    for index, detail in enumerate(details):
        if detail.tenant_id != normalized_tenant_id:
            raise IncidentInvestigationIntegrityError(
                f"problem detail[{index}] tenant_id does not match requested tenant_id"
            )
        summary = DiagnosticProblemSummary(
            problem_id=detail.problem_id,
            tenant_id=detail.tenant_id,
            status=detail.status,
            first_seen_at=detail.first_seen_at,
            last_seen_at=detail.last_seen_at,
            occurrence_count=detail.occurrence_count,
            grouping_provenance=detail.grouping_provenance,
        )
        contexts.append(
            IncidentInvestigationProblemContext(
                problem=summary,
                occurrences=detail.occurrences,
            )
        )

    return validate_incident_investigation_input(
        IncidentInvestigationInput(
            tenant_id=normalized_tenant_id,
            problem_contexts=tuple(contexts),
        )
    )


def validate_investigation_conclusion(
    conclusion: InvestigationConclusion,
) -> InvestigationConclusion:
    if not conclusion.investigated_problem_ids:
        raise InvestigationConclusionIntegrityError(
            "investigation conclusion requires at least one investigated ProblemId"
        )

    seen_problem_ids: set[ProblemId] = set()
    for problem_id in conclusion.investigated_problem_ids:
        if problem_id in seen_problem_ids:
            raise InvestigationConclusionIntegrityError(
                f"duplicate investigated ProblemId {problem_id!r}"
            )
        seen_problem_ids.add(problem_id)

    if conclusion.summary is not None:
        if type(conclusion.summary) is not str:
            raise TypeError("summary must be str when provided")
        normalized = conclusion.summary.strip()
        if not normalized:
            raise InvestigationConclusionIntegrityError(
                "summary must be non-empty when provided"
            )
        if conclusion.summary != normalized:
            raise InvestigationConclusionIntegrityError(
                "summary must not contain leading or trailing whitespace"
            )

    return conclusion
