# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Typed semantic grouping features and model ports (DIAG-5C-A).

Separates canonical diagnostic facts (A) from bounded model input (B) and
strategy hypotheses (C). Does not perform grouping, model calls, or persistence
queries.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import NewType, Protocol, runtime_checkable

from intergrax.contracts.execution_identity import EventId
from intergrax.runtime.diagnostics.deterministic_problem_grouping import (
    build_deterministic_problem_signature,
)
from intergrax.runtime.diagnostics.diagnostic_assessment import (
    DiagnosticAssessment,
    DiagnosticFinding,
    DiagnosticLimitation,
)
from intergrax.runtime.diagnostics.problem_grouping import (
    DeterministicProblemSignature,
    ProblemGroupingCandidate,
    ProblemGroupingSubject,
    ProblemGroupingSubjectRef,
    normalize_assessment,
)

ProblemGroupingRepresentationVersion = NewType("ProblemGroupingRepresentationVersion", str)

REPRESENTATION_VERSION_V1 = ProblemGroupingRepresentationVersion("1")
MAX_TEXT_EVIDENCE_CHARS = 2_048


class ProblemGroupingTextEvidenceSourceKind(StrEnum):
    """Typed origin for bounded diagnostic text exposed to model strategies."""

    OPERATOR_CLAIM = "operator_claim"
    FACTUAL_LIMITATION = "factual_limitation"


class ProblemGroupingFeatureIntegrityError(Exception):
    """Raised when semantic feature projection violates contracts."""


@dataclass(frozen=True, slots=True)
class ProblemGroupingTextEvidence:
    """
    Bounded, provenance-backed diagnostic text for semantic/model input.

    Not a generic message bag — every item declares its source kind and the
    canonical event/evidence identifiers that justify inclusion.
    """

    source_kind: ProblemGroupingTextEvidenceSourceKind
    text: str
    supporting_event_ids: tuple[EventId, ...]
    supporting_evidence_ids: tuple[EventId, ...]


@dataclass(frozen=True, slots=True)
class ProblemGroupingFeatureSet:
    """
    Bounded model/grouping input derived from one execution assessment.

    Links back to deterministic structural identity without rewriting canonical
    facts. Additional typed dimensions (component, provider, error code, etc.)
    are added only when upstream diagnostic projection supplies real facts.
    """

    representation_version: ProblemGroupingRepresentationVersion
    structural_signature: DeterministicProblemSignature
    text_evidence: tuple[ProblemGroupingTextEvidence, ...]


@dataclass(frozen=True, slots=True)
class ProblemGroupingSemanticInput:
    """
    Strategy-facing bundle: structural subject plus optional semantic features.

    Deterministic strategies may ignore ``features``; model-assisted strategies
    require features projected upstream — never by querying persistence.
    """

    subject: ProblemGroupingSubject
    features: ProblemGroupingFeatureSet | None = None


@runtime_checkable
class ProblemGroupingFeatureProjector(Protocol):
    """
    Projection boundary: typed diagnostic facts → bounded grouping features.

    Not a grouping engine and not a diagnostic engine. Performs no grouping
    decisions and does not query persistence.
    """

    @property
    def representation_version(self) -> ProblemGroupingRepresentationVersion: ...

    def project(self, assessment: DiagnosticAssessment) -> ProblemGroupingFeatureSet: ...


@runtime_checkable
class SemanticCandidateGenerator(Protocol):
    """
    Cheap candidate narrowing for model-assisted grouping.

    Returns neighborhoods (size >= 2) that merit deeper adjudication. Must not
    query persistence — consumes pre-projected semantic inputs only.
    """

    def generate_neighborhoods(
        self,
        inputs: tuple[ProblemGroupingSemanticInput, ...],
    ) -> tuple[tuple[ProblemGroupingSubjectRef, ...], ...]: ...


@runtime_checkable
class ProblemGroupingAdjudicator(Protocol):
    """
    Expensive grouping decision over one candidate neighborhood.

    Returns a validated candidate proposal or None when the neighborhood should
    not merge. Must not query persistence.
    """

    def adjudicate(
        self,
        neighborhood: tuple[ProblemGroupingSemanticInput, ...],
    ) -> ProblemGroupingCandidate | None: ...


def _bounded_text(value: str, *, field_name: str) -> str:
    if type(value) is not str:
        raise ProblemGroupingFeatureIntegrityError(f"{field_name} must be str")
    normalized = value.strip()
    if not normalized:
        raise ProblemGroupingFeatureIntegrityError(f"{field_name} must be non-empty")
    if len(normalized) > MAX_TEXT_EVIDENCE_CHARS:
        raise ProblemGroupingFeatureIntegrityError(
            f"{field_name} exceeds {MAX_TEXT_EVIDENCE_CHARS} characters"
        )
    return normalized


def _text_evidence_from_finding(finding: DiagnosticFinding) -> ProblemGroupingTextEvidence:
    return ProblemGroupingTextEvidence(
        source_kind=ProblemGroupingTextEvidenceSourceKind.OPERATOR_CLAIM,
        text=_bounded_text(finding.claim, field_name="finding.claim"),
        supporting_event_ids=finding.supporting_event_ids,
        supporting_evidence_ids=finding.supporting_evidence_ids,
    )


def _text_evidence_from_limitation(
    limitation: DiagnosticLimitation,
) -> ProblemGroupingTextEvidence:
    return ProblemGroupingTextEvidence(
        source_kind=ProblemGroupingTextEvidenceSourceKind.FACTUAL_LIMITATION,
        text=_bounded_text(
            limitation.factual_message,
            field_name="limitation.factual_message",
        ),
        supporting_event_ids=limitation.supporting_event_ids,
        supporting_evidence_ids=limitation.supporting_evidence_ids,
    )


def project_assessment_features(
    assessment: DiagnosticAssessment,
    *,
    representation_version: ProblemGroupingRepresentationVersion = REPRESENTATION_VERSION_V1,
) -> ProblemGroupingFeatureSet:
    """
    Default v1 feature projection from one ``DiagnosticAssessment``.

    Uses only facts already present on the assessment — no persistence access,
    no invented dimensions, no raw logs.
    """
    if not str(representation_version).strip():
        raise ProblemGroupingFeatureIntegrityError(
            "representation_version must be non-empty"
        )

    subject = normalize_assessment(assessment)
    text_evidence = tuple(
        _text_evidence_from_finding(finding) for finding in assessment.findings
    ) + tuple(
        _text_evidence_from_limitation(limitation)
        for limitation in assessment.limitations
    )

    return ProblemGroupingFeatureSet(
        representation_version=representation_version,
        structural_signature=build_deterministic_problem_signature(subject),
        text_evidence=text_evidence,
    )


def semantic_input_from_assessment(
    assessment: DiagnosticAssessment,
    *,
    representation_version: ProblemGroupingRepresentationVersion = REPRESENTATION_VERSION_V1,
) -> ProblemGroupingSemanticInput:
    """Bundle structural subject and v1 semantic features for one assessment."""
    subject = normalize_assessment(assessment)
    features = project_assessment_features(
        assessment,
        representation_version=representation_version,
    )
    return ProblemGroupingSemanticInput(subject=subject, features=features)
