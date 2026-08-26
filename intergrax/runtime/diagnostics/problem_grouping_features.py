# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Typed semantic grouping features and model ports (DIAG-5C-A / DIAG-5C-B).

Separates canonical diagnostic facts (A) from bounded model input (B) and
strategy hypotheses (C). Does not perform grouping, model calls, or persistence
queries.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import NewType, Protocol, runtime_checkable

from intergrax.contracts.execution_identity import EventId
from intergrax.contracts.execution_phase import ExecutionPhase
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
    ProblemGroupingInput,
    ProblemGroupingSubject,
    ProblemGroupingSubjectRef,
    normalize_assessment,
)
from intergrax.runtime.events.event_taxonomy import EventCategory

ProblemGroupingRepresentationVersion = NewType("ProblemGroupingRepresentationVersion", str)

REPRESENTATION_VERSION_V1 = ProblemGroupingRepresentationVersion("1")
REPRESENTATION_VERSION_V2 = ProblemGroupingRepresentationVersion("2")
MAX_TEXT_EVIDENCE_CHARS = 2_048

# Semantic causal ref-kind labels for bounded causal descriptors (not instance ids).
CAUSAL_SOURCE_REF_KIND_MESSAGE_BUS_TASK = "message_bus_task"
CAUSAL_TARGET_REF_KIND_RUNTIME_EXECUTION = "runtime_execution"


class ProblemGroupingTextEvidenceSourceKind(StrEnum):
    """Typed origin for bounded diagnostic text exposed to model strategies."""

    OPERATOR_CLAIM = "operator_claim"
    FACTUAL_LIMITATION = "factual_limitation"
    PROBLEM_SIGNAL_SAFE_MESSAGE = "problem_signal_safe_message"


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
class ProblemGroupingExecutionFeature:
    """
    Execution-level operational context for grouping strategies.

    Models phase, event taxonomy, and retry shape — not execution identity.
    ``TaskId`` / ``RunId`` / ``AttemptId`` belong in ``subject_ref`` only.
    """

    phase: ExecutionPhase | None = None
    event_category: EventCategory | None = None
    event_type: str | None = None
    is_retry_related: bool | None = None
    supporting_event_ids: tuple[EventId, ...] = ()
    supporting_evidence_ids: tuple[EventId, ...] = ()


@dataclass(frozen=True, slots=True)
class ProblemGroupingComponentFeature:
    """
    Component/subsystem context derived from platform problem signals.

    ``source_layer`` and ``source_component`` mirror ``PlatformProblemSignal``
    dimensions without importing observability models into strategy contracts.
    """

    source_layer: str
    source_component: str
    supporting_event_ids: tuple[EventId, ...] = ()
    supporting_evidence_ids: tuple[EventId, ...] = ()


@dataclass(frozen=True, slots=True)
class ProblemGroupingOperationFeature:
    """
    Operation/tool context for grouping strategies.

    Instance-specific node/step identifiers are contextual evidence, not
    deterministic problem-class identity.
    """

    agent_id: str | None = None
    tool_id: str | None = None
    capability: str | None = None
    node_id: str | None = None
    step_id: str | None = None
    operation: str | None = None
    supporting_event_ids: tuple[EventId, ...] = ()
    supporting_evidence_ids: tuple[EventId, ...] = ()


@dataclass(frozen=True, slots=True)
class ProblemGroupingIntegrationFeature:
    """
    Integration/provider context for grouping strategies.

    Uses validated semantic string identifiers — no global provider enum.
    """

    provider: str
    integration_id: str | None = None
    namespace: str | None = None
    supporting_event_ids: tuple[EventId, ...] = ()
    supporting_evidence_ids: tuple[EventId, ...] = ()


@dataclass(frozen=True, slots=True)
class ProblemGroupingFailureFeature:
    """
    Normalized failure context for grouping strategies.

    Mirrors ``PlatformProblemSignal`` failure dimensions without coupling
    strategies to observability export models.
    """

    problem_kind: str
    severity: str
    error_code: str | None = None
    exception_type: str | None = None
    supporting_event_ids: tuple[EventId, ...] = ()
    supporting_evidence_ids: tuple[EventId, ...] = ()


@dataclass(frozen=True, slots=True)
class ProblemGroupingCausalFeature:
    """
    Bounded causal-shape descriptor for grouping strategies.

    Captures relation semantics from ``PlatformCausalEvidence`` without
    exposing instance-heavy refs or persistence queries.
    """

    relation_kind: str
    source_ref_kind: str
    target_ref_kind: str
    source_provider: str | None = None
    supporting_event_ids: tuple[EventId, ...] = ()
    supporting_evidence_ids: tuple[EventId, ...] = ()


@dataclass(frozen=True, slots=True)
class ProblemGroupingFeatureSet:
    """
    Bounded model/grouping input derived from one execution assessment.

    Links back to deterministic structural identity without rewriting canonical
    facts. Feature tuples are analytical context in source presentation order —
    order does not define grouping equality; strategies decide consumption.

    v2 envelope adds typed execution, component, operation, integration,
    failure, and causal descriptors. Categories may be empty until a projector
    supplies real platform facts (DIAG-5C-C).
    """

    subject_ref: ProblemGroupingSubjectRef
    representation_version: ProblemGroupingRepresentationVersion
    structural_signature: DeterministicProblemSignature
    execution_context: tuple[ProblemGroupingExecutionFeature, ...] = ()
    component_context: tuple[ProblemGroupingComponentFeature, ...] = ()
    operation_context: tuple[ProblemGroupingOperationFeature, ...] = ()
    integration_context: tuple[ProblemGroupingIntegrationFeature, ...] = ()
    failure_context: tuple[ProblemGroupingFailureFeature, ...] = ()
    causal_context: tuple[ProblemGroupingCausalFeature, ...] = ()
    text_evidence: tuple[ProblemGroupingTextEvidence, ...] = ()

    def __post_init__(self) -> None:
        validate_problem_grouping_feature_set(self)


@runtime_checkable
class ProblemGroupingFeatureProjector(Protocol):
    """
    Projection boundary: typed diagnostic facts → bounded grouping features.

    Not a grouping engine and not a diagnostic engine. Performs no grouping
    decisions and does not query persistence.
    """

    @property
    def representation_version(self) -> ProblemGroupingRepresentationVersion: ...

    def project(
        self,
        assessment: DiagnosticAssessment,
        subject: ProblemGroupingSubject,
    ) -> ProblemGroupingFeatureSet: ...


@runtime_checkable
class SemanticCandidateGenerator(Protocol):
    """
    Cheap candidate narrowing for model-assisted grouping.

    Returns neighborhoods (size >= 2) that merit deeper adjudication. Must not
    query persistence — consumes pre-projected strategy inputs only.
    """

    def generate_neighborhoods(
        self,
        inputs: tuple[ProblemGroupingInput, ...],
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
        neighborhood: tuple[ProblemGroupingInput, ...],
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


def _semantic_identifier(value: str, *, field_name: str) -> str:
    if type(value) is not str:
        raise ProblemGroupingFeatureIntegrityError(f"{field_name} must be str")
    normalized = value.strip()
    if not normalized:
        raise ProblemGroupingFeatureIntegrityError(f"{field_name} must be non-empty")
    return normalized


def _optional_semantic_identifier(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _semantic_identifier(value, field_name=field_name)


def _validate_execution_feature(feature: ProblemGroupingExecutionFeature) -> None:
    if feature.event_type is not None:
        _semantic_identifier(feature.event_type, field_name="execution.event_type")


def _validate_component_feature(feature: ProblemGroupingComponentFeature) -> None:
    _semantic_identifier(feature.source_layer, field_name="component.source_layer")
    _semantic_identifier(feature.source_component, field_name="component.source_component")


def _validate_operation_feature(feature: ProblemGroupingOperationFeature) -> None:
    has_context = any(
        (
            feature.agent_id,
            feature.tool_id,
            feature.capability,
            feature.node_id,
            feature.step_id,
            feature.operation,
        )
    )
    if not has_context:
        raise ProblemGroupingFeatureIntegrityError(
            "operation feature must declare at least one operation dimension"
        )
    _optional_semantic_identifier(feature.agent_id, field_name="operation.agent_id")
    _optional_semantic_identifier(feature.tool_id, field_name="operation.tool_id")
    _optional_semantic_identifier(feature.capability, field_name="operation.capability")
    _optional_semantic_identifier(feature.node_id, field_name="operation.node_id")
    _optional_semantic_identifier(feature.step_id, field_name="operation.step_id")
    _optional_semantic_identifier(feature.operation, field_name="operation.operation")


def _validate_integration_feature(feature: ProblemGroupingIntegrationFeature) -> None:
    _semantic_identifier(feature.provider, field_name="integration.provider")
    _optional_semantic_identifier(
        feature.integration_id,
        field_name="integration.integration_id",
    )
    _optional_semantic_identifier(feature.namespace, field_name="integration.namespace")


def _validate_failure_feature(feature: ProblemGroupingFailureFeature) -> None:
    _semantic_identifier(feature.problem_kind, field_name="failure.problem_kind")
    _semantic_identifier(feature.severity, field_name="failure.severity")
    _optional_semantic_identifier(feature.error_code, field_name="failure.error_code")
    _optional_semantic_identifier(
        feature.exception_type,
        field_name="failure.exception_type",
    )


def _validate_causal_feature(feature: ProblemGroupingCausalFeature) -> None:
    _semantic_identifier(feature.relation_kind, field_name="causal.relation_kind")
    _semantic_identifier(feature.source_ref_kind, field_name="causal.source_ref_kind")
    _semantic_identifier(feature.target_ref_kind, field_name="causal.target_ref_kind")
    _optional_semantic_identifier(
        feature.source_provider,
        field_name="causal.source_provider",
    )


def _validate_text_evidence(evidence: ProblemGroupingTextEvidence) -> None:
    _bounded_text(evidence.text, field_name="text_evidence.text")


def validate_problem_grouping_feature_set(features: ProblemGroupingFeatureSet) -> None:
    """Validate bounded typing and semantic identifier constraints."""
    if not str(features.representation_version).strip():
        raise ProblemGroupingFeatureIntegrityError(
            "representation_version must be non-empty"
        )

    for feature in features.execution_context:
        _validate_execution_feature(feature)
    for feature in features.component_context:
        _validate_component_feature(feature)
    for feature in features.operation_context:
        _validate_operation_feature(feature)
    for feature in features.integration_context:
        _validate_integration_feature(feature)
    for feature in features.failure_context:
        _validate_failure_feature(feature)
    for feature in features.causal_context:
        _validate_causal_feature(feature)
    for evidence in features.text_evidence:
        _validate_text_evidence(evidence)


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
    subject: ProblemGroupingSubject | None = None,
    representation_version: ProblemGroupingRepresentationVersion = REPRESENTATION_VERSION_V2,
) -> ProblemGroupingFeatureSet:
    """
    Default feature projection from one ``DiagnosticAssessment``.

    Uses only facts already present on the assessment — no persistence access,
    no invented dimensions, no raw logs. Extended feature tuples remain empty
    until DIAG-5C-C populates them from typed platform facts.
    """
    if not str(representation_version).strip():
        raise ProblemGroupingFeatureIntegrityError(
            "representation_version must be non-empty"
        )

    resolved_subject = subject if subject is not None else normalize_assessment(assessment)
    text_evidence = tuple(
        _text_evidence_from_finding(finding) for finding in assessment.findings
    ) + tuple(
        _text_evidence_from_limitation(limitation)
        for limitation in assessment.limitations
    )

    return ProblemGroupingFeatureSet(
        subject_ref=resolved_subject.ref,
        representation_version=representation_version,
        structural_signature=build_deterministic_problem_signature(resolved_subject),
        text_evidence=text_evidence,
    )
