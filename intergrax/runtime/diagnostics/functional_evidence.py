# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Generic typed functional/AI pipeline evidence foundation (DIAG-FUNCTIONAL-2)."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.execution_identity import (
    AttemptId,
    EventId,
    RunId,
    TaskId,
    validate_attempt_id,
    validate_event_id,
    validate_run_id,
    validate_task_id,
)
from intergrax.runtime.observability.functional_validation_evidence import DiagnosticExecutionCorrelation
from intergrax.runtime.observability.export_attributes import ObservabilityArtifactReference

PLATFORM_FUNCTIONAL_EVIDENCE_SCHEMA = "platform_functional_evidence.v1"


class PipelineEvidenceKind(StrEnum):
    """Composable evidence kinds for arbitrary AI pipeline stages."""

    ARTIFACT_LINEAGE = "artifact_lineage"
    OPERATION_OUTCOME = "operation_outcome"
    CANDIDATE_RANK = "candidate_rank"
    SELECTION = "selection"
    OUTPUT_RELATION = "output_relation"
    VALIDATION = "validation"


class PipelineOperationStatus(StrEnum):
    """Provider-neutral operation completion semantics."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class ScoreSemantics(StrEnum):
    """Provider-neutral score interpretation — no assumed universal scale."""

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"
    DISTANCE = "distance"
    PROBABILITY = "probability"
    PROVIDER_OPAQUE = "provider_opaque"


class PipelineEvidenceScope(BaseModel):
    """Execution scope shared by all functional evidence facts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId | None = None

    @field_validator("task_id", mode="before")
    @classmethod
    def _validate_task_id_field(cls, value: object) -> TaskId:
        return validate_task_id(value)

    @field_validator("run_id", mode="before")
    @classmethod
    def _validate_run_id_field(cls, value: object) -> RunId:
        return validate_run_id(value)

    @field_validator("attempt_id", mode="before")
    @classmethod
    def _validate_attempt_id_field(cls, value: object) -> AttemptId | None:
        if value is None:
            return None
        return validate_attempt_id(value)

    @field_validator("tenant_id")
    @classmethod
    def _require_tenant_id(cls, value: str) -> str:
        if type(value) is not str:
            raise TypeError("tenant_id must be str")
        normalized = value.strip()
        if not normalized:
            raise ValueError("tenant_id must be non-empty and not whitespace-only")
        if value != normalized:
            raise ValueError("tenant_id must not contain leading or trailing whitespace")
        return normalized

    @classmethod
    def from_correlation(cls, correlation: DiagnosticExecutionCorrelation) -> PipelineEvidenceScope:
        return cls(
            tenant_id=correlation.tenant_id,
            task_id=correlation.task_id,
            run_id=correlation.run_id,
            attempt_id=correlation.attempt_id,
        )


class PipelineEvidenceProvenance(BaseModel):
    """Bounded provenance for one functional evidence fact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    producer_component: str
    operation_id: str
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    upstream_evidence_ids: tuple[EventId, ...] = ()

    @field_validator("producer_component", "operation_id")
    @classmethod
    def _require_non_empty(cls, value: str) -> str:
        if type(value) is not str:
            raise TypeError("provenance identifiers must be str")
        normalized = value.strip()
        if not normalized:
            raise ValueError("provenance identifiers must be non-empty")
        if value != normalized:
            raise ValueError("provenance identifiers must not contain leading or trailing whitespace")
        return normalized

    @field_validator("upstream_evidence_ids", mode="before")
    @classmethod
    def _validate_upstream_evidence_ids(cls, value: object) -> tuple[EventId, ...]:
        if value is None:
            return ()
        if type(value) is not tuple:
            raise TypeError("upstream_evidence_ids must be a tuple")
        return tuple(validate_event_id(item) for item in value)

    @field_validator("recorded_at")
    @classmethod
    def _require_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("recorded_at must be timezone-aware")
        return value


class TypedPipelineScore(BaseModel):
    """Provider-neutral score with explicit semantics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    raw_value: float
    semantics: ScoreSemantics
    scale_hint: str = ""

    @field_validator("scale_hint")
    @classmethod
    def _validate_scale_hint(cls, value: str) -> str:
        if type(value) is not str:
            raise TypeError("scale_hint must be str")
        if len(value) > 128:
            raise ValueError("scale_hint must be <= 128 characters")
        return value


class PipelineArtifactLineageFact(BaseModel):
    """Reference-only lineage between source and derived artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_artifact_ref: ObservabilityArtifactReference
    derived_artifact_ref: ObservabilityArtifactReference
    lineage_operation: str

    @field_validator("lineage_operation")
    @classmethod
    def _require_lineage_operation(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("lineage_operation must be non-empty")
        if value != normalized:
            raise ValueError("lineage_operation must not contain leading or trailing whitespace")
        return normalized


class PipelineOperationOutcomeFact(BaseModel):
    """Typed outcome for one pipeline operation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    operation_name: str
    status: PipelineOperationStatus
    output_artifact_ref: ObservabilityArtifactReference | None = None

    @field_validator("operation_name")
    @classmethod
    def _require_operation_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("operation_name must be non-empty")
        if value != normalized:
            raise ValueError("operation_name must not contain leading or trailing whitespace")
        return normalized


class PipelineCandidateFact(BaseModel):
    """One ranked candidate in a selection pipeline."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query_id: str
    candidate_artifact_ref: ObservabilityArtifactReference
    score: TypedPipelineScore | None = None
    rank: int
    selected: bool

    @field_validator("query_id")
    @classmethod
    def _require_query_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("query_id must be non-empty")
        if value != normalized:
            raise ValueError("query_id must not contain leading or trailing whitespace")
        return normalized

    @field_validator("rank")
    @classmethod
    def _validate_rank(cls, value: int) -> int:
        if type(value) is not int or isinstance(value, bool):
            raise TypeError("rank must be int")
        if value < 1:
            raise ValueError("rank must be >= 1")
        return value


class PipelineSelectionFact(BaseModel):
    """Explicit selection from a candidate set."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query_id: str
    selected_artifact_ref: ObservabilityArtifactReference
    candidate_count: int
    selection_reason: str = ""

    @field_validator("query_id")
    @classmethod
    def _require_query_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("query_id must be non-empty")
        if value != normalized:
            raise ValueError("query_id must not contain leading or trailing whitespace")
        return normalized

    @field_validator("candidate_count")
    @classmethod
    def _validate_candidate_count(cls, value: int) -> int:
        if type(value) is not int or isinstance(value, bool):
            raise TypeError("candidate_count must be int")
        if value < 0:
            raise ValueError("candidate_count must be >= 0")
        return value

    @field_validator("selection_reason")
    @classmethod
    def _validate_selection_reason(cls, value: str) -> str:
        if type(value) is not str:
            raise TypeError("selection_reason must be str")
        if len(value) > 256:
            raise ValueError("selection_reason must be <= 256 characters")
        return value


class PipelineOutputRelationFact(BaseModel):
    """Relation between selected evidence and produced output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    selected_artifact_ref: ObservabilityArtifactReference
    output_artifact_ref: ObservabilityArtifactReference
    relation_kind: str

    @field_validator("relation_kind")
    @classmethod
    def _require_relation_kind(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("relation_kind must be non-empty")
        if value != normalized:
            raise ValueError("relation_kind must not contain leading or trailing whitespace")
        return normalized


class PipelineValidationLinkFact(BaseModel):
    """Link from pipeline output to a functional validation record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    validation_id: EventId
    output_artifact_ref: ObservabilityArtifactReference | None = None

    @field_validator("validation_id", mode="before")
    @classmethod
    def _validate_validation_id_field(cls, value: object) -> EventId:
        return validate_event_id(value)


class PlatformFunctionalEvidence(BaseModel):
    """
    One canonical functional/AI pipeline evidence fact.

    Exactly one typed payload field must match ``kind``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["platform_functional_evidence.v1"] = PLATFORM_FUNCTIONAL_EVIDENCE_SCHEMA
    evidence_id: EventId
    kind: PipelineEvidenceKind
    scope: PipelineEvidenceScope
    provenance: PipelineEvidenceProvenance
    artifact_lineage: PipelineArtifactLineageFact | None = None
    operation_outcome: PipelineOperationOutcomeFact | None = None
    candidate: PipelineCandidateFact | None = None
    selection: PipelineSelectionFact | None = None
    output_relation: PipelineOutputRelationFact | None = None
    validation_link: PipelineValidationLinkFact | None = None

    @field_validator("evidence_id", mode="before")
    @classmethod
    def _validate_evidence_id_field(cls, value: object) -> EventId:
        return validate_event_id(value)

    @model_validator(mode="after")
    def _validate_kind_payload_alignment(self) -> PlatformFunctionalEvidence:
        payload_by_kind: dict[PipelineEvidenceKind, object | None] = {
            PipelineEvidenceKind.ARTIFACT_LINEAGE: self.artifact_lineage,
            PipelineEvidenceKind.OPERATION_OUTCOME: self.operation_outcome,
            PipelineEvidenceKind.CANDIDATE_RANK: self.candidate,
            PipelineEvidenceKind.SELECTION: self.selection,
            PipelineEvidenceKind.OUTPUT_RELATION: self.output_relation,
            PipelineEvidenceKind.VALIDATION: self.validation_link,
        }
        active_payload = payload_by_kind[self.kind]
        if active_payload is None:
            raise ValueError(f"kind {self.kind.value!r} requires matching typed payload")
        for other_kind, other_payload in payload_by_kind.items():
            if other_kind is not self.kind and other_payload is not None:
                raise ValueError(
                    f"kind {self.kind.value!r} cannot coexist with payload for {other_kind.value!r}",
                )
        return self


__all__ = [
    "PLATFORM_FUNCTIONAL_EVIDENCE_SCHEMA",
    "PipelineArtifactLineageFact",
    "PipelineCandidateFact",
    "PipelineEvidenceKind",
    "PipelineEvidenceProvenance",
    "PipelineEvidenceScope",
    "PipelineOperationOutcomeFact",
    "PipelineOperationStatus",
    "PipelineOutputRelationFact",
    "PipelineSelectionFact",
    "PipelineValidationLinkFact",
    "PlatformFunctionalEvidence",
    "ScoreSemantics",
    "TypedPipelineScore",
]
