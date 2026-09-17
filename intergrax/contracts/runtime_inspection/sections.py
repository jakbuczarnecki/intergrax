# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Composable runtime inspection snapshot sections (INSPECT-01-A)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.contracts.execution_identity import (
    AttemptId,
    EventId,
    ExecutionId,
    RunId,
    TaskId,
    validate_attempt_id,
    validate_event_id,
    validate_execution_id,
    validate_run_id,
    validate_task_id,
)
from intergrax.contracts.agent_runtime_governance import ToolAuthorizationDecisionState
from intergrax.contracts.execution_continuation import ExecutionContinuationLifecycleState
from intergrax.contracts.runtime_inspection.completeness import RuntimeInspectionCompleteness
from intergrax.contracts.execution_artifact_read import ExecutionArtifactLifecycleStatus
from intergrax.contracts.external_work_runtime_read import ExternalWorkRuntimeStatus
from intergrax.contracts.memory_runtime_read import (
    MemoryRuntimeOperationClass,
    MemoryRuntimeOperationStatus,
)
from intergrax.contracts.model_runtime_read import ModelRuntimeInvocationStatus
from intergrax.contracts.tool_runtime_read import ToolRuntimeInvocationOutcome


class RuntimeInspectionTimelineDomain(StrEnum):
    RUNTIME_EVENT = "runtime_event"
    CAUSAL_EVIDENCE = "causal_evidence"


class RuntimeInspectionIdentitySection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str = Field(min_length=1)
    task_id: TaskId | None = None
    run_id: RunId | None = None
    attempt_id: AttemptId | None = None
    execution_id: ExecutionId
    identity_completeness: RuntimeInspectionCompleteness

    @field_validator("execution_id", mode="before")
    @classmethod
    def _validate_execution_id(cls, value: object) -> ExecutionId:
        return validate_execution_id(value)

    @field_validator("task_id", mode="before")
    @classmethod
    def _validate_task_id(cls, value: object | None) -> TaskId | None:
        if value is None:
            return None
        return validate_task_id(value)

    @field_validator("run_id", mode="before")
    @classmethod
    def _validate_run_id(cls, value: object | None) -> RunId | None:
        if value is None:
            return None
        return validate_run_id(value)

    @field_validator("attempt_id", mode="before")
    @classmethod
    def _validate_attempt_id(cls, value: object | None) -> AttemptId | None:
        if value is None:
            return None
        return validate_attempt_id(value)


class RuntimeInspectionExecutionStateSection(BaseModel):
    """Canonical execution facts — derived from reconstruction, not diagnostics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    runtime_history_completeness: str = Field(min_length=1)
    attempt_count: int = Field(ge=0)
    has_runtime_events: bool
    has_causal_evidence: bool
    completeness: RuntimeInspectionCompleteness
    source_id: str = Field(min_length=1)


class RuntimeInspectionTimelineEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    observed_at: datetime
    sequence_key: int = Field(ge=1)
    domain: RuntimeInspectionTimelineDomain
    kind: str = Field(min_length=1)
    safe_summary: str = Field(min_length=1)
    event_id: EventId | None = None

    @field_validator("observed_at")
    @classmethod
    def _validate_observed_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("observed_at must be timezone-aware")
        return value

    @field_validator("event_id", mode="before")
    @classmethod
    def _validate_event_id(cls, value: object | None) -> EventId | None:
        if value is None:
            return None
        return validate_event_id(value)


class RuntimeInspectionTimelineSection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    entries: tuple[RuntimeInspectionTimelineEntry, ...] = Field(default_factory=tuple)
    is_truncated: bool = False
    completeness: RuntimeInspectionCompleteness
    source_id: str = Field(min_length=1)


class RuntimeInspectionDiagnosticFinding(BaseModel):
    """Diagnostic interpretation — never mixed with execution fact fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str = Field(min_length=1)
    certainty: str = Field(min_length=1)
    safe_summary: str = Field(min_length=1)


class RuntimeInspectionDiagnosticSection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    findings: tuple[RuntimeInspectionDiagnosticFinding, ...] = Field(default_factory=tuple)
    limitations: tuple[str, ...] = Field(default_factory=tuple)
    completeness: RuntimeInspectionCompleteness
    source_id: str = Field(min_length=1)
    incident_refs: tuple[str, ...] = Field(default_factory=tuple)


class RuntimeInspectionEvidenceReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: EventId
    kind: str = Field(min_length=1)
    source_id: str = Field(min_length=1)

    @field_validator("evidence_id", mode="before")
    @classmethod
    def _validate_evidence_id(cls, value: object) -> EventId:
        return validate_event_id(value)


class RuntimeInspectionEvidenceSection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    references: tuple[RuntimeInspectionEvidenceReference, ...] = Field(default_factory=tuple)
    completeness: RuntimeInspectionCompleteness
    source_id: str = Field(min_length=1)


class RuntimeInspectionToolInvocation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    invocation_id: str = Field(min_length=1)
    tool_id: str = Field(min_length=1)
    execution_id: ExecutionId
    attempt_id: AttemptId | None = None
    sequence_key: int = Field(ge=1)
    outcome: ToolRuntimeInvocationOutcome
    status_label: str = Field(min_length=1)
    failure_classification: str | None = None
    args_digest_ref: str | None = None
    provider_correlation_ref: str | None = None
    governance_evidence_refs: tuple[str, ...] = Field(default_factory=tuple)
    evidence_refs: tuple[str, ...] = Field(default_factory=tuple)
    safe_summary: str = Field(min_length=1)

    @field_validator("execution_id", mode="before")
    @classmethod
    def _validate_execution_id(cls, value: object) -> ExecutionId:
        return validate_execution_id(value)

    @field_validator("attempt_id", mode="before")
    @classmethod
    def _validate_attempt_id(cls, value: object | None) -> AttemptId | None:
        if value is None:
            return None
        return validate_attempt_id(value)


class RuntimeInspectionToolSection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    invocations: tuple[RuntimeInspectionToolInvocation, ...] = Field(default_factory=tuple)
    is_truncated: bool = False
    completeness: RuntimeInspectionCompleteness
    source_id: str = Field(min_length=1)
    source_available: bool = True


class RuntimeInspectionGovernanceDecisionEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    decision_ref: str = Field(min_length=1)
    outcome: ToolAuthorizationDecisionState
    policy_id: str | None = None
    policy_revision: str | None = None
    reason_classification: str = Field(min_length=1)
    approval_required: bool = False
    grant_ref: str | None = None
    evidence_refs: tuple[str, ...] = Field(default_factory=tuple)


class RuntimeInspectionGovernanceSection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    decisions: tuple[RuntimeInspectionGovernanceDecisionEntry, ...] = Field(default_factory=tuple)
    completeness: RuntimeInspectionCompleteness
    source_id: str = Field(min_length=1)
    source_available: bool = True


class RuntimeInspectionContinuationSection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    continuation_id: str | None = None
    lifecycle_state: ExecutionContinuationLifecycleState | None = None
    revision: int | None = Field(default=None, ge=0)
    pause_reason_classification: str | None = None
    waiting_for_human: bool = False
    resolution_status: str | None = None
    resume_status: str | None = None
    is_durable: bool | None = None
    approval_correlation_ref: str | None = None
    completeness: RuntimeInspectionCompleteness
    source_id: str = Field(min_length=1)
    source_available: bool = True


class RuntimeInspectionMemoryOperation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    operation_ref: str = Field(min_length=1)
    memory_class: str = Field(min_length=1)
    operation_class: MemoryRuntimeOperationClass
    operation_status: MemoryRuntimeOperationStatus
    record_ref: str | None = None
    source_category: str = Field(min_length=1)
    execution_id: ExecutionId
    attempt_id: AttemptId | None = None
    sequence_key: int = Field(ge=1)
    evidence_refs: tuple[str, ...] = Field(default_factory=tuple)
    safe_summary: str = Field(min_length=1)

    @field_validator("execution_id", mode="before")
    @classmethod
    def _validate_execution_id(cls, value: object) -> ExecutionId:
        return validate_execution_id(value)

    @field_validator("attempt_id", mode="before")
    @classmethod
    def _validate_attempt_id(cls, value: object | None) -> AttemptId | None:
        if value is None:
            return None
        return validate_attempt_id(value)


class RuntimeInspectionMemorySection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    operations: tuple[RuntimeInspectionMemoryOperation, ...] = Field(default_factory=tuple)
    is_truncated: bool = False
    completeness: RuntimeInspectionCompleteness
    source_id: str = Field(min_length=1)
    source_available: bool = True


class RuntimeInspectionModelInvocation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    invocation_ref: str = Field(min_length=1)
    model_ref: str = Field(min_length=1)
    capability_label: str = Field(min_length=1)
    invocation_status: ModelRuntimeInvocationStatus
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    finish_reason: str | None = None
    execution_id: ExecutionId
    attempt_id: AttemptId | None = None
    sequence_key: int = Field(ge=1)
    evidence_refs: tuple[str, ...] = Field(default_factory=tuple)
    safe_summary: str = Field(min_length=1)

    @field_validator("execution_id", mode="before")
    @classmethod
    def _validate_execution_id(cls, value: object) -> ExecutionId:
        return validate_execution_id(value)

    @field_validator("attempt_id", mode="before")
    @classmethod
    def _validate_attempt_id(cls, value: object | None) -> AttemptId | None:
        if value is None:
            return None
        return validate_attempt_id(value)


class RuntimeInspectionModelSection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    invocations: tuple[RuntimeInspectionModelInvocation, ...] = Field(default_factory=tuple)
    is_truncated: bool = False
    completeness: RuntimeInspectionCompleteness
    source_id: str = Field(min_length=1)
    source_available: bool = True


class RuntimeInspectionExternalWorkEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    work_ref: str = Field(min_length=1)
    work_class: str = Field(min_length=1)
    work_status: ExternalWorkRuntimeStatus
    provider_ref: str = Field(min_length=1)
    failure_classification: str = Field(min_length=1)
    retryable: bool = False
    execution_id: ExecutionId
    attempt_id: AttemptId | None = None
    sequence_key: int = Field(ge=1)
    evidence_refs: tuple[str, ...] = Field(default_factory=tuple)
    safe_summary: str = Field(min_length=1)

    @field_validator("execution_id", mode="before")
    @classmethod
    def _validate_execution_id(cls, value: object) -> ExecutionId:
        return validate_execution_id(value)

    @field_validator("attempt_id", mode="before")
    @classmethod
    def _validate_attempt_id(cls, value: object | None) -> AttemptId | None:
        if value is None:
            return None
        return validate_attempt_id(value)


class RuntimeInspectionExternalWorkSection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    work_entries: tuple[RuntimeInspectionExternalWorkEntry, ...] = Field(default_factory=tuple)
    is_truncated: bool = False
    completeness: RuntimeInspectionCompleteness
    source_id: str = Field(min_length=1)
    source_available: bool = True


class RuntimeInspectionArtifactEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_ref: str = Field(min_length=1)
    artifact_type: str = Field(min_length=1)
    lifecycle_status: ExecutionArtifactLifecycleStatus
    content_classification: str = Field(min_length=1)
    execution_id: ExecutionId
    attempt_id: AttemptId | None = None
    sequence_key: int = Field(ge=1)
    evidence_refs: tuple[str, ...] = Field(default_factory=tuple)
    safe_summary: str = Field(min_length=1)

    @field_validator("execution_id", mode="before")
    @classmethod
    def _validate_execution_id(cls, value: object) -> ExecutionId:
        return validate_execution_id(value)

    @field_validator("attempt_id", mode="before")
    @classmethod
    def _validate_attempt_id(cls, value: object | None) -> AttemptId | None:
        if value is None:
            return None
        return validate_attempt_id(value)


class RuntimeInspectionArtifactSection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    artifacts: tuple[RuntimeInspectionArtifactEntry, ...] = Field(default_factory=tuple)
    is_truncated: bool = False
    completeness: RuntimeInspectionCompleteness
    source_id: str = Field(min_length=1)
    source_available: bool = True


__all__ = [
    "RuntimeInspectionDiagnosticFinding",
    "RuntimeInspectionDiagnosticSection",
    "RuntimeInspectionEvidenceReference",
    "RuntimeInspectionEvidenceSection",
    "RuntimeInspectionExecutionStateSection",
    "RuntimeInspectionIdentitySection",
    "RuntimeInspectionTimelineDomain",
    "RuntimeInspectionTimelineEntry",
    "RuntimeInspectionTimelineSection",
    "RuntimeInspectionToolInvocation",
    "RuntimeInspectionToolSection",
    "RuntimeInspectionGovernanceDecisionEntry",
    "RuntimeInspectionGovernanceSection",
    "RuntimeInspectionContinuationSection",
    "RuntimeInspectionMemoryOperation",
    "RuntimeInspectionMemorySection",
    "RuntimeInspectionModelInvocation",
    "RuntimeInspectionModelSection",
    "RuntimeInspectionExternalWorkEntry",
    "RuntimeInspectionExternalWorkSection",
    "RuntimeInspectionArtifactEntry",
    "RuntimeInspectionArtifactSection",
]
