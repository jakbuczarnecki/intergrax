# © Artur Czarnecki. All rights reserved.

"""Typed contracts for local behavioral qualification session integrity (DS-E2E-15J-QI1)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping

from intergrax.runtime.nexus.tracing.execution.evaluator_model_attempt import (
    EvaluatorModelAttemptDiagV1,
)
from intergrax.runtime.nexus.tracing.execution.reconciliation_phase import (
    ReconciliationPhaseDiagV1,
)


QUALIFICATION_SESSION_SCHEMA_VERSION = "qualification_session.v1"

COMPLETION_ALIGNMENT_TRACE_SCHEMA = "incident.completion_alignment.v1"
CANONICAL_MODEL_ATTEMPT_TRACE_SCHEMA = EvaluatorModelAttemptDiagV1.schema_id()
RECONCILIATION_PHASE_TRACE_SCHEMA = ReconciliationPhaseDiagV1.schema_id()


class VersionMatchPolicy(StrEnum):
    EXACT = "exact"
    IGNORE = "ignore"


class QualificationIdentityStatus(StrEnum):
    MATCH = "match"
    MISMATCH = "mismatch"
    UNVERIFIABLE = "unverifiable"


class QualificationPreconditionFailureKind(StrEnum):
    SOURCE_DRIFT = "source_drift"
    MODEL_IDENTITY_MISMATCH = "model_identity_mismatch"
    PROVIDER_RUNTIME_MISMATCH = "provider_runtime_mismatch"
    CONFIG_MISMATCH = "config_mismatch"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    STRICT_TOOL_PRECONDITION_FAILED = "strict_tool_precondition_failed"
    STRUCTURED_OUTPUT_PRECONDITION_FAILED = "structured_output_precondition_failed"


class QualificationSessionState(StrEnum):
    CREATED = "created"
    PRECONDITIONS_PASSED = "preconditions_passed"
    RUNNING = "running"
    PARTIAL = "partial"
    FINALIZING = "finalizing"
    FINALIZED = "finalized"
    BLOCKED = "blocked"
    INVALID = "invalid"
    FAILED_FINALIZATION = "failed_finalization"


class QualificationSessionVerdict(StrEnum):
    PASS = "pass"
    INCONCLUSIVE = "inconclusive"
    BLOCKED = "blocked"
    PARTIAL = "partial"
    INVALID = "invalid"
    CRITICAL_REGRESSION = "critical_regression"
    FAILED_FINALIZATION = "failed_finalization"


class TraceReadbackStatus(StrEnum):
    PASS = "pass"
    PARTIAL = "partial"
    FAILED = "failed"
    NOT_AVAILABLE = "not_available"


class EvidenceCompletenessStatus(StrEnum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    NOT_AVAILABLE = "not_available"


class SafetyGateOutcome(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


class FinalizationPhase(StrEnum):
    RAW_COMPLETE = "raw_complete"
    DERIVED_COMPLETE = "derived_complete"
    VALIDATED = "validated"
    MANIFEST_COMPLETE = "manifest_complete"
    FINAL_REPORT_COMPLETE = "final_report_complete"
    SESSION_FINALIZED = "session_finalized"


@dataclass(frozen=True, slots=True)
class ProviderRuntimeVersion:
    major: int
    minor: int
    patch: int

    def normalized(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


@dataclass(frozen=True, slots=True)
class QualificationRuntimeIdentity:
    provider_kind: str
    runtime_version: ProviderRuntimeVersion | None
    endpoint_host: str
    model_name: str | None
    model_digest: str | None
    quantization: str | None


@dataclass(frozen=True, slots=True)
class QualificationExperimentIdentity:
    provider_kind: str
    provider_runtime_version: ProviderRuntimeVersion | None
    provider_runtime_version_policy: VersionMatchPolicy
    model_name: str
    model_digest: str | None
    model_digest_policy: VersionMatchPolicy
    quantization: str | None
    generation_config_fingerprint: str
    scenario_id: str
    input_id: str
    source_fingerprint: str
    config_fingerprint: str


@dataclass(frozen=True, slots=True)
class QualificationPreconditionFailure:
    kind: QualificationPreconditionFailureKind
    detail: str


@dataclass(frozen=True, slots=True)
class QualificationPreconditionResult:
    eligible: bool
    failures: tuple[QualificationPreconditionFailure, ...]


@dataclass(frozen=True, slots=True)
class SourceBlobFingerprint:
    path: str
    content_hash: str
    semantic_group: str


@dataclass(frozen=True, slots=True)
class SourceFingerprintSnapshot:
    repository_head_sha: str
    blobs: tuple[SourceBlobFingerprint, ...]

    def semantic_fingerprint(self) -> str:
        parts = sorted(f"{item.path}:{item.content_hash}" for item in self.blobs)
        return "|".join(parts)


@dataclass(frozen=True, slots=True)
class SourceDriftReport:
    repository_head_drift: bool
    qualification_semantic_source_drift: bool
    frozen_head_sha: str
    current_head_sha: str
    changed_blobs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class QualificationAttemptObservation:
    run_id: str
    node_id: str
    attempt_index: int


@dataclass(frozen=True, slots=True)
class TypedAlignmentEvent:
    alignment_mismatch_detected: bool
    alignment_direction: str | None
    alignment_correctable: bool
    alignment_correction_attempted: bool
    alignment_correction_succeeded: bool
    alignment_correction_exhausted: bool
    revision_authoritative_context_present: bool


@dataclass(frozen=True, slots=True)
class TypedAlignmentReadback:
    status: TraceReadbackStatus
    events: tuple[TypedAlignmentEvent, ...]


@dataclass(frozen=True, slots=True)
class ReconciliationPhaseObservation:
    run_id: str
    validation_invalid: bool
    entered_reconciliation: bool


@dataclass(frozen=True, slots=True)
class ReconciliationLeakAssessment:
    outcome: SafetyGateOutcome
    observations: tuple[ReconciliationPhaseObservation, ...]


@dataclass(frozen=True, slots=True)
class ThirdPassAssessment:
    outcome: SafetyGateOutcome
    attempt_evidence_complete: bool
    violating_attempts: tuple[QualificationAttemptObservation, ...]


@dataclass(frozen=True, slots=True)
class QualificationSpec:
    experiment_identity: QualificationExperimentIdentity
    run_count: int
    required_artifacts: tuple[str, ...]
    source_blob_paths: tuple[str, ...]
    semantic_source_groups: Mapping[str, tuple[str, ...]]
    max_evaluator_attempt_index: int
    source_checkpoint_run_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CanonicalRunRecord:
    run_index: int
    run_id: str
    trace_events: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class SessionIntegrityReport:
    runtime_identity_status: QualificationIdentityStatus
    model_identity_status: QualificationIdentityStatus
    source_identity_status: QualificationIdentityStatus
    config_identity_status: QualificationIdentityStatus
    trace_readback_status: TraceReadbackStatus
    attempt_evidence_status: EvidenceCompletenessStatus
    artifact_completeness_status: EvidenceCompletenessStatus
    finalization_status: QualificationSessionState
