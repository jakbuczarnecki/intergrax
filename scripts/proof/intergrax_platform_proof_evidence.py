# © Artur Czarnecki. All rights reserved.

"""Generic typed Platform Proof evidence contract (PP-REPORT-2).

Separate from:
- ``SuiteReceipt`` (suite orchestration)
- ``ToolsSqlInvestigationProofResult`` (proof-local evaluation result)
- future ProofReport presentation models
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from scripts.proof.intergrax_proof_contracts import ProofProfile, ProofStatus

PLATFORM_PROOF_EVIDENCE_SCHEMA_VERSION = "intergrax.platform_proof_evidence.v1"

_SECRET_FIELD_PATTERN = re.compile(
    r"(secret|password|token|api[_-]?key|authorization|credential)",
    re.IGNORECASE,
)

REPORT_SAFE_REDACTION_PLACEHOLDER = "[REDACTED]"

# Defense-in-depth only — primary guarantee is explicit ReportSafeText ownership.
_BEARER_HEADER_PATTERN = re.compile(
    r"(Authorization:\s*Bearer\s+)(\S+)",
    re.IGNORECASE,
)
_BEARER_TOKEN_PATTERN = re.compile(r"\bBearer\s+\S+", re.IGNORECASE)
_ENV_SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"\b([A-Z][A-Z0-9_]*(?:TOKEN|PASSWORD|API_KEY)|OPENAI_API_KEY|ANTHROPIC_API_KEY)=([^\s&'\"\\]+)",
    re.IGNORECASE,
)
_CREDENTIAL_URL_PATTERN = re.compile(
    r"([a-z][a-z0-9+.-]*://)([^:@/\s]+):([^@/\s]+)@",
    re.IGNORECASE,
)


class ProofEvidenceExecutionStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"
    CRASH = "CRASH"


class ProofStepExecutionStatus(StrEnum):
    OK = "ok"
    FAIL = "fail"
    SKIPPED = "skipped"


class ParticipantClass(StrEnum):
    PLATFORM = "PLATFORM"
    PROOF_OWNED = "PROOF_OWNED"
    EXTERNAL_VENDOR = "EXTERNAL_VENDOR"
    CONTROLLED_FIXTURE = "CONTROLLED_FIXTURE"
    REAL_BOUNDARY = "REAL_BOUNDARY"


class ReportSafeVisibility(StrEnum):
    SUMMARY_ONLY = "SUMMARY_ONLY"
    REPORT_SAFE = "REPORT_SAFE"
    REDACTED = "REDACTED"


class ReportSafeTextSourceKind(StrEnum):
    PROOF_AUTHORED = "PROOF_AUTHORED"
    RUNTIME_SANITIZED = "RUNTIME_SANITIZED"
    RUNTIME_EXPLICIT = "RUNTIME_EXPLICIT"


def sanitize_untrusted_report_text(text: str) -> tuple[str, bool]:
    """Deterministic defense-in-depth redaction for common secret-bearing patterns."""
    redaction_applied = False
    sanitized = text

    def _replace(pattern: re.Pattern[str], repl: str | re.Pattern[str]) -> None:
        nonlocal sanitized, redaction_applied
        updated = pattern.sub(repl, sanitized)
        if updated != sanitized:
            redaction_applied = True
            sanitized = updated

    _replace(
        _BEARER_HEADER_PATTERN,
        rf"\1{REPORT_SAFE_REDACTION_PLACEHOLDER}",
    )
    _replace(_BEARER_TOKEN_PATTERN, f"Bearer {REPORT_SAFE_REDACTION_PLACEHOLDER}")
    _replace(_ENV_SECRET_ASSIGNMENT_PATTERN, rf"\1={REPORT_SAFE_REDACTION_PLACEHOLDER}")
    _replace(
        _CREDENTIAL_URL_PATTERN,
        rf"\1{REPORT_SAFE_REDACTION_PLACEHOLDER}:{REPORT_SAFE_REDACTION_PLACEHOLDER}@",
    )
    return sanitized, redaction_applied


class ReportSafeText(BaseModel):
    """Text explicitly approved for human report rendering."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    visibility: ReportSafeVisibility = ReportSafeVisibility.REPORT_SAFE
    redaction_applied: bool = False
    source_kind: ReportSafeTextSourceKind = ReportSafeTextSourceKind.PROOF_AUTHORED

    @model_validator(mode="after")
    def _validate_redacted_payload(self) -> ReportSafeText:
        if self.visibility == ReportSafeVisibility.REDACTED:
            if self.text != REPORT_SAFE_REDACTION_PLACEHOLDER:
                raise ValueError("redacted ReportSafeText must not expose original value")
            if not self.redaction_applied:
                raise ValueError("redacted ReportSafeText must set redaction_applied")
        return self

    @classmethod
    def require_non_empty(cls, value: ReportSafeText, *, field_name: str) -> ReportSafeText:
        if not value.text.strip():
            raise ValueError(f"{field_name} must not be empty")
        return value


def proof_authored_report_safe_text(text: str) -> ReportSafeText:
    """Static or proof-builder-controlled text approved for reporting."""
    return ReportSafeText(
        text=text,
        visibility=ReportSafeVisibility.REPORT_SAFE,
        redaction_applied=False,
        source_kind=ReportSafeTextSourceKind.PROOF_AUTHORED,
    )


def sanitized_runtime_report_safe_text(text: str) -> ReportSafeText:
    """Runtime/provider text sanitized before report-safe persistence."""
    sanitized, redaction_applied = sanitize_untrusted_report_text(text)
    return ReportSafeText(
        text=sanitized,
        visibility=ReportSafeVisibility.REPORT_SAFE,
        redaction_applied=redaction_applied,
        source_kind=ReportSafeTextSourceKind.RUNTIME_SANITIZED,
    )


def explicit_runtime_report_safe_text(text: str) -> ReportSafeText:
    """Runtime text explicitly approved for reporting (defense-in-depth sanitization)."""
    sanitized, redaction_applied = sanitize_untrusted_report_text(text)
    return ReportSafeText(
        text=sanitized,
        visibility=ReportSafeVisibility.REPORT_SAFE,
        redaction_applied=redaction_applied,
        source_kind=ReportSafeTextSourceKind.RUNTIME_EXPLICIT,
    )


def redacted_report_safe_text() -> ReportSafeText:
    """Placeholder for values that must not appear in human reports."""
    return ReportSafeText(
        text=REPORT_SAFE_REDACTION_PLACEHOLDER,
        visibility=ReportSafeVisibility.REDACTED,
        redaction_applied=True,
        source_kind=ReportSafeTextSourceKind.RUNTIME_SANITIZED,
    )


def _reject_raw_report_safe_str(value: object) -> object:
    if isinstance(value, str):
        raise ValueError(
            "raw runtime string cannot be used for report-safe fields; "
            "wrap with proof_authored_report_safe_text, "
            "sanitized_runtime_report_safe_text, or explicit_runtime_report_safe_text"
        )
    return value


ReportSafeScalar = int | float | bool | None
ReportSafeFieldValue = Annotated[
    ReportSafeScalar | ReportSafeText | tuple[ReportSafeScalar | ReportSafeText, ...],
    Field(union_mode="left_to_right"),
]


class ReportSafeField(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    visibility: ReportSafeVisibility
    value: ReportSafeFieldValue | None = None
    is_secret: bool = False

    @field_validator("value", mode="before")
    @classmethod
    def _reject_raw_str_value(cls, value: object) -> object:
        return _reject_raw_report_safe_str(value)

    @model_validator(mode="after")
    def _reject_secret_as_report_safe(self) -> ReportSafeField:
        if self.visibility != ReportSafeVisibility.REPORT_SAFE:
            return self
        if self.is_secret or _SECRET_FIELD_PATTERN.search(self.name):
            raise ValueError(
                f"secret-bearing field cannot be marked report-safe: {self.name}"
            )
        return self


class ReportSafePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    visibility: ReportSafeVisibility = ReportSafeVisibility.REPORT_SAFE
    summary: ReportSafeText = Field(
        default_factory=lambda: proof_authored_report_safe_text("")
    )
    fields: tuple[ReportSafeField, ...] = ()


class ReportSafeValue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["scalar", "list", "object"]
    visibility: ReportSafeVisibility = ReportSafeVisibility.REPORT_SAFE
    scalar: ReportSafeScalar = None
    items: tuple[ReportSafeScalar, ...] = ()
    fields: tuple[ReportSafeField, ...] = ()

    @model_validator(mode="after")
    def _validate_kind_payload(self) -> ReportSafeValue:
        if self.kind == "scalar":
            if self.items or self.fields:
                raise ValueError("scalar ReportSafeValue must not include items or fields")
        elif self.kind == "list":
            if self.fields:
                raise ValueError("list ReportSafeValue must not include fields")
        elif self.kind == "object":
            if self.items:
                raise ValueError("object ReportSafeValue must not include items")
        for field in self.fields:
            if field.visibility == ReportSafeVisibility.REPORT_SAFE:
                if field.is_secret or _SECRET_FIELD_PATTERN.search(field.name):
                    raise ValueError(
                        f"secret-bearing field cannot be marked report-safe: {field.name}"
                    )
        return self


class ProofIdentityEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    proof_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    proof_version: str = Field(min_length=1)
    source_revision: str = Field(min_length=1)
    execution_profile: ProofProfile


class ExecutionMetadataEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: ProofEvidenceExecutionStatus
    started_at: datetime
    finished_at: datetime | None = None
    duration_ms: int | None = Field(default=None, ge=0)
    platform: str = Field(min_length=1)
    runtime_version: str | None = None
    source_dirty: bool | None = None
    command_executable: str | None = None
    command_argv: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_chronology(self) -> ExecutionMetadataEvidence:
        if self.finished_at is not None and self.finished_at < self.started_at:
            raise ValueError("finished_at must not precede started_at")
        return self


class ProofClaimEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim: str = Field(min_length=1)
    user_relevance: str = Field(min_length=1)
    success_criteria: tuple[str, ...]
    falsification_criteria: tuple[str, ...]
    excluded_claims: tuple[str, ...]


class ParticipantEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    participant_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    implementation: str = Field(min_length=1)
    version_or_model: str = Field(min_length=1)
    role: str = Field(min_length=1)
    participant_class: ParticipantClass


class ArchitectureEdgeEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    from_participant: str = Field(min_length=1)
    to_participant: str = Field(min_length=1)
    relationship: str = Field(min_length=1)


class ArchitectureEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    participants: tuple[ParticipantEvidence, ...]
    edges: tuple[ArchitectureEdgeEvidence, ...] = ()


class DatasetEnvironmentEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    row_count: int = Field(ge=0)
    seed: int | None = None
    scenario_version: str | None = None
    fingerprint_sha256: str = Field(min_length=1)
    infrastructure_identity: str | None = None
    access_mode: str | None = None
    ground_truth_checks: tuple[str, ...] = ()
    information_exposed_to_model: tuple[str, ...] = ()


class EnvironmentEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: DatasetEnvironmentEvidence | None = None


class MetricEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    value: int | float | str | bool
    unit: str | None = None
    description: str | None = None


class ToolInvocationEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_id: str = Field(min_length=1)
    provider_tool_name: str | None = None
    call_id: str | None = None
    safe_arguments: ReportSafePayload | None = None
    success: bool
    output_summary: ReportSafeText = Field(
        default_factory=lambda: proof_authored_report_safe_text("")
    )
    bounded_output: ReportSafePayload | None = None
    duration_ms: int | None = Field(default=None, ge=0)
    error: ReportSafeText | None = None

    @field_validator("output_summary", "error", mode="before")
    @classmethod
    def _reject_raw_str_fields(cls, value: object) -> object:
        return _reject_raw_report_safe_str(value)


class ProofExecutionStep(BaseModel):
    """Operational execution step — purpose is intent, not chain-of-thought."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step_index: int = Field(ge=0)
    step_id: str = Field(min_length=1)
    purpose: ReportSafeText
    evidence_basis_ids: tuple[str, ...] = ()
    action: ReportSafeText
    input: ReportSafePayload | None = None
    observation: ReportSafePayload | None = None
    evidence_created_ids: tuple[str, ...] = ()
    status: ProofStepExecutionStatus
    started_at: datetime | None = None
    duration_ms: int | None = Field(default=None, ge=0)
    participant_id: str | None = None
    tool_invocation: ToolInvocationEvidence | None = None
    error: ReportSafeText | None = None

    @model_validator(mode="after")
    def _validate_required_text(self) -> ProofExecutionStep:
        ReportSafeText.require_non_empty(self.purpose, field_name="purpose")
        ReportSafeText.require_non_empty(self.action, field_name="action")
        return self


class FinalOutputEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    present: bool
    output_type: str = "text"
    content: ReportSafeText = Field(
        default_factory=lambda: proof_authored_report_safe_text("")
    )
    report_safe: bool = True
    evidence_basis_ids: tuple[str, ...] = ()


class EvaluatorCheckEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    check_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    passed: bool
    explanation: ReportSafeText = Field(
        default_factory=lambda: proof_authored_report_safe_text("")
    )
    evidence_ids: tuple[str, ...] = ()


class EvaluatorSummaryEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool
    checks: tuple[EvaluatorCheckEvidence, ...] = ()
    failure_reasons: tuple[str, ...] = ()


class FailureClassification(StrEnum):
    PLATFORM_DEFECT = "PLATFORM_DEFECT"
    PROOF_DEFECT = "PROOF_DEFECT"
    MODEL_BEHAVIOR_FAILURE = "MODEL_BEHAVIOR_FAILURE"
    PROVIDER_CONFIGURATION = "PROVIDER_CONFIGURATION"
    ENVIRONMENT = "ENVIRONMENT"
    EXPECTED_FALSIFICATION = "EXPECTED_FALSIFICATION"
    BLOCKED_CONFIGURATION = "BLOCKED_CONFIGURATION"
    UNKNOWN = "UNKNOWN"


class FailureEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    classification: FailureClassification
    boundary: str = ""
    message: ReportSafeText
    completed_milestones: tuple[str, ...] = ()
    failed_milestone: str | None = None
    skipped_not_reached: tuple[str, ...] = ()
    provider_error_code: str | None = None
    exception_type: str | None = None
    evidence_ids: tuple[str, ...] = ()
    safe_diagnostic: ReportSafeText | None = None

    @model_validator(mode="after")
    def _validate_message(self) -> FailureEvidence:
        ReportSafeText.require_non_empty(self.message, field_name="message")
        return self


class ScenarioEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    question: str = Field(min_length=1)
    expected_behavior: str = Field(min_length=1)
    falsification_condition: str = Field(min_length=1)
    execution_status: ProofEvidenceExecutionStatus
    metrics: tuple[MetricEvidence, ...] = ()
    steps: tuple[ProofExecutionStep, ...] = ()
    final_output: FinalOutputEvidence | None = None
    evaluator: EvaluatorSummaryEvidence | None = None
    failure: FailureEvidence | None = None


class ConclusionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    supported_conclusions: tuple[str, ...] = ()
    unsupported_conclusions: tuple[str, ...] = ()
    open_questions: tuple[str, ...] = ()


class ReproductionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_revision: str = Field(min_length=1)
    command: str = Field(min_length=1)
    prerequisites: tuple[str, ...] = ()
    required_env_variable_names: tuple[str, ...] = ()
    dataset_fingerprint_sha256: str | None = None


class ProvenanceEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_schema_version: Literal["intergrax.platform_proof_evidence.v1"] = (
        PLATFORM_PROOF_EVIDENCE_SCHEMA_VERSION
    )
    proof_id: str = Field(min_length=1)
    source_revision: str = Field(min_length=1)
    generated_at: datetime
    execution_id: str = Field(min_length=1)
    evidence_checksum: str | None = None
    artifact_identity: str = Field(min_length=1)


class EvidenceNodeKind(StrEnum):
    TOOL_RESULT = "TOOL_RESULT"
    DATASET = "DATASET"
    FINAL_ANSWER = "FINAL_ANSWER"
    CHECK = "CHECK"
    STEP = "STEP"
    OTHER = "OTHER"


class EvidenceRelationship(StrEnum):
    EVIDENCE_BASIS = "EVIDENCE_BASIS"
    PRODUCED_BY = "PRODUCED_BY"
    SUPPORTS_CONCLUSION = "SUPPORTS_CONCLUSION"


class EvidenceNode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str = Field(min_length=1)
    kind: EvidenceNodeKind
    label: str = Field(min_length=1)
    summary: ReportSafeText = Field(
        default_factory=lambda: proof_authored_report_safe_text("")
    )
    producing_step_id: str | None = None


class EvidenceEdge(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    from_evidence_id: str = Field(min_length=1)
    to_evidence_id: str | None = None
    to_step_id: str | None = None
    relationship: EvidenceRelationship

    @model_validator(mode="after")
    def _has_target(self) -> EvidenceEdge:
        if self.to_evidence_id is None and self.to_step_id is None:
            raise ValueError("EvidenceEdge requires to_evidence_id or to_step_id")
        return self


class EvidenceGraphEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    nodes: tuple[EvidenceNode, ...] = ()
    edges: tuple[EvidenceEdge, ...] = ()


class ToolsSqlObservationEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    call_index: int = Field(ge=0)
    tool_id: str = Field(min_length=1)
    sql_text: str = Field(min_length=1)
    output_preview: str = ""
    success: bool


class ToolsSqlInvestigationExtension(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    extension_id: Literal["tools.tool-call-trace"] = "tools.tool-call-trace"
    sql_statements: tuple[str, ...] = ()
    tool_observations: tuple[ToolsSqlObservationEvidence, ...] = ()
    investigation_proof_step_count: int = Field(ge=0, default=0)
    successful_tool_calls: int = Field(ge=0, default=0)
    stop_reason: str = ""
    follow_up_has_valid_basis: bool | None = None


class DomainExtensionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tools: ToolsSqlInvestigationExtension | None = None


class PlatformProofEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["intergrax.platform_proof_evidence.v1"] = (
        PLATFORM_PROOF_EVIDENCE_SCHEMA_VERSION
    )
    proof_identity: ProofIdentityEvidence
    execution: ExecutionMetadataEvidence
    claim: ProofClaimEvidence
    architecture: ArchitectureEvidence
    participants: tuple[ParticipantEvidence, ...]
    environment: EnvironmentEvidence
    scenarios: tuple[ScenarioEvidence, ...] = ()
    evidence_graph: EvidenceGraphEvidence = Field(default_factory=EvidenceGraphEvidence)
    final_output: FinalOutputEvidence | None = None
    evaluator: EvaluatorSummaryEvidence | None = None
    limitations: tuple[str, ...] = ()
    conclusion: ConclusionEvidence = Field(default_factory=ConclusionEvidence)
    reproduction: ReproductionEvidence
    provenance: ProvenanceEvidence
    failure: FailureEvidence | None = None
    domain_extension: DomainExtensionEvidence = Field(
        default_factory=DomainExtensionEvidence
    )

    @model_validator(mode="after")
    def _validate_invariants(self) -> PlatformProofEvidence:
        step_ids: set[str] = set()
        for scenario in self.scenarios:
            for step in scenario.steps:
                if step.step_id in step_ids:
                    raise ValueError(f"duplicate step_id: {step.step_id}")
                step_ids.add(step.step_id)

        evidence_ids = {node.evidence_id for node in self.evidence_graph.nodes}
        if len(evidence_ids) != len(self.evidence_graph.nodes):
            raise ValueError("duplicate evidence_id in evidence graph")

        known_step_ids = step_ids
        for node in self.evidence_graph.nodes:
            if node.producing_step_id and node.producing_step_id not in known_step_ids:
                raise ValueError(
                    f"dangling producing_step_id: {node.producing_step_id}"
                )

        for edge in self.evidence_graph.edges:
            if edge.from_evidence_id not in evidence_ids:
                raise ValueError(f"dangling from_evidence_id: {edge.from_evidence_id}")
            if edge.to_evidence_id is not None and edge.to_evidence_id not in evidence_ids:
                raise ValueError(f"dangling to_evidence_id: {edge.to_evidence_id}")
            if edge.to_step_id is not None and edge.to_step_id not in known_step_ids:
                raise ValueError(f"dangling to_step_id: {edge.to_step_id}")

        for scenario in self.scenarios:
            for step in scenario.steps:
                for basis_id in step.evidence_basis_ids:
                    if basis_id not in evidence_ids:
                        raise ValueError(f"dangling evidence_basis_id: {basis_id}")
                for created_id in step.evidence_created_ids:
                    if created_id not in evidence_ids:
                        raise ValueError(f"dangling evidence_created_id: {created_id}")

        if self.execution.status in {
            ProofEvidenceExecutionStatus.CRASH,
            ProofEvidenceExecutionStatus.BLOCKED,
        } and self.failure is None:
            raise ValueError(
                f"failure evidence required for execution status {self.execution.status.value}"
            )

        return self


def execution_status_from_proof_status(
    status: ProofStatus,
) -> ProofEvidenceExecutionStatus:
    mapping: dict[ProofStatus, ProofEvidenceExecutionStatus] = {
        ProofStatus.PASS: ProofEvidenceExecutionStatus.PASS,
        ProofStatus.FAIL: ProofEvidenceExecutionStatus.FAIL,
        ProofStatus.BLOCKED_ENVIRONMENT: ProofEvidenceExecutionStatus.BLOCKED,
        ProofStatus.BLOCKED_CONFIGURATION: ProofEvidenceExecutionStatus.BLOCKED,
        ProofStatus.SKIPPED_PLATFORM: ProofEvidenceExecutionStatus.BLOCKED,
        ProofStatus.SKIPPED_PROFILE: ProofEvidenceExecutionStatus.BLOCKED,
    }
    return mapping[status]
