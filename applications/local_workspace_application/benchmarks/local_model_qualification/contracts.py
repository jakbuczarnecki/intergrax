# © Artur Czarnecki. All rights reserved.

"""Typed contracts for local model qualification benchmark results."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

BENCHMARK_ID = "LKW-CONVERSATIONAL-INTERACTION-1A-C2-LOCAL-MODEL-QUALIFICATION-MATRIX"
RESULT_SCHEMA_VERSION = "lkw.local_model_qualification.result.v1"
CORPUS_VERSION = "lkw.local_model_qualification.corpus.v1"
REPAIR_ATTEMPTS = 0


class StructuralFailureCategory(StrEnum):
    PROTOCOL_UNSUPPORTED = "PROTOCOL_UNSUPPORTED"
    MISSING_PLAN_TOOL_CALL = "MISSING_PLAN_TOOL_CALL"
    MULTIPLE_PLAN_TOOL_CALLS = "MULTIPLE_PLAN_TOOL_CALLS"
    UNEXPECTED_PLAN_TOOL = "UNEXPECTED_PLAN_TOOL"
    INVALID_TOOL_ARGUMENTS = "INVALID_TOOL_ARGUMENTS"
    DRAFT_VALIDATION_FAILED = "DRAFT_VALIDATION_FAILED"
    DRAFT_COMPILATION_FAILED = "DRAFT_COMPILATION_FAILED"
    CANONICAL_VALIDATION_FAILED = "CANONICAL_VALIDATION_FAILED"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"


class SemanticFailureCategory(StrEnum):
    SEMANTIC_MISMATCH = "SEMANTIC_MISMATCH"
    UNNECESSARY_WORKSPACE_ACTIVATE = "UNNECESSARY_WORKSPACE_ACTIVATE"
    UNEXPECTED_STATE_CHANGE = "UNEXPECTED_STATE_CHANGE"
    MISSING_REQUIRED_ACTION = "MISSING_REQUIRED_ACTION"
    WRONG_ACTION_TYPE = "WRONG_ACTION_TYPE"
    WRONG_ACTION_COUNT = "WRONG_ACTION_COUNT"
    WRONG_WORKSPACE_REFERENCE = "WRONG_WORKSPACE_REFERENCE"
    WRONG_SOURCE_EXTRACTION = "WRONG_SOURCE_EXTRACTION"
    WRONG_SOURCE_GROUPING = "WRONG_SOURCE_GROUPING"
    WRONG_ATTACHMENT_SELECTION = "WRONG_ATTACHMENT_SELECTION"
    WRONG_CANDIDATE_REFERENCE = "WRONG_CANDIDATE_REFERENCE"
    UNNECESSARY_CLARIFICATION = "UNNECESSARY_CLARIFICATION"
    MISSING_CLARIFICATION = "MISSING_CLARIFICATION"


class ModelStatus(StrEnum):
    COMPLETED = "COMPLETED"
    COMPLETED_WITH_FAILURES = "COMPLETED_WITH_FAILURES"
    NOT_INSTALLED = "NOT_INSTALLED"
    PULL_FAILED = "PULL_FAILED"
    MODEL_METADATA_UNAVAILABLE = "MODEL_METADATA_UNAVAILABLE"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"


class ProtocolStatus(StrEnum):
    QUALIFIED = "QUALIFIED"
    CONDITIONALLY_QUALIFIED = "CONDITIONALLY_QUALIFIED"
    NOT_QUALIFIED = "NOT_QUALIFIED"
    PROTOCOL_UNSUPPORTED = "PROTOCOL_UNSUPPORTED"
    SCHEMA_INCOMPATIBLE = "SCHEMA_INCOMPATIBLE"
    WARMUP_FAILED = "WARMUP_FAILED"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    NOT_RUN = "NOT_RUN"


class SchemaProbeStatus(StrEnum):
    PASS = "PASS"
    PROTOCOL_UNSUPPORTED = "PROTOCOL_UNSUPPORTED"
    SCHEMA_INCOMPATIBLE = "SCHEMA_INCOMPATIBLE"
    PROVIDER_ERROR = "PROVIDER_ERROR"


class WarmupStatus(StrEnum):
    PASS = "PASS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class CaseExecutionStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"


class ObservedExecutionMode(StrEnum):
    FULL_GPU = "FULL_GPU"
    PARTIAL_GPU_OFFLOAD = "PARTIAL_GPU_OFFLOAD"
    CPU_ONLY = "CPU_ONLY"
    UNKNOWN = "UNKNOWN"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LatencyStats(_FrozenModel):
    minimum: float
    median: float
    p95: float
    maximum: float


class WorkspaceReferenceSummary(_FrozenModel):
    kind: str
    value: str | None


class CaseResult(_FrozenModel):
    case_id: str
    repetition: int
    status: CaseExecutionStatus
    primary_failure_category: str | None = None
    failure_categories: tuple[str, ...] = ()
    latency_ms: float
    action_types: tuple[str, ...] = ()
    object_types: tuple[str, ...] = ()
    workspace_references: tuple[WorkspaceReferenceSummary, ...] = ()
    clarification_count: int = 0
    unsafe_state_change_count: int = 0
    error_type: str | None = None


class ProtocolResult(_FrozenModel):
    protocol: str
    capability_supported: bool
    schema_probe_status: SchemaProbeStatus
    warmup_status: WarmupStatus
    qualification_status: ProtocolStatus
    case_count: int
    pass_count: int
    failure_count: int
    semantic_success_rate: float
    invalid_draft_count: int
    provider_failure_count: int
    unsafe_state_change_count: int
    failure_category_counts: dict[str, int] = Field(default_factory=dict)
    latency_ms: LatencyStats
    case_results: tuple[CaseResult, ...] = ()


class ModelMetadata(_FrozenModel):
    digest: str | None = None
    artifact_size_bytes: int | None = None
    parameter_size: str | None = None
    quantization_level: str | None = None
    model_family: str | None = None
    context_length: int | None = None
    loaded_size_bytes: int | None = None
    size_vram_bytes: int | None = None


class ModelResult(_FrozenModel):
    name: str
    role: str
    installed: bool
    metadata: ModelMetadata
    declared_capabilities: tuple[str, ...] = ()
    observed_execution_mode: ObservedExecutionMode
    status: ModelStatus
    protocols: tuple[ProtocolResult, ...] = ()


class HostMetadata(_FrozenModel):
    operating_system: str
    os_release: str
    machine_architecture: str
    python_version: str
    cpu_description: str | None = None
    total_system_ram_bytes: int | None = None
    gpu_name: str | None = None
    gpu_total_vram_bytes: int | None = None
    nvidia_driver_version: str | None = None


class OllamaEnvironment(_FrozenModel):
    version: str | None = None
    host: str


class QualificationSummary(_FrozenModel):
    recommended_model: str | None = None
    recommended_protocol: str | None = None
    conditional_candidates: tuple[str, ...] = ()
    message: str


class LocalModelQualificationResult(_FrozenModel):
    schema_version: str = RESULT_SCHEMA_VERSION
    benchmark_id: str = BENCHMARK_ID
    generated_at_utc: str
    generated_from_commit: str | None
    configuration_sha256: str
    corpus_version: str = CORPUS_VERSION
    repair_attempts: int = REPAIR_ATTEMPTS
    host: HostMetadata
    ollama: OllamaEnvironment
    models: tuple[ModelResult, ...]
    summary: QualificationSummary

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
