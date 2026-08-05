# © Artur Czarnecki. All rights reserved.

"""Immutable contracts and safe errors for the universal proof harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from intergrax.runtime.token_optimization.contracts import (
    TokenOptimizationRequest,
)

SCHEMA_VERSION = "token-optimization-proof.v1"


class ProofError(Exception):
    """Base error whose public text contains only a stable reason code."""

    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)


class ProofConfigurationError(ProofError):
    """The TOML proof configuration is invalid or unavailable."""


class ProofCompositionError(ProofError):
    """The configured production components cannot be composed."""


class ProofExecutionError(ProofError):
    """A proof case failed without exposing provider or content data."""


class ProofProviderUnavailableError(ProofError):
    """The requested live provider cannot be constructed or reached."""


class ProofArtifactError(ProofError):
    """Safe artifact persistence failed."""


@dataclass(frozen=True, slots=True)
class ProofAdapterConfig:
    adapter_id: str
    provider: str
    adapter_type: str
    model: str
    base_url: str
    api_key_env: str | None
    timeout_seconds: float
    max_output_tokens: int
    temperature: float


@dataclass(frozen=True, slots=True)
class ProofRouterConfig:
    enabled: bool
    configuration_id: str
    minimum_confidence: float
    allow_structured_output_fallback: bool
    require_review_for_protected_lossy_content: bool


@dataclass(frozen=True, slots=True)
class ProofPipelineConfig:
    mode: str
    layer_ids: tuple[str, ...]
    failure_policy: str


@dataclass(frozen=True, slots=True)
class ProofOutputConfig:
    directory: Path
    fail_if_exists: bool = True


@dataclass(frozen=True, slots=True)
class ProofCaseInput:
    """A case carrying the canonical request, never its raw content in repr."""

    case_id: str
    request: TokenOptimizationRequest = field(repr=False)
    tags: tuple[str, ...] = ()

    def __repr__(self) -> str:
        return (
            f"ProofCaseInput(case_id={self.case_id!r}, "
            f"request=<redacted>, tags={self.tags!r})"
        )


@dataclass(frozen=True, slots=True)
class UniversalTokenOptimizationProofConfig:
    schema_version: str
    proof_id: str
    run_mode: str
    adapter: ProofAdapterConfig
    router: ProofRouterConfig
    pipeline: ProofPipelineConfig
    output: ProofOutputConfig
    cases: tuple[ProofCaseInput, ...]
    case_source: Path | None
    source_path: Path

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported proof schema version")
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("duplicate case IDs are not allowed")
        if not self.cases and self.case_source is None:
            raise ValueError("cases or case_source is required")
        if self.cases and self.case_source is not None:
            raise ValueError("cases and case_source are mutually exclusive")


@dataclass(frozen=True, slots=True)
class ProofMeasurement:
    available: bool = False
    value: int | None = None

    def __post_init__(self) -> None:
        if self.available and self.value is None:
            raise ValueError("available measurements require a value")
        if not self.available and self.value is not None:
            raise ValueError("unavailable measurements must have value=None")
        if self.value is not None and self.value < 0:
            raise ValueError("measurement values cannot be negative")


@dataclass(frozen=True, slots=True)
class UniversalProofEnvironmentSummary:
    provider: str
    model: str
    adapter_available: bool
    network_required: bool
    raw_content_included: bool = False

    def __post_init__(self) -> None:
        if self.raw_content_included:
            raise ValueError("proof environment must remain redaction-safe")


@dataclass(frozen=True, slots=True)
class UniversalProofCaseResult:
    case_id: str
    status: str
    router_status: str | None
    router_reason: str | None
    selected_configuration_id: str | None
    pipeline_status: str
    applied_layer_ids: tuple[str, ...]
    baseline_measurement: ProofMeasurement = field(default_factory=ProofMeasurement)
    optimized_measurement: ProofMeasurement = field(default_factory=ProofMeasurement)
    receipt_refs: tuple[str, ...] = ()
    error_reason_code: str | None = None
    raw_content_included: bool = False

    def __post_init__(self) -> None:
        if self.raw_content_included:
            raise ValueError("proof case result must remain redaction-safe")


@dataclass(frozen=True, slots=True)
class ProofArtifactRef:
    path: str
    sha256: str | None = None


@dataclass(frozen=True, slots=True)
class UniversalProofArtifactManifest:
    files: tuple[ProofArtifactRef, ...]
    raw_content_included: bool = False

    def __post_init__(self) -> None:
        if self.raw_content_included:
            raise ValueError("proof manifest must remain redaction-safe")
        paths = [item.path for item in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("duplicate artifact paths are not allowed")


@dataclass(frozen=True, slots=True)
class UniversalProofRunResult:
    schema_version: str
    proof_id: str
    run_id: str
    run_mode: str
    started_at: datetime
    completed_at: datetime
    adapter_id: str
    model: str
    case_count: int
    completed_count: int
    failed_count: int
    cases: tuple[UniversalProofCaseResult, ...]
    environment: UniversalProofEnvironmentSummary
    artifact_manifest: UniversalProofArtifactManifest
    success: bool
    raw_content_included: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported proof schema version")
        if self.raw_content_included:
            raise ValueError("proof run result must remain redaction-safe")
        if self.case_count != len(self.cases):
            raise ValueError("case_count must match cases")
        if self.completed_count + self.failed_count != self.case_count:
            raise ValueError("case counts do not balance")
