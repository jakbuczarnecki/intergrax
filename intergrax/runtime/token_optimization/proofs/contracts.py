# © Artur Czarnecki. All rights reserved.

"""Immutable contracts and safe errors for the universal proof harness."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from intergrax.runtime.token_optimization.contracts import (
    TokenOptimizationRequest,
)

SCHEMA_VERSION = "token-optimization-proof.v1"
ROUTER_TERMINAL_NON_EXECUTION_REASONS = {
    "blocked": frozenset(
        {
            "policy_disabled",
            "profile_off",
            "unsupported_adapter",
            "capability_resolution_failed",
            "source_type_not_supported",
            "confidence_below_threshold",
            "packing_input_required",
            "lossy_not_allowed",
        }
    ),
    "review_required": frozenset(
        {"model_requested_review", "protected_regions_require_review"}
    ),
}


def is_terminal_router_non_execution(
    router_status: str | None,
    router_reason: str | None,
) -> bool:
    """Return whether routing ended safely before pipeline execution."""
    return (
        router_status in ROUTER_TERMINAL_NON_EXECUTION_REASONS
        and router_reason in ROUTER_TERMINAL_NON_EXECUTION_REASONS[router_status]
    )


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


_SAFE_EVIDENCE_STRING_RE = re.compile(r"^[A-Za-z0-9._:-]+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_POLICY_OVERRIDE_REASONS = frozenset({"security_warning_requires_review"})


def _validate_evidence_string(value: str | None, field_name: str) -> None:
    if value is not None and (
        not value or len(value) > 128 or not _SAFE_EVIDENCE_STRING_RE.fullmatch(value)
    ):
        raise ValueError(f"{field_name} must be a bounded safe evidence string")


def _validate_optional_digest(value: str | None, field_name: str) -> None:
    if value is not None and not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


def _validate_count(value: int, field_name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be an exact non-negative int")


@dataclass(frozen=True, slots=True)
class ProofRouterEvidence:
    """Redaction-safe model decision plus final policy outcome."""

    status: str | None = None
    configuration_id: str | None = None
    reason_code: str | None = None
    review_required: bool | None = None
    confidence: float | None = None
    risk: str | None = None
    transport: str | None = None
    structured_output_fallback_used: bool | None = None
    model_configuration_id: str | None = None
    model_reason_code: str | None = None
    model_risk: str | None = None
    model_review_required: bool | None = None
    policy_override_applied: bool = False
    policy_override_reason: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "status",
            "configuration_id",
            "reason_code",
            "risk",
            "transport",
        ):
            _validate_evidence_string(object.__getattribute__(self, field_name), field_name)
        if self.review_required is not None and type(self.review_required) is not bool:
            raise ValueError("review_required must be bool or None")
        if self.structured_output_fallback_used is not None and (
            type(self.structured_output_fallback_used) is not bool
        ):
            raise ValueError("structured_output_fallback_used must be bool or None")
        for field_name in (
            "model_configuration_id",
            "model_reason_code",
            "model_risk",
        ):
            _validate_evidence_string(object.__getattribute__(self, field_name), field_name)
        if self.model_review_required is not None and type(
            self.model_review_required
        ) is not bool:
            raise ValueError("model_review_required must be bool or None")
        if type(self.policy_override_applied) is not bool:
            raise ValueError("policy_override_applied must be bool")
        if self.policy_override_reason is not None and (
            self.policy_override_reason not in _POLICY_OVERRIDE_REASONS
        ):
            raise ValueError("unknown policy_override_reason")
        if self.policy_override_applied and self.policy_override_reason is None:
            raise ValueError("policy overrides require a reason")
        if not self.policy_override_applied and self.policy_override_reason is not None:
            raise ValueError("inactive policy overrides cannot have a reason")
        if self.confidence is not None:
            if isinstance(self.confidence, bool) or not math.isfinite(self.confidence):
                raise ValueError("confidence must be finite")
            if not 0.0 <= self.confidence <= 1.0:
                raise ValueError("confidence must be between 0.0 and 1.0")


@dataclass(frozen=True, slots=True)
class ProofPipelineEvidence:
    """Redaction-safe snapshot of canonical pipeline completion metadata."""

    completed: bool | None = None
    fallback_applied: bool | None = None
    validation_status: str | None = None
    validation_reason_code: str | None = None
    required_layer_failure: str | None = None
    receipt_completion_status: bool | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "validation_status",
            "validation_reason_code",
            "required_layer_failure",
        ):
            _validate_evidence_string(object.__getattribute__(self, field_name), field_name)
        for field_name in (
            "completed",
            "fallback_applied",
            "receipt_completion_status",
        ):
            value = object.__getattribute__(self, field_name)
            if value is not None and type(value) is not bool:
                raise ValueError(f"{field_name} must be bool or None")


@dataclass(frozen=True, slots=True)
class ProofProtectedRegionEvidence:
    """Counts and aggregate identity only; never protected values."""

    input_protected_region_count: int = 0
    validated_protected_region_count: int = 0
    preserved_protected_region_count: int = 0
    protected_region_validation_status: str | None = None
    input_identity_digest: str | None = None
    preserved_identity_digest: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "input_protected_region_count",
            "validated_protected_region_count",
            "preserved_protected_region_count",
        ):
            _validate_count(object.__getattribute__(self, field_name), field_name)
        if self.validated_protected_region_count > self.input_protected_region_count:
            raise ValueError("validated protected-region count exceeds input count")
        if self.preserved_protected_region_count > self.input_protected_region_count:
            raise ValueError("preserved protected-region count exceeds input count")
        _validate_evidence_string(
            self.protected_region_validation_status,
            "protected_region_validation_status",
        )
        _validate_optional_digest(self.input_identity_digest, "input_identity_digest")
        _validate_optional_digest(
            self.preserved_identity_digest,
            "preserved_identity_digest",
        )
        if self.input_protected_region_count == 0 and (
            self.input_identity_digest is not None
            or self.preserved_identity_digest is not None
        ):
            raise ValueError("zero protected regions cannot have identity digests")
        if self.preserved_identity_digest is not None and (
            self.preserved_identity_digest != self.input_identity_digest
        ):
            raise ValueError("preserved identity digest must match input identity digest")
        if (
            self.protected_region_validation_status == "passed"
            and self.input_protected_region_count > 0
            and (
                self.preserved_protected_region_count
                != self.input_protected_region_count
                or self.preserved_identity_digest is None
            )
        ):
            raise ValueError("passed protected validation requires complete preservation")


@dataclass(frozen=True, slots=True)
class ProofPrefixIdentityEvidence:
    """Redaction-safe propagation of TOKEN-10B prefix identity."""

    identity_available: bool = False
    stable_prefix_identity: str | None = None
    tool_schema_hash: str | None = None
    message_envelope_hash: str | None = None
    identity_contract_version: str | None = None

    def __post_init__(self) -> None:
        if type(self.identity_available) is not bool:
            raise ValueError("identity_available must be bool")
        for field_name in (
            "stable_prefix_identity",
            "tool_schema_hash",
            "message_envelope_hash",
        ):
            _validate_optional_digest(
                object.__getattribute__(self, field_name),
                field_name,
            )
        _validate_evidence_string(
            self.identity_contract_version,
            "identity_contract_version",
        )
        if not self.identity_available and any(
            value is not None
            for value in (
                self.stable_prefix_identity,
                self.tool_schema_hash,
                self.message_envelope_hash,
                self.identity_contract_version,
            )
        ):
            raise ValueError("unavailable prefix identity cannot carry identity values")
        if self.identity_available and (
            self.stable_prefix_identity is None
            or self.identity_contract_version is None
        ):
            raise ValueError("available prefix identity requires digest and version")


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
    router_evidence: ProofRouterEvidence = field(default_factory=ProofRouterEvidence)
    pipeline_evidence: ProofPipelineEvidence = field(
        default_factory=ProofPipelineEvidence
    )
    protected_region_evidence: ProofProtectedRegionEvidence = field(
        default_factory=ProofProtectedRegionEvidence
    )
    prefix_identity_evidence: ProofPrefixIdentityEvidence = field(
        default_factory=ProofPrefixIdentityEvidence
    )

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
