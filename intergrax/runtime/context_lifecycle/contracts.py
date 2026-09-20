# © Artur Czarnecki. All rights reserved.

"""Unified Context Lifecycle shared contracts (CTX-UCL-1)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, StrEnum
from types import MappingProxyType
from typing import Any, TypeVar


_FORBIDDEN_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "content",
        "raw_content",
        "prompt",
        "raw_prompt",
        "messages",
        "source_text",
        "summary",
        "raw_summary",
        "tool_args",
        "evidence",
        "document_content",
    }
)

EnumT = TypeVar("EnumT", bound=Enum)
ContractT = TypeVar("ContractT")


def _require_enum(
    value: object,
    enum_type: type[EnumT],
    field_name: str,
) -> EnumT:
    if not isinstance(value, enum_type):
        raise ValueError(f"{field_name} must be {enum_type.__name__}")
    return value


def _require_instance(
    value: object,
    expected_type: type[ContractT],
    field_name: str,
) -> ContractT:
    if not isinstance(value, expected_type):
        raise ValueError(f"{field_name} must be {expected_type.__name__}")
    return value


def _require_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _require_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _require_non_empty(value: str, field_name: str) -> str:
    if not value:
        raise ValueError(f"{field_name} must be non-empty")
    return value


def _require_non_empty_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_strict_non_empty_text(value: object, field_name: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{field_name} must be a string")
    if not value:
        raise ValueError(f"{field_name} must be non-empty")
    if value != value.strip():
        raise ValueError(f"{field_name} must not contain surrounding whitespace")
    return value


def _require_strict_non_empty_texts(
    values: object,
    field_name: str,
) -> tuple[str, ...]:
    if type(values) not in (tuple, list):
        raise ValueError(f"{field_name} must be a sequence of strings")
    normalized = tuple(
        _require_strict_non_empty_text(value, f"{field_name} item") for value in values
    )
    return _reject_duplicates(normalized, field_name)


def _require_non_empty_texts(
    values: object,
    field_name: str,
) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise ValueError(f"{field_name} must be a sequence of strings")
    normalized = tuple(
        _require_non_empty_text(value, f"{field_name} item") for value in values
    )
    return _reject_duplicates(normalized, field_name)


def _require_non_negative(value: object, field_name: str) -> int:
    int_value = _require_int(value, field_name)
    if int_value < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return int_value


def _require_positive(value: object, field_name: str) -> int:
    int_value = _require_int(value, field_name)
    if int_value <= 0:
        raise ValueError(f"{field_name} must be > 0")
    return int_value


def _require_finite_quality_score(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("minimum_quality_score must be a number")
    score = float(value) if isinstance(value, int) else value
    if not math.isfinite(score):
        raise ValueError("minimum_quality_score must be finite")
    if score < 0.0 or score > 1.0:
        raise ValueError("minimum_quality_score must be between 0.0 and 1.0")
    return score


def _require_timezone_aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


def _reject_duplicates(values: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return values


def _normalize_safe_metadata(metadata: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if metadata is None:
        return MappingProxyType({})

    def _normalize_value(value: Any) -> Any:
        if isinstance(value, Enum):
            raise ValueError("safe_metadata must not contain enum values")

        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("safe_metadata must not contain non-finite floats")
            return value
        if isinstance(value, bytes):
            raise ValueError("safe_metadata must not contain bytes")
        if isinstance(value, (set, frozenset)):
            raise ValueError("safe_metadata must not contain sets")
        if isinstance(value, datetime):
            raise ValueError("safe_metadata must not contain datetime values")
        if isinstance(value, Mapping):
            return _normalize_mapping(value)
        if isinstance(value, (list, tuple)):
            return tuple(_normalize_value(item) for item in value)
        raise ValueError("safe_metadata must contain only JSON-serializable values")

    def _normalize_mapping(mapping: Mapping[Any, Any]) -> MappingProxyType:
        normalized: dict[str, Any] = {}
        for key, value in mapping.items():
            if not isinstance(key, str):
                raise ValueError("safe_metadata keys must be strings")
            if key.casefold() in _FORBIDDEN_METADATA_KEYS:
                raise ValueError(f"safe_metadata must not contain forbidden key: {key}")
            normalized[key] = _normalize_value(value)
        return MappingProxyType(normalized)

    return _normalize_mapping(metadata)


class ModelCallExecutionScope(StrEnum):
    """Typed execution scope for model invocations."""

    PRIMARY_MODEL_CALL = "primary_model_call"
    INTERNAL_OPTIMIZATION_CALL = "internal_optimization_call"


from intergrax.contracts.context_optimization_policy import (
    ArtifactCreationCoordinationStatus,
    ContextOptimizationDecision,
    ContextOptimizationMode,
    ContextOptimizationPolicy,
    DurableCompactionActivationMode,
    DurableCompactionEligibilityReasonCode,
    DurableCompactionPolicy,
    DurableCompactionValidationRequirement,
    EphemeralArtifactPersistencePolicy,
    OptimizationArtifactType,
)

class ReusableArtifactStatus(StrEnum):
    """Lifecycle status for reusable optimization artifacts."""

    VALIDATED = "validated"
    INVALIDATED = "invalidated"
    RETIRED = "retired"


class ArtifactValidationStatus(StrEnum):
    """Validation outcome for reusable artifacts."""

    PASSED = "passed"
    FAILED = "failed"
    REVOKED = "revoked"


class ArtifactCompatibilityStatus(StrEnum):
    """Compatibility evaluation outcome."""

    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"


class ContextOptimizationReasonCode(StrEnum):
    """Reason codes for optimization guard and coordination failures."""

    OPTIMIZATION_RECURSION_BLOCKED = "optimization_recursion_blocked"
    OPTIMIZATION_DEPTH_EXCEEDED = "optimization_depth_exceeded"
    DUPLICATE_ACTIVE_ARTIFACT_CREATION = "duplicate_active_artifact_creation"
    ARTIFACT_CREATION_IN_PROGRESS = "artifact_creation_in_progress"
    ARTIFACT_CREATION_RESERVATION_CONFLICT = "artifact_creation_reservation_conflict"
    ARTIFACT_CREATION_LEASE_EXPIRED = "artifact_creation_lease_expired"
    ARTIFACT_CREATION_FAILED = "artifact_creation_failed"


class DurableCompactionSourceIdentity:
    """Immutable durable target identity; durable sources use source_refs, not ranges."""

    tenant_id: str
    context_scope_id: str
    source_revision: int
    expected_active_revision: int
    source_refs: tuple[str, ...]
    source_content_hash: str
    artifact_lookup_key: ArtifactLookupKey
    strategy_id: str
    strategy_version: str
    lossiness_profile: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tenant_id",
            _require_strict_non_empty_text(self.tenant_id, "tenant_id"),
        )
        object.__setattr__(
            self,
            "context_scope_id",
            _require_strict_non_empty_text(self.context_scope_id, "context_scope_id"),
        )
        object.__setattr__(
            self,
            "source_revision",
            _require_non_negative(self.source_revision, "source_revision"),
        )
        object.__setattr__(
            self,
            "expected_active_revision",
            _require_positive(self.expected_active_revision, "expected_active_revision"),
        )
        object.__setattr__(
            self,
            "source_refs",
            _require_non_empty_source_refs(self.source_refs, "source_refs"),
        )
        object.__setattr__(
            self,
            "source_content_hash",
            _require_sha256_hex(self.source_content_hash, "source_content_hash"),
        )
        lookup_key = _require_instance(
            self.artifact_lookup_key,
            ArtifactLookupKey,
            "artifact_lookup_key",
        )
        object.__setattr__(self, "artifact_lookup_key", lookup_key)
        object.__setattr__(
            self,
            "strategy_id",
            _require_strict_non_empty_text(self.strategy_id, "strategy_id"),
        )
        object.__setattr__(
            self,
            "strategy_version",
            _require_strict_non_empty_text(self.strategy_version, "strategy_version"),
        )
        object.__setattr__(
            self,
            "lossiness_profile",
            _require_strict_non_empty_text(self.lossiness_profile, "lossiness_profile"),
        )

        if lookup_key.tenant_id != self.tenant_id:
            raise ValueError("artifact_lookup_key.tenant_id must match tenant_id")
        if lookup_key.context_scope_id != self.context_scope_id:
            raise ValueError("artifact_lookup_key.context_scope_id must match context_scope_id")
        if lookup_key.source_content_hash != self.source_content_hash:
            raise ValueError("artifact_lookup_key.source_content_hash must match source_content_hash")
        if lookup_key.strategy_id != self.strategy_id:
            raise ValueError("artifact_lookup_key.strategy_id must match strategy_id")
        if lookup_key.strategy_version != self.strategy_version:
            raise ValueError("artifact_lookup_key.strategy_version must match strategy_version")
        if lookup_key.lossiness_profile != self.lossiness_profile:
            raise ValueError("artifact_lookup_key.lossiness_profile must match lossiness_profile")
        if lookup_key.source_refs != self.source_refs:
            raise ValueError("artifact_lookup_key.source_refs must match source_refs")


@dataclass(frozen=True, slots=True)
class DurableCompactionStabilityEvidence:
    """Observed immutable evidence that a durable source has stabilized."""

    observed_stable_revision_count: int
    observed_source_revision: int
    observed_source_content_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observed_stable_revision_count",
            _require_positive(
                self.observed_stable_revision_count,
                "observed_stable_revision_count",
            ),
        )
        object.__setattr__(
            self,
            "observed_source_revision",
            _require_non_negative(self.observed_source_revision, "observed_source_revision"),
        )
        object.__setattr__(
            self,
            "observed_source_content_hash",
            _require_sha256_hex(
                self.observed_source_content_hash,
                "observed_source_content_hash",
            ),
        )


@dataclass(frozen=True, slots=True)
class DurableCompactionEligibilityDecision:
    """Immutable durable compaction eligibility outcome."""

    eligible: bool
    reason_code: DurableCompactionEligibilityReasonCode | None
    policy_hash: str
    target_identity_hash: str
    evaluated_mode: ContextOptimizationMode

    def __post_init__(self) -> None:
        object.__setattr__(self, "eligible", _require_bool(self.eligible, "eligible"))
        object.__setattr__(
            self,
            "evaluated_mode",
            _require_enum(self.evaluated_mode, ContextOptimizationMode, "evaluated_mode"),
        )
        object.__setattr__(
            self,
            "policy_hash",
            _require_sha256_hex(self.policy_hash, "policy_hash"),
        )
        object.__setattr__(
            self,
            "target_identity_hash",
            _require_sha256_hex(self.target_identity_hash, "target_identity_hash"),
        )
        if self.eligible:
            if self.reason_code is not None:
                raise ValueError("eligible decision requires reason_code is None")
        elif self.reason_code is None:
            raise ValueError("ineligible decision requires reason_code")
        elif not isinstance(self.reason_code, DurableCompactionEligibilityReasonCode):
            raise ValueError("reason_code must be DurableCompactionEligibilityReasonCode")


@dataclass(frozen=True, slots=True)
class DurableCompactionActivationRequirements:
    """Immutable activation safety prerequisites for future durable compaction."""

    expected_active_revision: int
    candidate_artifact_id: str
    validated_artifact_id: str
    lineage_reference: str
    creation_receipt_reference: str
    rollback_source_reference: str
    raw_content_included: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "expected_active_revision",
            _require_positive(self.expected_active_revision, "expected_active_revision"),
        )
        object.__setattr__(
            self,
            "candidate_artifact_id",
            _require_strict_non_empty_text(
                self.candidate_artifact_id,
                "candidate_artifact_id",
            ),
        )
        object.__setattr__(
            self,
            "validated_artifact_id",
            _require_strict_non_empty_text(
                self.validated_artifact_id,
                "validated_artifact_id",
            ),
        )
        object.__setattr__(
            self,
            "lineage_reference",
            _require_strict_non_empty_text(self.lineage_reference, "lineage_reference"),
        )
        object.__setattr__(
            self,
            "creation_receipt_reference",
            _require_strict_non_empty_text(
                self.creation_receipt_reference,
                "creation_receipt_reference",
            ),
        )
        object.__setattr__(
            self,
            "rollback_source_reference",
            _require_strict_non_empty_text(
                self.rollback_source_reference,
                "rollback_source_reference",
            ),
        )
        if _require_bool(self.raw_content_included, "raw_content_included") is not False:
            raise ValueError("raw_content_included must be False")


def assess_durable_compaction_eligibility(
    *,
    policy: ContextOptimizationPolicy,
    target: DurableCompactionSourceIdentity,
    stability_evidence: DurableCompactionStabilityEvidence | None = None,
    raw_content_included: bool = False,
    expected_policy_hash: str | None = None,
    expected_target_identity_hash: str | None = None,
) -> DurableCompactionEligibilityDecision:
    """Evaluate durable compaction eligibility without executing compaction."""
    policy = _require_instance(policy, ContextOptimizationPolicy, "policy")
    target = _require_instance(
        target,
        DurableCompactionSourceIdentity,
        "target",
    )
    from intergrax.runtime.context_lifecycle.serialization import (
        compute_durable_compaction_policy_hash,
        compute_durable_compaction_source_identity_hash,
    )

    durable_policy = policy.durable_compaction or DurableCompactionPolicy()
    actual_policy_hash = compute_durable_compaction_policy_hash(durable_policy)
    actual_target_hash = compute_durable_compaction_source_identity_hash(target)
    if expected_policy_hash is not None:
        expected_policy_hash = _require_sha256_hex(expected_policy_hash, "expected_policy_hash")
    if expected_target_identity_hash is not None:
        expected_target_identity_hash = _require_sha256_hex(
            expected_target_identity_hash,
            "expected_target_identity_hash",
        )
    raw_content_included = _require_bool(raw_content_included, "raw_content_included")
    base_kwargs = {
        "policy_hash": actual_policy_hash,
        "target_identity_hash": actual_target_hash,
        "evaluated_mode": policy.mode,
    }

    def _ineligible(
        reason_code: DurableCompactionEligibilityReasonCode,
    ) -> DurableCompactionEligibilityDecision:
        return DurableCompactionEligibilityDecision(
            eligible=False,
            reason_code=reason_code,
            **base_kwargs,
        )

    if expected_policy_hash is not None and actual_policy_hash != expected_policy_hash:
        return _ineligible(
            DurableCompactionEligibilityReasonCode.POLICY_TARGET_IDENTITY_MISMATCH
        )
    if (
        expected_target_identity_hash is not None
        and actual_target_hash != expected_target_identity_hash
    ):
        return _ineligible(
            DurableCompactionEligibilityReasonCode.POLICY_TARGET_IDENTITY_MISMATCH
        )

    if raw_content_included:
        return _ineligible(DurableCompactionEligibilityReasonCode.RAW_CONTENT_FORBIDDEN)

    if not policy.enabled:
        return _ineligible(DurableCompactionEligibilityReasonCode.POLICY_DISABLED)

    if policy.mode is not ContextOptimizationMode.DURABLE_COMPACTION:
        return _ineligible(DurableCompactionEligibilityReasonCode.WRONG_OPTIMIZATION_MODE)

    if policy.durable_compaction is None or not durable_policy.enabled:
        return _ineligible(DurableCompactionEligibilityReasonCode.DURABLE_COMPACTION_DISABLED)

    if target.source_revision < 0:
        return _ineligible(DurableCompactionEligibilityReasonCode.MISSING_SOURCE_REVISION)

    if target.expected_active_revision <= 0:
        return _ineligible(
            DurableCompactionEligibilityReasonCode.MISSING_EXPECTED_ACTIVE_REVISION
        )

    if target.artifact_lookup_key.artifact_type not in policy.allowed_artifact_types:
        return _ineligible(DurableCompactionEligibilityReasonCode.ARTIFACT_TYPE_NOT_ALLOWED)

    if target.artifact_lookup_key.artifact_type is not OptimizationArtifactType.MESSAGE_SEQUENCE:
        return _ineligible(DurableCompactionEligibilityReasonCode.ARTIFACT_TYPE_NOT_ALLOWED)

    if target.strategy_id not in policy.allowed_strategy_ids:
        return _ineligible(DurableCompactionEligibilityReasonCode.STRATEGY_NOT_ALLOWED)

    if target.strategy_id not in durable_policy.allowed_strategy_ids:
        return _ineligible(DurableCompactionEligibilityReasonCode.STRATEGY_NOT_ALLOWED)

    if target.lossiness_profile not in durable_policy.allowed_lossiness_profiles:
        return _ineligible(DurableCompactionEligibilityReasonCode.LOSSINESS_NOT_ALLOWED)

    if (
        target.lossiness_profile in _SUPPORTED_DURABLE_LOSSINESS_PROFILES
        and not policy.allow_lossy
    ):
        return _ineligible(DurableCompactionEligibilityReasonCode.LOSSINESS_NOT_ALLOWED)

    if target.strategy_id in _DURABLE_LLM_SUMMARY_STRATEGY_IDS:
        if not policy.allow_llm_summarization:
            return _ineligible(DurableCompactionEligibilityReasonCode.LOSSINESS_NOT_ALLOWED)

    if stability_evidence is None or not isinstance(
        stability_evidence,
        DurableCompactionStabilityEvidence,
    ):
        return _ineligible(
            DurableCompactionEligibilityReasonCode.STABILITY_REQUIREMENT_UNAVAILABLE
        )

    if (
        stability_evidence.observed_source_revision != target.source_revision
        or stability_evidence.observed_source_content_hash != target.source_content_hash
    ):
        return _ineligible(
            DurableCompactionEligibilityReasonCode.STABILITY_REQUIREMENT_UNAVAILABLE
        )

    if (
        stability_evidence.observed_stable_revision_count
        < durable_policy.minimum_stable_revision_count
    ):
        return _ineligible(
            DurableCompactionEligibilityReasonCode.STABILITY_REQUIREMENT_NOT_MET
        )

    if not policy.require_receipt:
        return _ineligible(DurableCompactionEligibilityReasonCode.RECEIPT_REQUIREMENT_UNAVAILABLE)

    if not policy.require_rollback_metadata:
        return _ineligible(DurableCompactionEligibilityReasonCode.ROLLBACK_REQUIREMENT_UNAVAILABLE)

    return DurableCompactionEligibilityDecision(
        eligible=True,
        reason_code=None,
        **base_kwargs,
    )


class ArtifactCompatibilityReason(StrEnum):
    """Reason codes for artifact compatibility evaluation."""

    TENANT_SCOPE_MISMATCH = "tenant_scope_mismatch"
    CONTEXT_SCOPE_MISMATCH = "context_scope_mismatch"
    ARTIFACT_TYPE_MISMATCH = "artifact_type_mismatch"
    SOURCE_IDENTITY_MISMATCH = "source_identity_mismatch"
    SOURCE_CONTENT_HASH_MISMATCH = "source_content_hash_mismatch"
    STRATEGY_MISMATCH = "strategy_mismatch"
    POLICY_VERSION_MISMATCH = "policy_version_mismatch"
    VALIDATION_CONTRACT_VERSION_MISMATCH = "validation_contract_version_mismatch"
    COMPRESSION_TARGET_INSUFFICIENT = "compression_target_insufficient"
    LOSSINESS_PROFILE_MISMATCH = "lossiness_profile_mismatch"
    PROTECTED_REGION_POLICY_MISMATCH = "protected_region_policy_mismatch"
    MODEL_FAMILY_MISMATCH = "model_family_mismatch"
    LOCALE_MISMATCH = "locale_mismatch"
    ARTIFACT_NOT_VALID = "artifact_not_valid"
    ARTIFACT_INVALIDATED = "artifact_invalidated"
    ARTIFACT_RETIRED = "artifact_retired"


@dataclass(frozen=True, slots=True)
class ArtifactSourceRange:
    """Inclusive sequence range for artifact source identity."""

    start_sequence: int
    end_sequence: int

    def __post_init__(self) -> None:
        start = _require_non_negative(self.start_sequence, "start_sequence")
        end = _require_non_negative(self.end_sequence, "end_sequence")
        if start > end:
            raise ValueError("start_sequence must be <= end_sequence")
        object.__setattr__(self, "start_sequence", start)
        object.__setattr__(self, "end_sequence", end)


class UclArtifactOwnershipKind(StrEnum):
    """Canonical UCL artifact workspace ownership classification."""

    WORKSPACE = "workspace"
    LEGACY_UNKNOWN = "legacy_unknown"


@dataclass(frozen=True, slots=True)
class UclArtifactOwnershipScope:
    """Immutable workspace ownership scope for UCL optimization artifacts."""

    tenant_id: str
    workspace_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "tenant_id", _require_strict_non_empty_text(self.tenant_id, "tenant_id"))
        object.__setattr__(
            self,
            "workspace_id",
            _require_strict_non_empty_text(self.workspace_id, "workspace_id"),
        )


@dataclass(frozen=True, slots=True)
class UclArtifactOwnership:
    """Canonical persisted ownership fact for a reusable optimization artifact."""

    kind: UclArtifactOwnershipKind
    scope: UclArtifactOwnershipScope | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kind",
            _require_enum(self.kind, UclArtifactOwnershipKind, "kind"),
        )
        if self.kind is UclArtifactOwnershipKind.WORKSPACE:
            if self.scope is None:
                raise ValueError("WORKSPACE ownership requires scope")
            object.__setattr__(
                self,
                "scope",
                _require_instance(self.scope, UclArtifactOwnershipScope, "scope"),
            )
        elif self.kind is UclArtifactOwnershipKind.LEGACY_UNKNOWN:
            if self.scope is not None:
                raise ValueError("LEGACY_UNKNOWN ownership must not include scope")

    @staticmethod
    def for_workspace(scope: UclArtifactOwnershipScope) -> UclArtifactOwnership:
        return UclArtifactOwnership(
            kind=UclArtifactOwnershipKind.WORKSPACE,
            scope=scope,
        )

    @staticmethod
    def legacy_unknown() -> UclArtifactOwnership:
        return UclArtifactOwnership(kind=UclArtifactOwnershipKind.LEGACY_UNKNOWN)


@dataclass(frozen=True, slots=True)
class ArtifactCompressionTarget:
    """Compression target specification for artifact identity."""

    target_tokens: int | None = None
    budget_class: str | None = None

    def __post_init__(self) -> None:
        has_tokens = self.target_tokens is not None
        has_budget = self.budget_class is not None
        if has_tokens == has_budget:
            raise ValueError("exactly one of target_tokens or budget_class must be provided")
        if has_tokens:
            object.__setattr__(
                self,
                "target_tokens",
                _require_positive(self.target_tokens, "target_tokens"),  # type: ignore[arg-type]
            )
        if has_budget:
            object.__setattr__(
                self,
                "budget_class",
                _require_non_empty(self.budget_class, "budget_class"),  # type: ignore[arg-type]
            )


@dataclass(frozen=True, slots=True)
class ArtifactLookupKey:
    """Canonical artifact compatibility identity for catalog lookup."""

    tenant_id: str
    context_scope_id: str
    artifact_type: OptimizationArtifactType
    source_content_hash: str
    strategy_id: str
    strategy_version: str
    policy_version: str
    validation_contract_version: str
    compression_target: ArtifactCompressionTarget
    lossiness_profile: str
    source_refs: tuple[str, ...] = ()
    source_range: ArtifactSourceRange | None = None
    protected_region_policy_version: str | None = None
    model_family: str | None = None
    locale: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "artifact_type",
            _require_enum(self.artifact_type, OptimizationArtifactType, "artifact_type"),
        )
        object.__setattr__(
            self,
            "compression_target",
            _require_instance(
                self.compression_target,
                ArtifactCompressionTarget,
                "compression_target",
            ),
        )
        if self.source_range is not None:
            object.__setattr__(
                self,
                "source_range",
                _require_instance(self.source_range, ArtifactSourceRange, "source_range"),
            )

        object.__setattr__(self, "tenant_id", _require_non_empty(self.tenant_id, "tenant_id"))
        object.__setattr__(
            self,
            "context_scope_id",
            _require_non_empty(self.context_scope_id, "context_scope_id"),
        )
        object.__setattr__(
            self,
            "source_content_hash",
            _require_non_empty(self.source_content_hash, "source_content_hash"),
        )
        object.__setattr__(self, "strategy_id", _require_non_empty(self.strategy_id, "strategy_id"))
        object.__setattr__(
            self,
            "strategy_version",
            _require_non_empty(self.strategy_version, "strategy_version"),
        )
        object.__setattr__(
            self,
            "policy_version",
            _require_non_empty(self.policy_version, "policy_version"),
        )
        object.__setattr__(
            self,
            "validation_contract_version",
            _require_non_empty(self.validation_contract_version, "validation_contract_version"),
        )
        object.__setattr__(
            self,
            "lossiness_profile",
            _require_non_empty(self.lossiness_profile, "lossiness_profile"),
        )

        has_refs = bool(self.source_refs)
        has_range = self.source_range is not None
        if has_refs == has_range:
            raise ValueError("exactly one of source_refs or source_range must be provided")

        if has_refs:
            refs = tuple(self.source_refs)
            if any(not ref for ref in refs):
                raise ValueError("source_refs must not contain empty values")
            object.__setattr__(self, "source_refs", _reject_duplicates(refs, "source_refs"))

        if self.protected_region_policy_version is not None:
            object.__setattr__(
                self,
                "protected_region_policy_version",
                _require_non_empty(
                    self.protected_region_policy_version,
                    "protected_region_policy_version",
                ),
            )
        if self.model_family is not None:
            object.__setattr__(
                self,
                "model_family",
                _require_non_empty(self.model_family, "model_family"),
            )
        if self.locale is not None:
            object.__setattr__(self, "locale", _require_non_empty(self.locale, "locale"))


@dataclass(frozen=True, slots=True)
class OptimizationExecutionGuard:
    """Recursion and execution-scope guard contract."""

    execution_scope: ModelCallExecutionScope
    operation_id: str
    parent_operation_id: str | None
    optimization_depth: int
    active_artifact_lookup_key_hashes: tuple[str, ...] = ()
    active_strategy_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "execution_scope",
            _require_enum(
                self.execution_scope,
                ModelCallExecutionScope,
                "execution_scope",
            ),
        )
        object.__setattr__(self, "operation_id", _require_non_empty(self.operation_id, "operation_id"))
        depth = _require_non_negative(self.optimization_depth, "optimization_depth")

        if depth > 1:
            raise ValueError(ContextOptimizationReasonCode.OPTIMIZATION_DEPTH_EXCEEDED.value)

        if self.execution_scope is ModelCallExecutionScope.PRIMARY_MODEL_CALL:
            if depth != 0:
                raise ValueError("PRIMARY_MODEL_CALL requires optimization_depth == 0")
            if self.parent_operation_id is not None:
                raise ValueError("PRIMARY_MODEL_CALL requires parent_operation_id is None")
        elif self.execution_scope is ModelCallExecutionScope.INTERNAL_OPTIMIZATION_CALL:
            if depth != 1:
                raise ValueError("INTERNAL_OPTIMIZATION_CALL requires optimization_depth == 1")
            if self.parent_operation_id is None:
                raise ValueError("INTERNAL_OPTIMIZATION_CALL requires parent_operation_id")
            else:
                object.__setattr__(
                    self,
                    "parent_operation_id",
                    _require_non_empty(self.parent_operation_id, "parent_operation_id"),
                )

        key_hashes = tuple(self.active_artifact_lookup_key_hashes)
        if any(not key_hash for key_hash in key_hashes):
            raise ValueError("active_artifact_lookup_key_hashes must not contain empty values")
        object.__setattr__(
            self,
            "active_artifact_lookup_key_hashes",
            _reject_duplicates(key_hashes, "active_artifact_lookup_key_hashes"),
        )

        strategy_ids = tuple(self.active_strategy_ids)
        if any(not strategy_id for strategy_id in strategy_ids):
            raise ValueError("active_strategy_ids must not contain empty values")
        object.__setattr__(
            self,
            "active_strategy_ids",
            _reject_duplicates(strategy_ids, "active_strategy_ids"),
        )


@dataclass(frozen=True, slots=True)
class ArtifactCreationReservation:
    """Single-flight artifact creation reservation contract."""

    reservation_id: str
    artifact_lookup_key_hash: str
    tenant_id: str
    workspace_id: str
    owner_operation_id: str
    acquired_at: datetime
    lease_deadline: datetime

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reservation_id",
            _require_non_empty(self.reservation_id, "reservation_id"),
        )
        object.__setattr__(
            self,
            "artifact_lookup_key_hash",
            _require_non_empty(self.artifact_lookup_key_hash, "artifact_lookup_key_hash"),
        )
        object.__setattr__(self, "tenant_id", _require_non_empty(self.tenant_id, "tenant_id"))
        object.__setattr__(
            self,
            "workspace_id",
            _require_strict_non_empty_text(self.workspace_id, "workspace_id"),
        )
        object.__setattr__(
            self,
            "owner_operation_id",
            _require_non_empty(self.owner_operation_id, "owner_operation_id"),
        )
        acquired = _require_timezone_aware(self.acquired_at, "acquired_at")
        deadline = _require_timezone_aware(self.lease_deadline, "lease_deadline")
        if deadline <= acquired:
            raise ValueError("lease_deadline must be > acquired_at")
        object.__setattr__(self, "acquired_at", acquired)
        object.__setattr__(self, "lease_deadline", deadline)


@dataclass(frozen=True, slots=True)
class ArtifactValidationSummary:
    """Validation summary for reusable optimization artifacts."""

    status: ArtifactValidationStatus
    validation_contract_version: str
    validated_at: datetime
    reason_codes: tuple[str, ...] = ()
    safe_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "status",
            _require_enum(self.status, ArtifactValidationStatus, "status"),
        )
        object.__setattr__(
            self,
            "validation_contract_version",
            _require_non_empty(self.validation_contract_version, "validation_contract_version"),
        )
        object.__setattr__(
            self,
            "validated_at",
            _require_timezone_aware(self.validated_at, "validated_at"),
        )
        codes = tuple(self.reason_codes)
        if any(not code for code in codes):
            raise ValueError("reason_codes must not contain empty values")
        object.__setattr__(self, "reason_codes", _reject_duplicates(codes, "reason_codes"))
        object.__setattr__(self, "safe_metadata", _normalize_safe_metadata(self.safe_metadata))


@dataclass(frozen=True, slots=True)
class ArtifactCompatibilityResult:
    """Compatibility evaluation result contract."""

    status: ArtifactCompatibilityStatus
    artifact_id: str
    requested_lookup_key_hash: str
    artifact_lookup_key_hash: str
    reasons: tuple[ArtifactCompatibilityReason, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "status",
            _require_enum(self.status, ArtifactCompatibilityStatus, "status"),
        )
        object.__setattr__(self, "artifact_id", _require_non_empty(self.artifact_id, "artifact_id"))
        object.__setattr__(
            self,
            "requested_lookup_key_hash",
            _require_non_empty(self.requested_lookup_key_hash, "requested_lookup_key_hash"),
        )
        object.__setattr__(
            self,
            "artifact_lookup_key_hash",
            _require_non_empty(self.artifact_lookup_key_hash, "artifact_lookup_key_hash"),
        )
        reasons = tuple(
            _require_enum(item, ArtifactCompatibilityReason, "reasons item")
            for item in self.reasons
        )
        if self.status is ArtifactCompatibilityStatus.COMPATIBLE:
            if reasons:
                raise ValueError("COMPATIBLE requires empty reasons")
            if self.requested_lookup_key_hash != self.artifact_lookup_key_hash:
                raise ValueError("COMPATIBLE requires matching lookup key hashes")
        elif self.status is ArtifactCompatibilityStatus.INCOMPATIBLE:
            if not reasons:
                raise ValueError("INCOMPATIBLE requires non-empty reasons")
        object.__setattr__(self, "reasons", reasons)


@dataclass(frozen=True, slots=True)
class ReusableOptimizationArtifact:
    """Metadata-only reusable optimization artifact record."""

    artifact_id: str
    lookup_key: ArtifactLookupKey
    ownership: UclArtifactOwnership
    artifact_content_hash: str
    created_at: datetime
    created_by_executor: str
    validation: ArtifactValidationSummary
    status: ReusableArtifactStatus = ReusableArtifactStatus.VALIDATED
    invalidation_reason: str | None = None
    supersedes_artifact_id: str | None = None
    receipt_ref: str | None = None
    safe_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", _require_non_empty(self.artifact_id, "artifact_id"))
        object.__setattr__(
            self,
            "lookup_key",
            _require_instance(self.lookup_key, ArtifactLookupKey, "lookup_key"),
        )
        object.__setattr__(
            self,
            "ownership",
            _require_instance(self.ownership, UclArtifactOwnership, "ownership"),
        )
        if self.ownership.kind is UclArtifactOwnershipKind.WORKSPACE:
            scope = self.ownership.scope
            if scope is None or scope.tenant_id != self.lookup_key.tenant_id:
                raise ValueError("ownership.scope.tenant_id must equal lookup_key.tenant_id")
        object.__setattr__(
            self,
            "validation",
            _require_instance(self.validation, ArtifactValidationSummary, "validation"),
        )
        object.__setattr__(
            self,
            "status",
            _require_enum(self.status, ReusableArtifactStatus, "status"),
        )
        object.__setattr__(
            self,
            "artifact_content_hash",
            _require_non_empty(self.artifact_content_hash, "artifact_content_hash"),
        )
        object.__setattr__(
            self,
            "created_by_executor",
            _require_non_empty(self.created_by_executor, "created_by_executor"),
        )
        object.__setattr__(
            self,
            "created_at",
            _require_timezone_aware(self.created_at, "created_at"),
        )

        if self.validation.validation_contract_version != self.lookup_key.validation_contract_version:
            raise ValueError(
                "validation.validation_contract_version must equal "
                "lookup_key.validation_contract_version"
            )

        if self.status is ReusableArtifactStatus.VALIDATED:
            if self.validation.status is not ArtifactValidationStatus.PASSED:
                raise ValueError("VALIDATED requires validation.status == PASSED")
            if self.invalidation_reason is not None:
                raise ValueError("VALIDATED requires invalidation_reason is None")
        elif self.status is ReusableArtifactStatus.INVALIDATED:
            if self.invalidation_reason is None:
                raise ValueError("INVALIDATED requires invalidation_reason")
        elif self.status is ReusableArtifactStatus.RETIRED:
            if self.invalidation_reason is None:
                raise ValueError("RETIRED requires invalidation_reason")

        if self.invalidation_reason is not None:
            object.__setattr__(
                self,
                "invalidation_reason",
                _require_non_empty(self.invalidation_reason, "invalidation_reason"),
            )

        if self.supersedes_artifact_id is not None:
            superseded = _require_non_empty(self.supersedes_artifact_id, "supersedes_artifact_id")
            if superseded == self.artifact_id:
                raise ValueError("supersedes_artifact_id cannot equal artifact_id")
            object.__setattr__(self, "supersedes_artifact_id", superseded)

        if self.receipt_ref is not None:
            object.__setattr__(
                self,
                "receipt_ref",
                _require_non_empty(self.receipt_ref, "receipt_ref"),
            )

        object.__setattr__(self, "safe_metadata", _normalize_safe_metadata(self.safe_metadata))

    @property
    def workspace_id(self) -> str | None:
        """Canonical workspace owner when persisted; None for legacy-unknown ownership."""
        if self.ownership.kind is not UclArtifactOwnershipKind.WORKSPACE:
            return None
        scope = self.ownership.scope
        return scope.workspace_id if scope is not None else None
