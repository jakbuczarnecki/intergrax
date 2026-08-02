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


def _require_non_empty(value: str, field_name: str) -> str:
    if not value:
        raise ValueError(f"{field_name} must be non-empty")
    return value


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


class ContextOptimizationMode(StrEnum):
    """Context optimization operating mode."""

    EPHEMERAL_ASSEMBLY = "ephemeral_assembly"
    DURABLE_COMPACTION = "durable_compaction"


class ContextOptimizationDecision(StrEnum):
    """Canonical UCL optimization decision outcome."""

    NO_OP = "no_op"
    SELECT_ONLY = "select_only"
    REUSE_ARTIFACT = "reuse_artifact"
    CREATE_ARTIFACT = "create_artifact"
    POLICY_BLOCKED = "policy_blocked"
    FAIL_CLOSED = "fail_closed"


class OptimizationArtifactType(StrEnum):
    """Typed optimization artifact classification."""

    TEXT = "text"
    MESSAGE_SEQUENCE = "message_sequence"
    FRAGMENT_SET = "fragment_set"
    TOOL_CATALOG = "tool_catalog"
    STRUCTURED_DATA = "structured_data"


class ArtifactCreationCoordinationStatus(StrEnum):
    """Reservation/concurrency coordination status."""

    ARTIFACT_AVAILABLE = "artifact_available"
    ACQUIRED = "acquired"
    ALREADY_IN_PROGRESS = "already_in_progress"
    RESERVATION_EXPIRED = "reservation_expired"
    RESERVATION_CONFLICT = "reservation_conflict"


class EphemeralArtifactPersistencePolicy(StrEnum):
    """Persistence policy for ephemeral assembly artifacts."""

    DO_NOT_PERSIST = "do_not_persist_ephemeral_artifact"
    PERSIST_REUSABLE = "persist_reusable_artifact"
    PERSIST_AFTER_VALIDATION = "persist_only_after_validation"
    PERSIST_AFTER_HUMAN_REVIEW = "persist_only_after_human_review"


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
class ContextOptimizationPolicy:
    """Normalized context optimization policy contract."""

    policy_version: str
    validation_contract_version: str
    enabled: bool = False
    mode: ContextOptimizationMode = ContextOptimizationMode.EPHEMERAL_ASSEMBLY
    allow_lossy: bool = False
    allow_llm_summarization: bool = False
    allow_artifact_reuse: bool = True
    allow_administrative_refresh: bool = False
    allowed_artifact_types: tuple[OptimizationArtifactType, ...] = ()
    allowed_strategy_ids: tuple[str, ...] = ()
    require_receipt: bool = True
    require_rollback_metadata: bool = False
    require_human_review: bool = False
    ephemeral_artifact_persistence: EphemeralArtifactPersistencePolicy = (
        EphemeralArtifactPersistencePolicy.DO_NOT_PERSIST
    )
    recent_tail_min_messages: int = 0
    protected_region_policy_version: str | None = None
    minimum_quality_score: float | None = None
    reservation_lease_seconds: int = 60
    cache_policy_ref: str | None = None
    retention_policy_ref: str | None = None
    safe_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
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
            "mode",
            _require_enum(self.mode, ContextOptimizationMode, "mode"),
        )
        object.__setattr__(
            self,
            "ephemeral_artifact_persistence",
            _require_enum(
                self.ephemeral_artifact_persistence,
                EphemeralArtifactPersistencePolicy,
                "ephemeral_artifact_persistence",
            ),
        )
        object.__setattr__(
            self,
            "recent_tail_min_messages",
            _require_non_negative(self.recent_tail_min_messages, "recent_tail_min_messages"),
        )
        object.__setattr__(
            self,
            "reservation_lease_seconds",
            _require_positive(self.reservation_lease_seconds, "reservation_lease_seconds"),
        )
        object.__setattr__(
            self,
            "minimum_quality_score",
            _require_finite_quality_score(self.minimum_quality_score),
        )

        if self.allow_llm_summarization and not self.allow_lossy:
            raise ValueError("allow_llm_summarization requires allow_lossy")

        if self.mode is ContextOptimizationMode.DURABLE_COMPACTION:
            if not self.require_receipt:
                raise ValueError("DURABLE_COMPACTION requires require_receipt")
            if not self.require_rollback_metadata:
                raise ValueError("DURABLE_COMPACTION requires require_rollback_metadata")

        if (
            self.ephemeral_artifact_persistence
            is EphemeralArtifactPersistencePolicy.PERSIST_AFTER_HUMAN_REVIEW
            and not self.require_human_review
        ):
            raise ValueError("PERSIST_AFTER_HUMAN_REVIEW requires require_human_review")

        if (
            self.ephemeral_artifact_persistence
            is not EphemeralArtifactPersistencePolicy.DO_NOT_PERSIST
            and not self.allow_artifact_reuse
        ):
            raise ValueError("artifact persistence requires allow_artifact_reuse")

        artifact_types = tuple(
            _require_enum(item, OptimizationArtifactType, "allowed_artifact_types item")
            for item in self.allowed_artifact_types
        )
        if len(artifact_types) != len(set(artifact_types)):
            raise ValueError("allowed_artifact_types must not contain duplicates")
        object.__setattr__(self, "allowed_artifact_types", artifact_types)

        strategy_ids = tuple(self.allowed_strategy_ids)
        if any(not strategy_id for strategy_id in strategy_ids):
            raise ValueError("allowed_strategy_ids must not contain empty values")
        object.__setattr__(
            self,
            "allowed_strategy_ids",
            _reject_duplicates(strategy_ids, "allowed_strategy_ids"),
        )

        if self.protected_region_policy_version is not None:
            object.__setattr__(
                self,
                "protected_region_policy_version",
                _require_non_empty(
                    self.protected_region_policy_version,
                    "protected_region_policy_version",
                ),
            )
        if self.cache_policy_ref is not None:
            object.__setattr__(
                self,
                "cache_policy_ref",
                _require_non_empty(self.cache_policy_ref, "cache_policy_ref"),
            )
        if self.retention_policy_ref is not None:
            object.__setattr__(
                self,
                "retention_policy_ref",
                _require_non_empty(self.retention_policy_ref, "retention_policy_ref"),
            )

        object.__setattr__(self, "safe_metadata", _normalize_safe_metadata(self.safe_metadata))


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
