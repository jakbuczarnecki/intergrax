# © Artur Czarnecki. All rights reserved.

"""Context optimization policy — public declarative contract (CTX-UCL)."""

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


class DurableCompactionActivationMode(StrEnum):
    """Frozen activation modes for durable compaction policy."""

    COMPARE_AND_SWAP = "compare_and_swap"
    MANUAL_REVIEW_THEN_COMPARE_AND_SWAP = "manual_review_then_compare_and_swap"


_SUPPORTED_DURABLE_COMPACTION_ACTIVATION_MODES: frozenset[DurableCompactionActivationMode] = (
    frozenset(DurableCompactionActivationMode)
)


class DurableCompactionValidationRequirement(StrEnum):
    """Minimum validation stages required before durable activation."""

    STRUCTURAL = "structural"
    STRUCTURAL_AND_PROTECTED = "structural_and_protected"
    FULL = "full"


class DurableCompactionEligibilityReasonCode(StrEnum):
    """Stable fail-closed reason codes for durable compaction eligibility."""

    POLICY_DISABLED = "policy_disabled"
    DURABLE_COMPACTION_DISABLED = "durable_compaction_disabled"
    WRONG_OPTIMIZATION_MODE = "wrong_optimization_mode"
    MISSING_SOURCE_REVISION = "missing_source_revision"
    MISSING_EXPECTED_ACTIVE_REVISION = "missing_expected_active_revision"
    INVALID_SOURCE_IDENTITY = "invalid_source_identity"
    STRATEGY_NOT_ALLOWED = "strategy_not_allowed"
    ARTIFACT_TYPE_NOT_ALLOWED = "artifact_type_not_allowed"
    LOSSINESS_NOT_ALLOWED = "lossiness_not_allowed"
    STABILITY_REQUIREMENT_UNAVAILABLE = "stability_requirement_unavailable"
    STABILITY_REQUIREMENT_NOT_MET = "stability_requirement_not_met"
    VALIDATION_REQUIREMENT_UNAVAILABLE = "validation_requirement_unavailable"
    LINEAGE_REQUIREMENT_UNAVAILABLE = "lineage_requirement_unavailable"
    RECEIPT_REQUIREMENT_UNAVAILABLE = "receipt_requirement_unavailable"
    ROLLBACK_REQUIREMENT_UNAVAILABLE = "rollback_requirement_unavailable"
    RAW_CONTENT_FORBIDDEN = "raw_content_forbidden"
    POLICY_TARGET_IDENTITY_MISMATCH = "policy_target_identity_mismatch"


_SUPPORTED_DURABLE_LOSSINESS_PROFILES: frozenset[str] = frozenset({"lossy_summary"})
_DURABLE_LLM_SUMMARY_STRATEGY_IDS: frozenset[str] = frozenset(
    {"message_sequence_summarization.v1"}
)


def _require_sha256_hex(value: str, field_name: str) -> str:
    digest = _require_non_empty_text(value, field_name)
    if len(digest) != 64:
        raise ValueError(f"{field_name} must be a 64-character lowercase hex SHA-256 digest")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a valid hex SHA-256 digest") from exc
    if digest != digest.lower():
        raise ValueError(f"{field_name} must be lowercase hex")
    return digest


def _require_non_empty_source_refs(refs: object, field_name: str) -> tuple[str, ...]:
    if type(refs) not in (list, tuple):
        raise ValueError(f"{field_name} must be a sequence of strings")
    if not refs:
        raise ValueError(f"{field_name} must not be empty")
    normalized: list[str] = []
    for ref in refs:
        normalized.append(_require_strict_non_empty_text(ref, field_name))
    return _reject_duplicates(tuple(normalized), field_name)


@dataclass(frozen=True, slots=True)
class DurableCompactionPolicy:
    """Nested durable compaction policy governed by ContextOptimizationPolicy."""

    enabled: bool = False
    activation_mode: DurableCompactionActivationMode = (
        DurableCompactionActivationMode.COMPARE_AND_SWAP
    )
    minimum_validation_requirement: DurableCompactionValidationRequirement = (
        DurableCompactionValidationRequirement.FULL
    )
    allowed_strategy_ids: tuple[str, ...] = ()
    allowed_lossiness_profiles: tuple[str, ...] = ()
    minimum_stable_revision_count: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", _require_bool(self.enabled, "enabled"))
        object.__setattr__(
            self,
            "activation_mode",
            _require_enum(
                self.activation_mode,
                DurableCompactionActivationMode,
                "activation_mode",
            ),
        )
        if self.activation_mode not in _SUPPORTED_DURABLE_COMPACTION_ACTIVATION_MODES:
            raise ValueError("activation_mode is not supported")
        object.__setattr__(
            self,
            "minimum_validation_requirement",
            _require_enum(
                self.minimum_validation_requirement,
                DurableCompactionValidationRequirement,
                "minimum_validation_requirement",
            ),
        )
        stable_count = _require_positive(
            self.minimum_stable_revision_count,
            "minimum_stable_revision_count",
        )
        object.__setattr__(self, "minimum_stable_revision_count", stable_count)

        strategy_ids = _require_strict_non_empty_texts(
            self.allowed_strategy_ids,
            "allowed_strategy_ids",
        )
        object.__setattr__(
            self,
            "allowed_strategy_ids",
            strategy_ids,
        )

        lossiness_profiles = _require_strict_non_empty_texts(
            self.allowed_lossiness_profiles,
            "allowed_lossiness_profiles",
        )
        unsupported_profiles = set(lossiness_profiles) - _SUPPORTED_DURABLE_LOSSINESS_PROFILES
        if unsupported_profiles:
            raise ValueError(
                "allowed_lossiness_profiles contains unsupported profile(s)"
            )
        object.__setattr__(
            self,
            "allowed_lossiness_profiles",
            lossiness_profiles,
        )

        if self.enabled:
            if not self.allowed_strategy_ids:
                raise ValueError("enabled durable compaction requires allowed_strategy_ids")
            if not self.allowed_lossiness_profiles:
                raise ValueError(
                    "enabled durable compaction requires allowed_lossiness_profiles"
                )

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
    durable_compaction: DurableCompactionPolicy | None = None

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

        strategy_ids = _require_non_empty_texts(
            self.allowed_strategy_ids,
            "allowed_strategy_ids",
        )
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

        if self.durable_compaction is not None:
            object.__setattr__(
                self,
                "durable_compaction",
                _require_instance(
                    self.durable_compaction,
                    DurableCompactionPolicy,
                    "durable_compaction",
                ),
            )
            if (
                self.durable_compaction.enabled
                and self.mode is not ContextOptimizationMode.DURABLE_COMPACTION
            ):
                raise ValueError(
                    "enabled durable_compaction requires DURABLE_COMPACTION mode"
                )

