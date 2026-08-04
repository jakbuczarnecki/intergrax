# © Artur Czarnecki. All rights reserved.

"""Safe and canonical serialization for Unified Context Lifecycle contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from intergrax.runtime.context_lifecycle.contracts import (
    ArtifactCompatibilityResult,
    ArtifactCompressionTarget,
    ArtifactCreationReservation,
    ArtifactLookupKey,
    ArtifactSourceRange,
    ContextOptimizationPolicy,
    DurableCompactionActivationMode,
    DurableCompactionActivationRequirements,
    DurableCompactionPolicy,
    DurableCompactionSourceIdentity,
    DurableCompactionValidationRequirement,
    OptimizationArtifactType,
    OptimizationExecutionGuard,
    ReusableOptimizationArtifact,
)
from intergrax.runtime.context_lifecycle.repository import (
    ArtifactCreationCoordinationResult,
    OptimizationArtifactReference,
    OptimizationArtifactRepositoryCapabilities,
    StoredOptimizationArtifact,
)

_RAW_CONTENT_INCLUDED = False


def _compression_target_to_dict(target: ArtifactCompressionTarget) -> dict[str, Any]:
    if target.target_tokens is not None:
        return {"target_tokens": target.target_tokens}
    return {"budget_class": target.budget_class}


def _source_range_to_dict(source_range: ArtifactSourceRange) -> dict[str, int]:
    return {
        "end_sequence": source_range.end_sequence,
        "start_sequence": source_range.start_sequence,
    }


def artifact_lookup_key_to_canonical_dict(key: ArtifactLookupKey) -> dict[str, Any]:
    """Return the canonical identity dictionary for deterministic hashing."""
    payload: dict[str, Any] = {
        "artifact_type": key.artifact_type.value,
        "compression_target": _compression_target_to_dict(key.compression_target),
        "context_scope_id": key.context_scope_id,
        "lossiness_profile": key.lossiness_profile,
        "policy_version": key.policy_version,
        "source_content_hash": key.source_content_hash,
        "strategy_id": key.strategy_id,
        "strategy_version": key.strategy_version,
        "tenant_id": key.tenant_id,
        "validation_contract_version": key.validation_contract_version,
    }
    if key.source_refs:
        payload["source_refs"] = list(key.source_refs)
    if key.source_range is not None:
        payload["source_range"] = _source_range_to_dict(key.source_range)
    if key.protected_region_policy_version is not None:
        payload["protected_region_policy_version"] = key.protected_region_policy_version
    if key.model_family is not None:
        payload["model_family"] = key.model_family
    if key.locale is not None:
        payload["locale"] = key.locale
    return payload


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_artifact_lookup_key_hash(key: ArtifactLookupKey) -> str:
    """Return a deterministic SHA-256 digest for the lookup key identity."""
    canonical = _canonical_json(artifact_lookup_key_to_canonical_dict(key))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _artifact_lookup_key_to_safe_dict(
    key: ArtifactLookupKey,
    *,
    artifact_lookup_key_hash: str,
) -> dict[str, Any]:
    return {
        "artifact_lookup_key_hash": artifact_lookup_key_hash,
        "artifact_type": key.artifact_type.value,
        "compression_target": _compression_target_to_dict(key.compression_target),
        "lossiness_profile": key.lossiness_profile,
        "policy_version": key.policy_version,
        "raw_content_included": _RAW_CONTENT_INCLUDED,
        "strategy_id": key.strategy_id,
        "strategy_version": key.strategy_version,
        "validation_contract_version": key.validation_contract_version,
    }


def _datetime_to_iso(value: datetime) -> str:
    return value.isoformat()


def _to_json_safe_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _to_json_safe_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_json_safe_value(item) for item in value]
    return value


def _reject_unknown_keys(payload: Mapping[str, Any], allowed: frozenset[str]) -> None:
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"unknown fields: {sorted(unknown)}")


def durable_compaction_policy_to_canonical_dict(
    policy: DurableCompactionPolicy,
) -> dict[str, Any]:
    """Return canonical durable compaction policy identity dictionary."""
    return {
        "activation_mode": policy.activation_mode.value,
        "allowed_lossiness_profiles": list(policy.allowed_lossiness_profiles),
        "allowed_strategy_ids": list(policy.allowed_strategy_ids),
        "enabled": policy.enabled,
        "minimum_stable_revision_count": policy.minimum_stable_revision_count,
        "minimum_validation_requirement": policy.minimum_validation_requirement.value,
    }


def compute_durable_compaction_policy_hash(policy: DurableCompactionPolicy) -> str:
    """Return deterministic SHA-256 digest for durable compaction policy identity."""
    canonical = _canonical_json(durable_compaction_policy_to_canonical_dict(policy))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def durable_compaction_source_identity_to_canonical_dict(
    identity: DurableCompactionSourceIdentity,
) -> dict[str, Any]:
    """Return canonical durable compaction source identity dictionary."""
    return {
        "artifact_lookup_key": artifact_lookup_key_to_canonical_dict(identity.artifact_lookup_key),
        "context_scope_id": identity.context_scope_id,
        "expected_active_revision": identity.expected_active_revision,
        "lossiness_profile": identity.lossiness_profile,
        "source_content_hash": identity.source_content_hash,
        "source_refs": list(identity.source_refs),
        "source_revision": identity.source_revision,
        "strategy_id": identity.strategy_id,
        "strategy_version": identity.strategy_version,
        "tenant_id": identity.tenant_id,
    }


def compute_durable_compaction_source_identity_hash(
    identity: DurableCompactionSourceIdentity,
) -> str:
    """Return deterministic SHA-256 digest for durable compaction source identity."""
    canonical = _canonical_json(durable_compaction_source_identity_to_canonical_dict(identity))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def durable_compaction_activation_requirements_to_canonical_dict(
    requirements: DurableCompactionActivationRequirements,
) -> dict[str, Any]:
    """Return canonical durable compaction activation requirements dictionary."""
    return {
        "candidate_artifact_id": requirements.candidate_artifact_id,
        "creation_receipt_reference": requirements.creation_receipt_reference,
        "expected_active_revision": requirements.expected_active_revision,
        "lineage_reference": requirements.lineage_reference,
        "raw_content_included": False,
        "rollback_source_reference": requirements.rollback_source_reference,
        "validated_artifact_id": requirements.validated_artifact_id,
    }


def durable_compaction_policy_from_canonical_dict(
    payload: Mapping[str, Any],
) -> DurableCompactionPolicy:
    """Decode durable compaction policy from canonical dictionary."""
    _reject_unknown_keys(
        payload,
        frozenset(
            {
                "activation_mode",
                "allowed_lossiness_profiles",
                "allowed_strategy_ids",
                "enabled",
                "minimum_stable_revision_count",
                "minimum_validation_requirement",
            }
        ),
    )
    try:
        activation_mode = DurableCompactionActivationMode(payload["activation_mode"])
    except (KeyError, ValueError) as exc:
        raise ValueError("activation_mode must be a supported enum value") from exc
    try:
        minimum_validation_requirement = DurableCompactionValidationRequirement(
            payload["minimum_validation_requirement"]
        )
    except (KeyError, ValueError) as exc:
        raise ValueError(
            "minimum_validation_requirement must be a supported enum value"
        ) from exc
    enabled = payload["enabled"]
    if not isinstance(enabled, bool):
        raise ValueError("enabled must be a boolean")
    return DurableCompactionPolicy(
        enabled=enabled,
        activation_mode=activation_mode,
        minimum_validation_requirement=minimum_validation_requirement,
        allowed_strategy_ids=tuple(payload["allowed_strategy_ids"]),
        allowed_lossiness_profiles=tuple(payload["allowed_lossiness_profiles"]),
        minimum_stable_revision_count=payload["minimum_stable_revision_count"],
    )


def durable_compaction_source_identity_from_canonical_dict(
    payload: Mapping[str, Any],
) -> DurableCompactionSourceIdentity:
    """Decode durable compaction source identity from canonical dictionary."""
    _reject_unknown_keys(
        payload,
        frozenset(
            {
                "artifact_lookup_key",
                "context_scope_id",
                "expected_active_revision",
                "lossiness_profile",
                "source_content_hash",
                "source_refs",
                "source_revision",
                "strategy_id",
                "strategy_version",
                "tenant_id",
            }
        ),
    )
    lookup_payload = payload["artifact_lookup_key"]
    if not isinstance(lookup_payload, Mapping):
        raise ValueError("artifact_lookup_key must be a mapping")
    _reject_unknown_keys(
        lookup_payload,
        frozenset(
            {
                "artifact_type",
                "compression_target",
                "context_scope_id",
                "lossiness_profile",
                "policy_version",
                "source_content_hash",
                "source_refs",
                "strategy_id",
                "strategy_version",
                "tenant_id",
                "validation_contract_version",
            }
        ),
    )
    compression_target_payload = lookup_payload["compression_target"]
    if not isinstance(compression_target_payload, Mapping):
        raise ValueError("compression_target must be a mapping")
    _reject_unknown_keys(compression_target_payload, frozenset({"budget_class", "target_tokens"}))
    if ("target_tokens" in compression_target_payload) == (
        "budget_class" in compression_target_payload
    ):
        raise ValueError(
            "compression_target must contain exactly one of target_tokens or budget_class"
        )
    if "target_tokens" in compression_target_payload:
        compression_target = ArtifactCompressionTarget(
            target_tokens=compression_target_payload["target_tokens"]
        )
    else:
        compression_target = ArtifactCompressionTarget(
            budget_class=compression_target_payload["budget_class"]
        )
    lookup_key = ArtifactLookupKey(
        tenant_id=lookup_payload["tenant_id"],
        context_scope_id=lookup_payload["context_scope_id"],
        artifact_type=OptimizationArtifactType(lookup_payload["artifact_type"]),
        source_content_hash=lookup_payload["source_content_hash"],
        strategy_id=lookup_payload["strategy_id"],
        strategy_version=lookup_payload["strategy_version"],
        policy_version=lookup_payload["policy_version"],
        validation_contract_version=lookup_payload["validation_contract_version"],
        compression_target=compression_target,
        lossiness_profile=lookup_payload["lossiness_profile"],
        source_refs=tuple(lookup_payload["source_refs"]),
    )
    return DurableCompactionSourceIdentity(
        tenant_id=payload["tenant_id"],
        context_scope_id=payload["context_scope_id"],
        source_revision=payload["source_revision"],
        expected_active_revision=payload["expected_active_revision"],
        source_refs=tuple(payload["source_refs"]),
        source_content_hash=payload["source_content_hash"],
        artifact_lookup_key=lookup_key,
        strategy_id=payload["strategy_id"],
        strategy_version=payload["strategy_version"],
        lossiness_profile=payload["lossiness_profile"],
    )


def _durable_compaction_policy_to_safe_dict(
    policy: DurableCompactionPolicy,
) -> dict[str, Any]:
    return {
        **durable_compaction_policy_to_canonical_dict(policy),
        "raw_content_included": _RAW_CONTENT_INCLUDED,
    }


def context_optimization_policy_to_safe_dict(policy: ContextOptimizationPolicy) -> dict[str, Any]:
    """Return telemetry-safe policy serialization."""
    payload: dict[str, Any] = {
        "allow_administrative_refresh": policy.allow_administrative_refresh,
        "allow_artifact_reuse": policy.allow_artifact_reuse,
        "allow_llm_summarization": policy.allow_llm_summarization,
        "allow_lossy": policy.allow_lossy,
        "allowed_artifact_types": [item.value for item in policy.allowed_artifact_types],
        "allowed_strategy_ids": list(policy.allowed_strategy_ids),
        "cache_policy_ref": policy.cache_policy_ref,
        "enabled": policy.enabled,
        "ephemeral_artifact_persistence": policy.ephemeral_artifact_persistence.value,
        "minimum_quality_score": policy.minimum_quality_score,
        "mode": policy.mode.value,
        "policy_version": policy.policy_version,
        "protected_region_policy_version": policy.protected_region_policy_version,
        "raw_content_included": _RAW_CONTENT_INCLUDED,
        "recent_tail_min_messages": policy.recent_tail_min_messages,
        "require_human_review": policy.require_human_review,
        "require_receipt": policy.require_receipt,
        "require_rollback_metadata": policy.require_rollback_metadata,
        "reservation_lease_seconds": policy.reservation_lease_seconds,
        "retention_policy_ref": policy.retention_policy_ref,
        "safe_metadata": _to_json_safe_value(policy.safe_metadata),
        "validation_contract_version": policy.validation_contract_version,
    }
    if policy.durable_compaction is not None:
        payload["durable_compaction"] = _durable_compaction_policy_to_safe_dict(
            policy.durable_compaction
        )
    return payload


def optimization_execution_guard_to_safe_dict(guard: OptimizationExecutionGuard) -> dict[str, Any]:
    """Return telemetry-safe execution guard serialization."""
    return {
        "active_artifact_lookup_key_hashes": list(guard.active_artifact_lookup_key_hashes),
        "active_strategy_ids": list(guard.active_strategy_ids),
        "execution_scope": guard.execution_scope.value,
        "operation_id": guard.operation_id,
        "optimization_depth": guard.optimization_depth,
        "parent_operation_id": guard.parent_operation_id,
        "raw_content_included": _RAW_CONTENT_INCLUDED,
    }


def artifact_creation_reservation_to_safe_dict(
    reservation: ArtifactCreationReservation,
) -> dict[str, Any]:
    """Return telemetry-safe reservation serialization."""
    return {
        "acquired_at": _datetime_to_iso(reservation.acquired_at),
        "artifact_lookup_key_hash": reservation.artifact_lookup_key_hash,
        "lease_deadline": _datetime_to_iso(reservation.lease_deadline),
        "owner_operation_id": reservation.owner_operation_id,
        "raw_content_included": _RAW_CONTENT_INCLUDED,
        "reservation_id": reservation.reservation_id,
    }


def reusable_optimization_artifact_to_safe_dict(
    artifact: ReusableOptimizationArtifact,
) -> dict[str, Any]:
    """Return telemetry-safe reusable artifact serialization."""
    lookup_key_hash = compute_artifact_lookup_key_hash(artifact.lookup_key)
    return {
        "artifact_content_hash": artifact.artifact_content_hash,
        "artifact_id": artifact.artifact_id,
        "created_at": _datetime_to_iso(artifact.created_at),
        "created_by_executor": artifact.created_by_executor,
        "invalidation_reason": artifact.invalidation_reason,
        "lookup_key": _artifact_lookup_key_to_safe_dict(
            artifact.lookup_key,
            artifact_lookup_key_hash=lookup_key_hash,
        ),
        "raw_content_included": _RAW_CONTENT_INCLUDED,
        "receipt_ref": artifact.receipt_ref,
        "safe_metadata": _to_json_safe_value(artifact.safe_metadata),
        "status": artifact.status.value,
        "supersedes_artifact_id": artifact.supersedes_artifact_id,
        "validation": {
            "reason_codes": list(artifact.validation.reason_codes),
            "safe_metadata": _to_json_safe_value(artifact.validation.safe_metadata),
            "status": artifact.validation.status.value,
            "validated_at": _datetime_to_iso(artifact.validation.validated_at),
            "validation_contract_version": artifact.validation.validation_contract_version,
        },
    }


def artifact_compatibility_result_to_safe_dict(
    result: ArtifactCompatibilityResult,
) -> dict[str, Any]:
    """Return telemetry-safe compatibility result serialization."""
    return {
        "artifact_id": result.artifact_id,
        "artifact_lookup_key_hash": result.artifact_lookup_key_hash,
        "raw_content_included": _RAW_CONTENT_INCLUDED,
        "reasons": [reason.value for reason in result.reasons],
        "requested_lookup_key_hash": result.requested_lookup_key_hash,
        "status": result.status.value,
    }


def optimization_artifact_repository_capabilities_to_safe_dict(
    capabilities: OptimizationArtifactRepositoryCapabilities,
) -> dict[str, Any]:
    """Return telemetry-safe repository capability serialization."""
    return {
        "backend_id": capabilities.backend_id,
        "durable": capabilities.durable,
        "raw_content_included": _RAW_CONTENT_INCLUDED,
        "reference_only": capabilities.reference_only,
        "shared_across_processes": capabilities.shared_across_processes,
        "supports_bounded_wait": capabilities.supports_bounded_wait,
        "supports_single_flight": capabilities.supports_single_flight,
    }


def optimization_artifact_reference_to_safe_dict(
    reference: OptimizationArtifactReference,
) -> dict[str, Any]:
    """Return telemetry-safe artifact reference serialization."""
    return {
        "artifact_content_hash": reference.artifact_content_hash,
        "artifact_id": reference.artifact_id,
        "artifact_lookup_key_hash": reference.artifact_lookup_key_hash,
        "artifact_type": reference.artifact_type.value,
        "raw_content_included": _RAW_CONTENT_INCLUDED,
    }


def stored_optimization_artifact_to_safe_dict(
    artifact: StoredOptimizationArtifact,
) -> dict[str, Any]:
    """Return telemetry-safe stored artifact serialization without payload."""
    metadata_safe = reusable_optimization_artifact_to_safe_dict(artifact.metadata)
    return {
        **metadata_safe,
        "encoding": artifact.encoding,
        "media_type": artifact.media_type,
        "payload_size_bytes": len(artifact.payload),
        "raw_content_included": _RAW_CONTENT_INCLUDED,
    }


def artifact_creation_coordination_result_to_safe_dict(
    result: ArtifactCreationCoordinationResult,
) -> dict[str, Any]:
    """Return telemetry-safe coordination result serialization."""
    payload: dict[str, Any] = {
        "artifact_lookup_key_hash": result.artifact_lookup_key_hash,
        "raw_content_included": _RAW_CONTENT_INCLUDED,
        "state_version": result.state_version,
        "status": result.status.value,
    }
    if result.reservation is not None:
        payload["reservation"] = artifact_creation_reservation_to_safe_dict(result.reservation)
    if result.artifact_reference is not None:
        payload["artifact_reference"] = optimization_artifact_reference_to_safe_dict(
            result.artifact_reference
        )
    if result.reason_code is not None:
        payload["reason_code"] = result.reason_code.value
    return payload
