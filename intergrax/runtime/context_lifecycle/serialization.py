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
    OptimizationExecutionGuard,
    ReusableOptimizationArtifact,
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


def _mapping_to_json_dict(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(metadata)))


def context_optimization_policy_to_safe_dict(policy: ContextOptimizationPolicy) -> dict[str, Any]:
    """Return telemetry-safe policy serialization."""
    return {
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
        "safe_metadata": _mapping_to_json_dict(policy.safe_metadata),
        "validation_contract_version": policy.validation_contract_version,
    }


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
        "safe_metadata": _mapping_to_json_dict(artifact.safe_metadata),
        "status": artifact.status.value,
        "supersedes_artifact_id": artifact.supersedes_artifact_id,
        "validation": {
            "reason_codes": list(artifact.validation.reason_codes),
            "safe_metadata": _mapping_to_json_dict(artifact.validation.safe_metadata),
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
