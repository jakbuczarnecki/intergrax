# © Artur Czarnecki. All rights reserved.

"""Unified Context Lifecycle serialization tests."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime

import pytest

from intergrax.runtime.context_lifecycle import (
    ArtifactCompatibilityReason,
    ArtifactCompatibilityResult,
    ArtifactCompatibilityStatus,
    ArtifactCompressionTarget,
    ArtifactCreationReservation,
    ArtifactLookupKey,
    ArtifactValidationStatus,
    ArtifactValidationSummary,
    ContextOptimizationPolicy,
    OptimizationArtifactType,
    OptimizationExecutionGuard,
    ModelCallExecutionScope,
    ReusableArtifactStatus,
    ReusableOptimizationArtifact,
    artifact_compatibility_result_to_safe_dict,
    artifact_creation_reservation_to_safe_dict,
    artifact_lookup_key_to_canonical_dict,
    compute_artifact_lookup_key_hash,
    context_optimization_policy_to_safe_dict,
    optimization_execution_guard_to_safe_dict,
    reusable_optimization_artifact_to_safe_dict,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _lookup_key(**overrides: object) -> ArtifactLookupKey:
    defaults: dict[str, object] = {
        "tenant_id": "tenant-1",
        "context_scope_id": "scope-1",
        "artifact_type": OptimizationArtifactType.MESSAGE_SEQUENCE,
        "source_content_hash": "hash-abc",
        "strategy_id": "strategy.summarize",
        "strategy_version": "1.0.0",
        "policy_version": "policy-v1",
        "validation_contract_version": "validation-v1",
        "compression_target": ArtifactCompressionTarget(target_tokens=1000),
        "lossiness_profile": "lossy_summary",
        "source_refs": ("msg-1", "msg-2"),
    }
    defaults.update(overrides)
    return ArtifactLookupKey(**defaults)  # type: ignore[arg-type]


def _artifact() -> ReusableOptimizationArtifact:
    lookup_key = _lookup_key()
    return ReusableOptimizationArtifact(
        artifact_id="artifact-1",
        lookup_key=lookup_key,
        artifact_content_hash="content-hash",
        created_at=datetime(2026, 8, 2, 12, 0, 0, tzinfo=UTC),
        created_by_executor="executor.message_sequence",
        validation=ArtifactValidationSummary(
            status=ArtifactValidationStatus.PASSED,
            validation_contract_version="validation-v1",
            validated_at=datetime(2026, 8, 2, 12, 0, 0, tzinfo=UTC),
        ),
        status=ReusableArtifactStatus.VALIDATED,
    )


def test_same_key_produces_same_sha256() -> None:
    key = _lookup_key()
    digest_a = compute_artifact_lookup_key_hash(key)
    digest_b = compute_artifact_lookup_key_hash(key)
    assert digest_a == digest_b
    assert _SHA256_PATTERN.match(digest_a)


def test_separate_object_with_same_values_produces_same_sha256() -> None:
    digest_a = compute_artifact_lookup_key_hash(_lookup_key())
    digest_b = compute_artifact_lookup_key_hash(_lookup_key())
    assert digest_a == digest_b


def test_reordered_source_refs_produces_different_sha256() -> None:
    digest_a = compute_artifact_lookup_key_hash(_lookup_key(source_refs=("msg-1", "msg-2")))
    digest_b = compute_artifact_lookup_key_hash(_lookup_key(source_refs=("msg-2", "msg-1")))
    assert digest_a != digest_b


def test_changed_tenant_produces_different_sha256() -> None:
    digest_a = compute_artifact_lookup_key_hash(_lookup_key())
    digest_b = compute_artifact_lookup_key_hash(_lookup_key(tenant_id="tenant-2"))
    assert digest_a != digest_b


def test_changed_scope_produces_different_sha256() -> None:
    digest_a = compute_artifact_lookup_key_hash(_lookup_key())
    digest_b = compute_artifact_lookup_key_hash(_lookup_key(context_scope_id="scope-2"))
    assert digest_a != digest_b


def test_changed_source_hash_produces_different_sha256() -> None:
    digest_a = compute_artifact_lookup_key_hash(_lookup_key())
    digest_b = compute_artifact_lookup_key_hash(_lookup_key(source_content_hash="hash-def"))
    assert digest_a != digest_b


def test_changed_strategy_version_produces_different_sha256() -> None:
    digest_a = compute_artifact_lookup_key_hash(_lookup_key())
    digest_b = compute_artifact_lookup_key_hash(_lookup_key(strategy_version="2.0.0"))
    assert digest_a != digest_b


def test_changed_policy_version_produces_different_sha256() -> None:
    digest_a = compute_artifact_lookup_key_hash(_lookup_key())
    digest_b = compute_artifact_lookup_key_hash(_lookup_key(policy_version="policy-v2"))
    assert digest_a != digest_b


def test_changed_validation_version_produces_different_sha256() -> None:
    digest_a = compute_artifact_lookup_key_hash(_lookup_key())
    digest_b = compute_artifact_lookup_key_hash(
        _lookup_key(validation_contract_version="validation-v2")
    )
    assert digest_a != digest_b


def test_changed_compression_target_produces_different_sha256() -> None:
    digest_a = compute_artifact_lookup_key_hash(_lookup_key())
    digest_b = compute_artifact_lookup_key_hash(
        _lookup_key(compression_target=ArtifactCompressionTarget(target_tokens=500))
    )
    assert digest_a != digest_b


def test_canonical_dict_preserves_source_ref_order() -> None:
    canonical = artifact_lookup_key_to_canonical_dict(_lookup_key(source_refs=("z", "a")))
    assert canonical["source_refs"] == ["z", "a"]


def test_safe_serialization_is_json_serializable_and_stable() -> None:
    artifact = _artifact()
    payload = reusable_optimization_artifact_to_safe_dict(artifact)
    first = json.dumps(payload, sort_keys=True)
    second = json.dumps(payload, sort_keys=True)
    assert first == second
    json.loads(first)


def test_safe_serialization_uses_iso_datetime_and_enum_values() -> None:
    artifact = _artifact()
    payload = reusable_optimization_artifact_to_safe_dict(artifact)
    assert payload["created_at"].endswith("+00:00")
    assert payload["status"] == "validated"
    assert payload["lookup_key"]["artifact_type"] == "message_sequence"


def test_safe_serialization_sets_raw_content_included_false() -> None:
    artifact = _artifact()
    policy = ContextOptimizationPolicy(
        policy_version="policy-v1",
        validation_contract_version="validation-v1",
    )
    guard = OptimizationExecutionGuard(
        execution_scope=ModelCallExecutionScope.PRIMARY_MODEL_CALL,
        operation_id="op-1",
        parent_operation_id=None,
        optimization_depth=0,
    )
    reservation = ArtifactCreationReservation(
        reservation_id="res-1",
        artifact_lookup_key_hash="hash-1",
        tenant_id="tenant-1",
        owner_operation_id="op-1",
        acquired_at=datetime(2026, 8, 2, 12, 0, 0, tzinfo=UTC),
        lease_deadline=datetime(2026, 8, 2, 12, 1, 0, tzinfo=UTC),
    )
    compatibility = ArtifactCompatibilityResult(
        status=ArtifactCompatibilityStatus.INCOMPATIBLE,
        artifact_id="artifact-1",
        requested_lookup_key_hash="hash-1",
        artifact_lookup_key_hash="hash-2",
        reasons=(ArtifactCompatibilityReason.POLICY_VERSION_MISMATCH,),
    )

    assert reusable_optimization_artifact_to_safe_dict(artifact)["raw_content_included"] is False
    assert context_optimization_policy_to_safe_dict(policy)["raw_content_included"] is False
    assert optimization_execution_guard_to_safe_dict(guard)["raw_content_included"] is False
    assert artifact_creation_reservation_to_safe_dict(reservation)["raw_content_included"] is False
    assert artifact_compatibility_result_to_safe_dict(compatibility)["raw_content_included"] is False


def test_safe_serialization_does_not_expose_sensitive_identity_fields() -> None:
    artifact = _artifact()
    payload = reusable_optimization_artifact_to_safe_dict(artifact)
    serialized = json.dumps(payload)

    assert "tenant-1" not in serialized
    assert "scope-1" not in serialized
    assert "msg-1" not in serialized
    assert "msg-2" not in serialized
    assert "tenant_id" not in payload["lookup_key"]
    assert "context_scope_id" not in payload["lookup_key"]
    assert "source_refs" not in payload["lookup_key"]
    assert "source_range" not in payload["lookup_key"]


def test_safe_serialization_does_not_expose_prompt_content_values() -> None:
    artifact = ReusableOptimizationArtifact(
        artifact_id="artifact-1",
        lookup_key=_lookup_key(),
        artifact_content_hash="content-hash",
        created_at=datetime(2026, 8, 2, 12, 0, 0, tzinfo=UTC),
        created_by_executor="executor.message_sequence",
        validation=ArtifactValidationSummary(
            status=ArtifactValidationStatus.PASSED,
            validation_contract_version="validation-v1",
            validated_at=datetime(2026, 8, 2, 12, 0, 0, tzinfo=UTC),
        ),
        safe_metadata={"note": "metadata only"},
    )
    payload = reusable_optimization_artifact_to_safe_dict(artifact)
    assert "prompt" not in payload
    assert "content" not in payload
    assert "messages" not in payload
    assert "source_text" not in payload["safe_metadata"]
    assert "raw_summary" not in payload["safe_metadata"]


def test_public_imports_from_context_lifecycle_package() -> None:
    from intergrax.runtime.context_lifecycle import (  # noqa: PLC0415
        ArtifactLookupKey,
        ContextOptimizationDecision,
        ContextOptimizationPolicy,
        ModelCallExecutionScope,
        ReusableOptimizationArtifact,
        compute_artifact_lookup_key_hash,
    )

    assert ArtifactLookupKey is not None
    assert ContextOptimizationDecision is not None
    assert ContextOptimizationPolicy is not None
    assert ModelCallExecutionScope is not None
    assert ReusableOptimizationArtifact is not None
    assert compute_artifact_lookup_key_hash(_lookup_key())


# --- recursive safe serialization ---


def test_nested_safe_metadata_serializes_to_json_safe_structures() -> None:
    metadata = {
        "outer": {
            "count": 2,
            "flags": [True, False],
            "nested": {"name": "safe"},
        },
    }
    policy = ContextOptimizationPolicy(
        policy_version="policy-v1",
        validation_contract_version="validation-v1",
        safe_metadata=metadata,
    )
    payload = context_optimization_policy_to_safe_dict(policy)
    serialized = json.dumps(payload)

    assert payload["safe_metadata"]["outer"]["count"] == 2
    assert payload["safe_metadata"]["outer"]["flags"] == [True, False]
    assert payload["safe_metadata"]["outer"]["nested"]["name"] == "safe"
    assert isinstance(payload["safe_metadata"], dict)
    assert isinstance(payload["safe_metadata"]["outer"], dict)
    assert isinstance(payload["safe_metadata"]["outer"]["flags"], list)
    json.loads(serialized)


def test_nested_safe_metadata_serialization_survives_input_mutation() -> None:
    metadata = {
        "outer": {
            "count": 2,
            "nested": {"name": "safe"},
        },
    }
    policy = ContextOptimizationPolicy(
        policy_version="policy-v1",
        validation_contract_version="validation-v1",
        safe_metadata=metadata,
    )
    payload_before = context_optimization_policy_to_safe_dict(policy)
    metadata["outer"]["count"] = 99
    metadata["outer"]["nested"]["name"] = "mutated"
    payload_after = context_optimization_policy_to_safe_dict(policy)

    assert payload_before == payload_after
    assert payload_after["safe_metadata"]["outer"]["count"] == 2
    assert payload_after["safe_metadata"]["outer"]["nested"]["name"] == "safe"
