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
    DurableCompactionPolicy,
    DurableCompactionSourceIdentity,
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


# --- TOKEN-10E-1 durable compaction serialization ---

_SHA256_HASH = "a" * 64


def _durable_policy() -> DurableCompactionPolicy:
    return DurableCompactionPolicy(
        enabled=True,
        allowed_strategy_ids=("message_sequence_summarization.v1",),
        allowed_lossiness_profiles=("lossy_summary",),
    )


def _durable_identity(**overrides: object) -> DurableCompactionSourceIdentity:
    lookup_key = _lookup_key(
        source_content_hash=_SHA256_HASH,
        strategy_id="message_sequence_summarization.v1",
    )
    values: dict[str, object] = {
        "tenant_id": lookup_key.tenant_id,
        "context_scope_id": lookup_key.context_scope_id,
        "source_revision": 2,
        "expected_active_revision": 3,
        "source_refs": lookup_key.source_refs,
        "source_content_hash": lookup_key.source_content_hash,
        "artifact_lookup_key": lookup_key,
        "strategy_id": lookup_key.strategy_id,
        "strategy_version": lookup_key.strategy_version,
        "lossiness_profile": lookup_key.lossiness_profile,
    }
    values.update(overrides)
    return DurableCompactionSourceIdentity(
        **values,
    )  # type: ignore[arg-type]


def test_durable_policy_canonical_serialization_is_deterministic() -> None:
    from intergrax.runtime.context_lifecycle import (
        compute_durable_compaction_policy_hash,
        durable_compaction_policy_to_canonical_dict,
    )

    policy = _durable_policy()
    first = durable_compaction_policy_to_canonical_dict(policy)
    second = durable_compaction_policy_to_canonical_dict(policy)
    assert first == second
    assert compute_durable_compaction_policy_hash(policy) == compute_durable_compaction_policy_hash(
        policy
    )


def test_durable_target_canonical_serialization_is_deterministic() -> None:
    from intergrax.runtime.context_lifecycle import (
        compute_durable_compaction_source_identity_hash,
        durable_compaction_source_identity_to_canonical_dict,
    )

    identity = _durable_identity()
    first = durable_compaction_source_identity_to_canonical_dict(identity)
    second = durable_compaction_source_identity_to_canonical_dict(identity)
    assert first == second
    assert compute_durable_compaction_source_identity_hash(
        identity
    ) == compute_durable_compaction_source_identity_hash(identity)


def test_durable_policy_round_trip_preserves_semantic_equality() -> None:
    from intergrax.runtime.context_lifecycle import (
        durable_compaction_policy_from_canonical_dict,
        durable_compaction_policy_to_canonical_dict,
    )

    policy = _durable_policy()
    restored = durable_compaction_policy_from_canonical_dict(
        durable_compaction_policy_to_canonical_dict(policy)
    )
    assert restored == policy


def test_durable_target_round_trip_preserves_semantic_equality() -> None:
    from intergrax.runtime.context_lifecycle import (
        durable_compaction_source_identity_from_canonical_dict,
        durable_compaction_source_identity_to_canonical_dict,
    )

    identity = _durable_identity()
    restored = durable_compaction_source_identity_from_canonical_dict(
        durable_compaction_source_identity_to_canonical_dict(identity)
    )
    assert restored == identity


def test_durable_stability_evidence_round_trip_preserves_semantic_equality() -> None:
    from intergrax.runtime.context_lifecycle import (
        DurableCompactionStabilityEvidence,
        durable_compaction_stability_evidence_from_canonical_dict,
        durable_compaction_stability_evidence_to_canonical_dict,
    )

    evidence = DurableCompactionStabilityEvidence(
        observed_stable_revision_count=2,
        observed_source_revision=2,
        observed_source_content_hash=_SHA256_HASH,
    )
    restored = durable_compaction_stability_evidence_from_canonical_dict(
        durable_compaction_stability_evidence_to_canonical_dict(evidence)
    )

    assert restored == evidence


def test_durable_policy_hash_ignores_mapping_key_order() -> None:
    from intergrax.runtime.context_lifecycle import (
        compute_durable_compaction_policy_hash,
        durable_compaction_policy_from_canonical_dict,
        durable_compaction_policy_to_canonical_dict,
    )

    policy = _durable_policy()
    payload_a = durable_compaction_policy_to_canonical_dict(policy)
    payload_b = dict(reversed(tuple(payload_a.items())))
    policy_a = durable_compaction_policy_from_canonical_dict(payload_a)
    policy_b = durable_compaction_policy_from_canonical_dict(payload_b)

    assert durable_compaction_policy_to_canonical_dict(policy_a) == (
        durable_compaction_policy_to_canonical_dict(policy_b)
    )
    assert compute_durable_compaction_policy_hash(policy_a) == (
        compute_durable_compaction_policy_hash(policy_b)
    )


def test_durable_target_source_ref_order_changes_hash() -> None:
    from intergrax.runtime.context_lifecycle import (
        compute_durable_compaction_source_identity_hash,
        DurableCompactionSourceIdentity,
    )

    identity_a = _durable_identity()
    lookup_b = _lookup_key(
        source_content_hash=_SHA256_HASH,
        strategy_id="message_sequence_summarization.v1",
        source_refs=("msg-2", "msg-1"),
    )
    identity_b = DurableCompactionSourceIdentity(
        tenant_id=lookup_b.tenant_id,
        context_scope_id=lookup_b.context_scope_id,
        source_revision=2,
        expected_active_revision=3,
        source_refs=lookup_b.source_refs,
        source_content_hash=lookup_b.source_content_hash,
        artifact_lookup_key=lookup_b,
        strategy_id=lookup_b.strategy_id,
        strategy_version=lookup_b.strategy_version,
        lossiness_profile=lookup_b.lossiness_profile,
    )
    assert compute_durable_compaction_source_identity_hash(
        identity_a
    ) != compute_durable_compaction_source_identity_hash(identity_b)


def test_durable_target_revision_change_changes_hash() -> None:
    from intergrax.runtime.context_lifecycle import compute_durable_compaction_source_identity_hash

    identity_a = _durable_identity()
    identity_b = _durable_identity(source_revision=3)
    assert compute_durable_compaction_source_identity_hash(
        identity_a
    ) != compute_durable_compaction_source_identity_hash(identity_b)


def test_durable_target_expected_active_revision_change_changes_hash() -> None:
    from intergrax.runtime.context_lifecycle import compute_durable_compaction_source_identity_hash

    identity_a = _durable_identity()
    identity_b = _durable_identity(expected_active_revision=4)
    assert compute_durable_compaction_source_identity_hash(
        identity_a
    ) != compute_durable_compaction_source_identity_hash(identity_b)


def test_durable_target_strategy_version_change_changes_hash() -> None:
    from intergrax.runtime.context_lifecycle import (
        compute_durable_compaction_source_identity_hash,
        DurableCompactionSourceIdentity,
    )

    identity_a = _durable_identity()
    lookup_b = _lookup_key(
        source_content_hash=_SHA256_HASH,
        strategy_id="message_sequence_summarization.v1",
        strategy_version="2.0.0",
    )
    identity_b = DurableCompactionSourceIdentity(
        tenant_id=lookup_b.tenant_id,
        context_scope_id=lookup_b.context_scope_id,
        source_revision=2,
        expected_active_revision=3,
        source_refs=lookup_b.source_refs,
        source_content_hash=lookup_b.source_content_hash,
        artifact_lookup_key=lookup_b,
        strategy_id=lookup_b.strategy_id,
        strategy_version=lookup_b.strategy_version,
        lossiness_profile=lookup_b.lossiness_profile,
    )
    assert compute_durable_compaction_source_identity_hash(
        identity_a
    ) != compute_durable_compaction_source_identity_hash(identity_b)


def test_durable_target_lossiness_change_changes_hash() -> None:
    from intergrax.runtime.context_lifecycle import (
        compute_durable_compaction_source_identity_hash,
        DurableCompactionSourceIdentity,
    )

    identity_a = _durable_identity()
    lookup_b = _lookup_key(
        source_content_hash=_SHA256_HASH,
        strategy_id="message_sequence_summarization.v1",
        lossiness_profile="lossless",
    )
    identity_b = DurableCompactionSourceIdentity(
        tenant_id=lookup_b.tenant_id,
        context_scope_id=lookup_b.context_scope_id,
        source_revision=2,
        expected_active_revision=3,
        source_refs=lookup_b.source_refs,
        source_content_hash=lookup_b.source_content_hash,
        artifact_lookup_key=lookup_b,
        strategy_id=lookup_b.strategy_id,
        strategy_version=lookup_b.strategy_version,
        lossiness_profile=lookup_b.lossiness_profile,
    )
    assert compute_durable_compaction_source_identity_hash(
        identity_a
    ) != compute_durable_compaction_source_identity_hash(identity_b)


def test_durable_serialized_payload_excludes_raw_content_and_marks_marker_false() -> None:
    from intergrax.runtime.context_lifecycle import (
        durable_compaction_activation_requirements_to_canonical_dict,
        DurableCompactionActivationRequirements,
    )

    requirements = DurableCompactionActivationRequirements(
        expected_active_revision=3,
        candidate_artifact_id="artifact-candidate",
        validated_artifact_id="artifact-validated",
        lineage_reference="lineage-1",
        creation_receipt_reference="receipt-1",
        rollback_source_reference="rollback-1",
    )
    payload = durable_compaction_activation_requirements_to_canonical_dict(requirements)
    assert "raw_content" not in payload
    assert payload["raw_content_included"] is False


def test_durable_policy_decode_rejects_unknown_fields() -> None:
    from intergrax.runtime.context_lifecycle import (
        durable_compaction_policy_from_canonical_dict,
        durable_compaction_policy_to_canonical_dict,
    )

    payload = durable_compaction_policy_to_canonical_dict(_durable_policy())
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        durable_compaction_policy_from_canonical_dict(payload)


def test_durable_policy_decode_rejects_malformed_enum() -> None:
    from intergrax.runtime.context_lifecycle import (
        durable_compaction_policy_from_canonical_dict,
        durable_compaction_policy_to_canonical_dict,
    )

    payload = durable_compaction_policy_to_canonical_dict(_durable_policy())
    payload["activation_mode"] = "auto_activate"
    with pytest.raises(ValueError, match="activation_mode must be a supported enum value"):
        durable_compaction_policy_from_canonical_dict(payload)


@pytest.mark.parametrize("field", ["allowed_strategy_ids", "allowed_lossiness_profiles"])
def test_durable_policy_decode_rejects_string_sequence(field: str) -> None:
    from intergrax.runtime.context_lifecycle import durable_compaction_policy_from_canonical_dict

    payload = {
        "activation_mode": "compare_and_swap",
        "allowed_strategy_ids": ["message_sequence_summarization.v1"],
        "allowed_lossiness_profiles": ["lossy_summary"],
        "enabled": True,
        "minimum_stable_revision_count": 1,
        "minimum_validation_requirement": "full",
    }
    payload[field] = "not-a-sequence"
    with pytest.raises(ValueError, match=f"{field} must be a list or tuple"):
        durable_compaction_policy_from_canonical_dict(payload)


@pytest.mark.parametrize(
    "items",
    [
        ["message_sequence_summarization.v1", 1],
        [" message_sequence_summarization.v1"],
        ["message_sequence_summarization.v1 "],
        ["message_sequence_summarization.v1", "message_sequence_summarization.v1"],
    ],
)
def test_durable_policy_decode_rejects_invalid_sequence_items(items: list[object]) -> None:
    from intergrax.runtime.context_lifecycle import durable_compaction_policy_from_canonical_dict

    payload = {
        "activation_mode": "compare_and_swap",
        "allowed_strategy_ids": items,
        "allowed_lossiness_profiles": ["lossy_summary"],
        "enabled": True,
        "minimum_stable_revision_count": 1,
        "minimum_validation_requirement": "full",
    }
    with pytest.raises(ValueError):
        durable_compaction_policy_from_canonical_dict(payload)


def test_durable_source_decode_rejects_invalid_source_refs() -> None:
    from intergrax.runtime.context_lifecycle import (
        durable_compaction_source_identity_from_canonical_dict,
        durable_compaction_source_identity_to_canonical_dict,
    )

    payload = durable_compaction_source_identity_to_canonical_dict(_durable_identity())
    payload["source_refs"] = "msg-1"
    with pytest.raises(ValueError, match="source_refs must be a list or tuple"):
        durable_compaction_source_identity_from_canonical_dict(payload)

    payload = durable_compaction_source_identity_to_canonical_dict(_durable_identity())
    payload["source_refs"] = ["msg-1", 1]
    with pytest.raises(ValueError, match="source_refs items must be strings"):
        durable_compaction_source_identity_from_canonical_dict(payload)

    payload = durable_compaction_source_identity_to_canonical_dict(_durable_identity())
    payload["source_refs"] = ["msg-1", "msg-1"]
    with pytest.raises(ValueError, match="source_refs must not contain duplicates"):
        durable_compaction_source_identity_from_canonical_dict(payload)


def test_nested_lookup_decode_rejects_string_source_refs() -> None:
    from intergrax.runtime.context_lifecycle import (
        durable_compaction_source_identity_from_canonical_dict,
        durable_compaction_source_identity_to_canonical_dict,
    )

    payload = durable_compaction_source_identity_to_canonical_dict(_durable_identity())
    payload["artifact_lookup_key"]["source_refs"] = "msg-1"
    with pytest.raises(ValueError, match="source_refs must be a list or tuple"):
        durable_compaction_source_identity_from_canonical_dict(payload)


@pytest.mark.parametrize("field", ["allowed_strategy_ids", "allowed_lossiness_profiles"])
def test_durable_policy_decode_rejects_missing_required_field(field: str) -> None:
    from intergrax.runtime.context_lifecycle import (
        durable_compaction_policy_from_canonical_dict,
        durable_compaction_policy_to_canonical_dict,
    )

    payload = durable_compaction_policy_to_canonical_dict(_durable_policy())
    del payload[field]
    with pytest.raises(ValueError, match=f"missing required field: {field}"):
        durable_compaction_policy_from_canonical_dict(payload)


def test_durable_source_decode_preserves_supported_lookup_fields() -> None:
    from intergrax.runtime.context_lifecycle import (
        durable_compaction_source_identity_from_canonical_dict,
        durable_compaction_source_identity_to_canonical_dict,
    )

    identity = _durable_identity(
        artifact_lookup_key=_lookup_key(
            source_content_hash=_SHA256_HASH,
            strategy_id="message_sequence_summarization.v1",
            protected_region_policy_version="protected-v1",
            model_family="model-family",
            locale="pl-PL",
        )
    )
    restored = durable_compaction_source_identity_from_canonical_dict(
        durable_compaction_source_identity_to_canonical_dict(identity)
    )

    assert restored == identity
