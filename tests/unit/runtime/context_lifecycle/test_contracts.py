# © Artur Czarnecki. All rights reserved.

"""Unified Context Lifecycle contract validation tests."""

from __future__ import annotations

from datetime import UTC, datetime
from types import MappingProxyType

import pytest

from intergrax.runtime.context_lifecycle import (
    ArtifactCompatibilityReason,
    ArtifactCompatibilityResult,
    ArtifactCompatibilityStatus,
    ArtifactCompressionTarget,
    ArtifactCreationReservation,
    ArtifactLookupKey,
    ArtifactSourceRange,
    ArtifactValidationStatus,
    ArtifactValidationSummary,
    ContextOptimizationDecision,
    ContextOptimizationMode,
    ContextOptimizationPolicy,
    ContextOptimizationReasonCode,
    EphemeralArtifactPersistencePolicy,
    ModelCallExecutionScope,
    OptimizationArtifactType,
    OptimizationExecutionGuard,
    ReusableArtifactStatus,
    ReusableOptimizationArtifact,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _lookup_key_with_refs(**overrides: object) -> ArtifactLookupKey:
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


def _lookup_key_with_range(**overrides: object) -> ArtifactLookupKey:
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
        "source_range": ArtifactSourceRange(start_sequence=0, end_sequence=5),
    }
    defaults.update(overrides)
    return ArtifactLookupKey(**defaults)  # type: ignore[arg-type]


def _validation_summary(**overrides: object) -> ArtifactValidationSummary:
    defaults: dict[str, object] = {
        "status": ArtifactValidationStatus.PASSED,
        "validation_contract_version": "validation-v1",
        "validated_at": datetime(2026, 8, 2, 12, 0, 0, tzinfo=UTC),
    }
    defaults.update(overrides)
    return ArtifactValidationSummary(**defaults)  # type: ignore[arg-type]


def _reusable_artifact(**overrides: object) -> ReusableOptimizationArtifact:
    defaults: dict[str, object] = {
        "artifact_id": "artifact-1",
        "lookup_key": _lookup_key_with_refs(),
        "artifact_content_hash": "content-hash",
        "created_at": datetime(2026, 8, 2, 12, 0, 0, tzinfo=UTC),
        "created_by_executor": "executor.message_sequence",
        "validation": _validation_summary(),
    }
    defaults.update(overrides)
    return ReusableOptimizationArtifact(**defaults)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("enum_cls", "member", "expected"),
    [
        (ModelCallExecutionScope, "PRIMARY_MODEL_CALL", "primary_model_call"),
        (ModelCallExecutionScope, "INTERNAL_OPTIMIZATION_CALL", "internal_optimization_call"),
        (ContextOptimizationMode, "EPHEMERAL_ASSEMBLY", "ephemeral_assembly"),
        (ContextOptimizationMode, "DURABLE_COMPACTION", "durable_compaction"),
        (ContextOptimizationDecision, "NO_OP", "no_op"),
        (ContextOptimizationDecision, "SELECT_ONLY", "select_only"),
        (ContextOptimizationDecision, "REUSE_ARTIFACT", "reuse_artifact"),
        (ContextOptimizationDecision, "CREATE_ARTIFACT", "create_artifact"),
        (ContextOptimizationDecision, "POLICY_BLOCKED", "policy_blocked"),
        (ContextOptimizationDecision, "FAIL_CLOSED", "fail_closed"),
        (OptimizationArtifactType, "TEXT", "text"),
        (OptimizationArtifactType, "MESSAGE_SEQUENCE", "message_sequence"),
        (OptimizationArtifactType, "FRAGMENT_SET", "fragment_set"),
        (OptimizationArtifactType, "TOOL_CATALOG", "tool_catalog"),
        (OptimizationArtifactType, "STRUCTURED_DATA", "structured_data"),
        (
            EphemeralArtifactPersistencePolicy,
            "DO_NOT_PERSIST",
            "do_not_persist_ephemeral_artifact",
        ),
        (
            EphemeralArtifactPersistencePolicy,
            "PERSIST_REUSABLE",
            "persist_reusable_artifact",
        ),
        (
            EphemeralArtifactPersistencePolicy,
            "PERSIST_AFTER_VALIDATION",
            "persist_only_after_validation",
        ),
        (
            EphemeralArtifactPersistencePolicy,
            "PERSIST_AFTER_HUMAN_REVIEW",
            "persist_only_after_human_review",
        ),
        (ReusableArtifactStatus, "VALIDATED", "validated"),
        (ReusableArtifactStatus, "INVALIDATED", "invalidated"),
        (ReusableArtifactStatus, "RETIRED", "retired"),
        (ArtifactValidationStatus, "PASSED", "passed"),
        (ArtifactValidationStatus, "FAILED", "failed"),
        (ArtifactValidationStatus, "REVOKED", "revoked"),
        (ArtifactCompatibilityStatus, "COMPATIBLE", "compatible"),
        (ArtifactCompatibilityStatus, "INCOMPATIBLE", "incompatible"),
        (
            ContextOptimizationReasonCode,
            "OPTIMIZATION_DEPTH_EXCEEDED",
            "optimization_depth_exceeded",
        ),
        (
            ArtifactCompatibilityReason,
            "TENANT_SCOPE_MISMATCH",
            "tenant_scope_mismatch",
        ),
    ],
)
def test_enum_serialized_values(enum_cls: type, member: str, expected: str) -> None:
    assert getattr(enum_cls, member).value == expected


def test_representative_contracts_are_frozen() -> None:
    policy = ContextOptimizationPolicy(
        policy_version="policy-v1",
        validation_contract_version="validation-v1",
    )
    with pytest.raises(AttributeError):
        policy.enabled = True  # type: ignore[misc]


def test_artifact_source_range_valid_single_entry() -> None:
    source_range = ArtifactSourceRange(start_sequence=3, end_sequence=3)
    assert source_range.start_sequence == 3
    assert source_range.end_sequence == 3


def test_artifact_source_range_valid_multi_entry() -> None:
    source_range = ArtifactSourceRange(start_sequence=0, end_sequence=10)
    assert source_range.end_sequence == 10


def test_artifact_source_range_rejects_negative_start() -> None:
    with pytest.raises(ValueError, match="start_sequence"):
        ArtifactSourceRange(start_sequence=-1, end_sequence=0)


def test_artifact_source_range_rejects_negative_end() -> None:
    with pytest.raises(ValueError, match="end_sequence"):
        ArtifactSourceRange(start_sequence=0, end_sequence=-1)


def test_artifact_source_range_rejects_start_greater_than_end() -> None:
    with pytest.raises(ValueError, match="start_sequence must be <= end_sequence"):
        ArtifactSourceRange(start_sequence=5, end_sequence=2)


def test_artifact_compression_target_accepts_target_tokens() -> None:
    target = ArtifactCompressionTarget(target_tokens=500)
    assert target.target_tokens == 500
    assert target.budget_class is None


def test_artifact_compression_target_accepts_budget_class() -> None:
    target = ArtifactCompressionTarget(budget_class="medium")
    assert target.budget_class == "medium"
    assert target.target_tokens is None


def test_artifact_compression_target_rejects_neither() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        ArtifactCompressionTarget()


def test_artifact_compression_target_rejects_both() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        ArtifactCompressionTarget(target_tokens=100, budget_class="medium")


def test_artifact_compression_target_rejects_zero_tokens() -> None:
    with pytest.raises(ValueError, match="target_tokens must be > 0"):
        ArtifactCompressionTarget(target_tokens=0)


def test_artifact_compression_target_rejects_negative_tokens() -> None:
    with pytest.raises(ValueError, match="target_tokens must be > 0"):
        ArtifactCompressionTarget(target_tokens=-1)


def test_artifact_compression_target_rejects_empty_budget_class() -> None:
    with pytest.raises(ValueError, match="budget_class must be non-empty"):
        ArtifactCompressionTarget(budget_class="")


def test_artifact_lookup_key_accepts_source_refs_variant() -> None:
    key = _lookup_key_with_refs()
    assert key.source_refs == ("msg-1", "msg-2")
    assert key.source_range is None


def test_artifact_lookup_key_accepts_source_range_variant() -> None:
    key = _lookup_key_with_range()
    assert key.source_range is not None
    assert key.source_refs == ()


def test_artifact_lookup_key_rejects_neither_source_locator() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        ArtifactLookupKey(
            tenant_id="tenant-1",
            context_scope_id="scope-1",
            artifact_type=OptimizationArtifactType.TEXT,
            source_content_hash="hash",
            strategy_id="strategy",
            strategy_version="1",
            policy_version="policy",
            validation_contract_version="validation",
            compression_target=ArtifactCompressionTarget(target_tokens=100),
            lossiness_profile="lossless",
        )


def test_artifact_lookup_key_rejects_both_source_locators() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        ArtifactLookupKey(
            tenant_id="tenant-1",
            context_scope_id="scope-1",
            artifact_type=OptimizationArtifactType.TEXT,
            source_content_hash="hash",
            strategy_id="strategy",
            strategy_version="1",
            policy_version="policy",
            validation_contract_version="validation",
            compression_target=ArtifactCompressionTarget(target_tokens=100),
            lossiness_profile="lossless",
            source_refs=("msg-1",),
            source_range=ArtifactSourceRange(start_sequence=0, end_sequence=1),
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "tenant_id",
        "context_scope_id",
        "source_content_hash",
        "strategy_id",
        "strategy_version",
        "policy_version",
        "validation_contract_version",
        "lossiness_profile",
    ],
)
def test_artifact_lookup_key_rejects_empty_fields(field_name: str) -> None:
    values = {
        "tenant_id": "tenant-1",
        "context_scope_id": "scope-1",
        "artifact_type": OptimizationArtifactType.TEXT,
        "source_content_hash": "hash",
        "strategy_id": "strategy",
        "strategy_version": "1",
        "policy_version": "policy",
        "validation_contract_version": "validation",
        "compression_target": ArtifactCompressionTarget(target_tokens=100),
        "lossiness_profile": "lossless",
        "source_refs": ("msg-1",),
    }
    values[field_name] = ""
    with pytest.raises(ValueError, match=field_name):
        ArtifactLookupKey(**values)


def test_artifact_lookup_key_rejects_duplicate_refs() -> None:
    with pytest.raises(ValueError, match="source_refs must not contain duplicates"):
        _lookup_key_with_refs(source_refs=("msg-1", "msg-1"))


def test_artifact_lookup_key_preserves_source_ref_order() -> None:
    key = _lookup_key_with_refs(source_refs=("z", "a", "m"))
    assert key.source_refs == ("z", "a", "m")


def test_context_optimization_policy_defaults() -> None:
    policy = ContextOptimizationPolicy(
        policy_version="policy-v1",
        validation_contract_version="validation-v1",
    )
    assert policy.enabled is False
    assert policy.mode is ContextOptimizationMode.EPHEMERAL_ASSEMBLY


def test_context_optimization_policy_rejects_llm_without_lossy() -> None:
    with pytest.raises(ValueError, match="allow_llm_summarization requires allow_lossy"):
        ContextOptimizationPolicy(
            policy_version="policy-v1",
            validation_contract_version="validation-v1",
            allow_llm_summarization=True,
        )


def test_context_optimization_policy_rejects_durable_without_receipt() -> None:
    with pytest.raises(ValueError, match="DURABLE_COMPACTION requires require_receipt"):
        ContextOptimizationPolicy(
            policy_version="policy-v1",
            validation_contract_version="validation-v1",
            mode=ContextOptimizationMode.DURABLE_COMPACTION,
            require_receipt=False,
            require_rollback_metadata=True,
        )


def test_context_optimization_policy_rejects_durable_without_rollback_metadata() -> None:
    with pytest.raises(ValueError, match="DURABLE_COMPACTION requires require_rollback_metadata"):
        ContextOptimizationPolicy(
            policy_version="policy-v1",
            validation_contract_version="validation-v1",
            mode=ContextOptimizationMode.DURABLE_COMPACTION,
            require_receipt=True,
            require_rollback_metadata=False,
        )


def test_context_optimization_policy_rejects_human_review_persistence_without_review() -> None:
    with pytest.raises(ValueError, match="PERSIST_AFTER_HUMAN_REVIEW requires require_human_review"):
        ContextOptimizationPolicy(
            policy_version="policy-v1",
            validation_contract_version="validation-v1",
            ephemeral_artifact_persistence=EphemeralArtifactPersistencePolicy.PERSIST_AFTER_HUMAN_REVIEW,
        )


def test_context_optimization_policy_rejects_persistence_without_reuse() -> None:
    with pytest.raises(ValueError, match="artifact persistence requires allow_artifact_reuse"):
        ContextOptimizationPolicy(
            policy_version="policy-v1",
            validation_contract_version="validation-v1",
            allow_artifact_reuse=False,
            ephemeral_artifact_persistence=EphemeralArtifactPersistencePolicy.PERSIST_REUSABLE,
        )


@pytest.mark.parametrize("score", [-0.1, 1.1])
def test_context_optimization_policy_rejects_invalid_quality_score(score: float) -> None:
    with pytest.raises(ValueError, match="minimum_quality_score"):
        ContextOptimizationPolicy(
            policy_version="policy-v1",
            validation_contract_version="validation-v1",
            minimum_quality_score=score,
        )


def test_context_optimization_policy_rejects_duplicate_artifact_types() -> None:
    with pytest.raises(ValueError, match="allowed_artifact_types must not contain duplicates"):
        ContextOptimizationPolicy(
            policy_version="policy-v1",
            validation_contract_version="validation-v1",
            allowed_artifact_types=(
                OptimizationArtifactType.TEXT,
                OptimizationArtifactType.TEXT,
            ),
        )


def test_context_optimization_policy_rejects_duplicate_strategy_ids() -> None:
    with pytest.raises(ValueError, match="allowed_strategy_ids must not contain duplicates"):
        ContextOptimizationPolicy(
            policy_version="policy-v1",
            validation_contract_version="validation-v1",
            allowed_strategy_ids=("strategy.a", "strategy.a"),
        )


def test_optimization_execution_guard_valid_primary() -> None:
    guard = OptimizationExecutionGuard(
        execution_scope=ModelCallExecutionScope.PRIMARY_MODEL_CALL,
        operation_id="op-primary",
        parent_operation_id=None,
        optimization_depth=0,
    )
    assert guard.optimization_depth == 0


def test_optimization_execution_guard_valid_first_internal() -> None:
    guard = OptimizationExecutionGuard(
        execution_scope=ModelCallExecutionScope.INTERNAL_OPTIMIZATION_CALL,
        operation_id="op-internal",
        parent_operation_id="op-primary",
        optimization_depth=1,
    )
    assert guard.parent_operation_id == "op-primary"


def test_optimization_execution_guard_rejects_primary_with_parent() -> None:
    with pytest.raises(ValueError, match="PRIMARY_MODEL_CALL requires parent_operation_id is None"):
        OptimizationExecutionGuard(
            execution_scope=ModelCallExecutionScope.PRIMARY_MODEL_CALL,
            operation_id="op-primary",
            parent_operation_id="op-parent",
            optimization_depth=0,
        )


def test_optimization_execution_guard_rejects_primary_with_depth_one() -> None:
    with pytest.raises(ValueError, match="PRIMARY_MODEL_CALL requires optimization_depth == 0"):
        OptimizationExecutionGuard(
            execution_scope=ModelCallExecutionScope.PRIMARY_MODEL_CALL,
            operation_id="op-primary",
            parent_operation_id=None,
            optimization_depth=1,
        )


def test_optimization_execution_guard_rejects_internal_without_parent() -> None:
    with pytest.raises(ValueError, match="INTERNAL_OPTIMIZATION_CALL requires parent_operation_id"):
        OptimizationExecutionGuard(
            execution_scope=ModelCallExecutionScope.INTERNAL_OPTIMIZATION_CALL,
            operation_id="op-internal",
            parent_operation_id=None,
            optimization_depth=1,
        )


def test_optimization_execution_guard_rejects_internal_with_depth_zero() -> None:
    with pytest.raises(ValueError, match="INTERNAL_OPTIMIZATION_CALL requires optimization_depth == 1"):
        OptimizationExecutionGuard(
            execution_scope=ModelCallExecutionScope.INTERNAL_OPTIMIZATION_CALL,
            operation_id="op-internal",
            parent_operation_id="op-primary",
            optimization_depth=0,
        )


def test_optimization_execution_guard_rejects_depth_greater_than_one() -> None:
    with pytest.raises(
        ValueError,
        match=ContextOptimizationReasonCode.OPTIMIZATION_DEPTH_EXCEEDED.value,
    ):
        OptimizationExecutionGuard(
            execution_scope=ModelCallExecutionScope.INTERNAL_OPTIMIZATION_CALL,
            operation_id="op-internal",
            parent_operation_id="op-primary",
            optimization_depth=2,
        )


def test_optimization_execution_guard_rejects_duplicate_active_key() -> None:
    with pytest.raises(ValueError, match="active_artifact_lookup_key_hashes must not contain duplicates"):
        OptimizationExecutionGuard(
            execution_scope=ModelCallExecutionScope.PRIMARY_MODEL_CALL,
            operation_id="op-primary",
            parent_operation_id=None,
            optimization_depth=0,
            active_artifact_lookup_key_hashes=("hash-a", "hash-a"),
        )


def test_optimization_execution_guard_rejects_duplicate_active_strategy() -> None:
    with pytest.raises(ValueError, match="active_strategy_ids must not contain duplicates"):
        OptimizationExecutionGuard(
            execution_scope=ModelCallExecutionScope.PRIMARY_MODEL_CALL,
            operation_id="op-primary",
            parent_operation_id=None,
            optimization_depth=0,
            active_strategy_ids=("strategy.a", "strategy.a"),
        )


def test_artifact_creation_reservation_valid_timezone_aware() -> None:
    acquired = datetime(2026, 8, 2, 12, 0, 0, tzinfo=UTC)
    deadline = datetime(2026, 8, 2, 12, 1, 0, tzinfo=UTC)
    reservation = ArtifactCreationReservation(
        reservation_id="res-1",
        artifact_lookup_key_hash="hash-1",
        tenant_id="tenant-1",
        owner_operation_id="op-1",
        acquired_at=acquired,
        lease_deadline=deadline,
    )
    assert reservation.lease_deadline > reservation.acquired_at


def test_artifact_creation_reservation_rejects_naive_datetime() -> None:
    acquired = datetime(2026, 8, 2, 12, 0, 0)
    deadline = datetime(2026, 8, 2, 12, 1, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="acquired_at must be timezone-aware"):
        ArtifactCreationReservation(
            reservation_id="res-1",
            artifact_lookup_key_hash="hash-1",
            tenant_id="tenant-1",
            owner_operation_id="op-1",
            acquired_at=acquired,
            lease_deadline=deadline,
        )


def test_artifact_creation_reservation_rejects_equal_deadline() -> None:
    acquired = datetime(2026, 8, 2, 12, 0, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="lease_deadline must be > acquired_at"):
        ArtifactCreationReservation(
            reservation_id="res-1",
            artifact_lookup_key_hash="hash-1",
            tenant_id="tenant-1",
            owner_operation_id="op-1",
            acquired_at=acquired,
            lease_deadline=acquired,
        )


def test_artifact_creation_reservation_rejects_deadline_before_acquired() -> None:
    acquired = datetime(2026, 8, 2, 12, 1, 0, tzinfo=UTC)
    deadline = datetime(2026, 8, 2, 12, 0, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="lease_deadline must be > acquired_at"):
        ArtifactCreationReservation(
            reservation_id="res-1",
            artifact_lookup_key_hash="hash-1",
            tenant_id="tenant-1",
            owner_operation_id="op-1",
            acquired_at=acquired,
            lease_deadline=deadline,
        )


@pytest.mark.parametrize(
    "field_name",
    ["reservation_id", "artifact_lookup_key_hash", "tenant_id", "owner_operation_id"],
)
def test_artifact_creation_reservation_rejects_empty_identifiers(field_name: str) -> None:
    values = {
        "reservation_id": "res-1",
        "artifact_lookup_key_hash": "hash-1",
        "tenant_id": "tenant-1",
        "owner_operation_id": "op-1",
        "acquired_at": datetime(2026, 8, 2, 12, 0, 0, tzinfo=UTC),
        "lease_deadline": datetime(2026, 8, 2, 12, 1, 0, tzinfo=UTC),
    }
    values[field_name] = ""
    with pytest.raises(ValueError, match=field_name):
        ArtifactCreationReservation(**values)


def test_reusable_optimization_artifact_valid() -> None:
    artifact = _reusable_artifact()
    assert artifact.status is ReusableArtifactStatus.VALIDATED


def test_reusable_optimization_artifact_rejects_validation_version_mismatch() -> None:
    with pytest.raises(ValueError, match="validation.validation_contract_version must equal"):
        _reusable_artifact(
            validation=_validation_summary(validation_contract_version="validation-v2"),
        )


def test_reusable_optimization_artifact_rejects_validated_with_failed_validation() -> None:
    with pytest.raises(ValueError, match="VALIDATED requires validation.status == PASSED"):
        _reusable_artifact(
            validation=_validation_summary(status=ArtifactValidationStatus.FAILED),
        )


def test_reusable_optimization_artifact_rejects_validated_with_invalidation_reason() -> None:
    with pytest.raises(ValueError, match="VALIDATED requires invalidation_reason is None"):
        _reusable_artifact(invalidation_reason="stale")


def test_reusable_optimization_artifact_rejects_invalidated_without_reason() -> None:
    with pytest.raises(ValueError, match="INVALIDATED requires invalidation_reason"):
        _reusable_artifact(status=ReusableArtifactStatus.INVALIDATED)


def test_reusable_optimization_artifact_rejects_retired_without_reason() -> None:
    with pytest.raises(ValueError, match="RETIRED requires invalidation_reason"):
        _reusable_artifact(status=ReusableArtifactStatus.RETIRED)


def test_reusable_optimization_artifact_rejects_self_superseding() -> None:
    with pytest.raises(ValueError, match="supersedes_artifact_id cannot equal artifact_id"):
        _reusable_artifact(supersedes_artifact_id="artifact-1")


def test_reusable_optimization_artifact_rejects_naive_created_at() -> None:
    with pytest.raises(ValueError, match="created_at must be timezone-aware"):
        _reusable_artifact(created_at=datetime(2026, 8, 2, 12, 0, 0))


def test_artifact_compatibility_result_compatible_with_identical_hashes() -> None:
    result = ArtifactCompatibilityResult(
        status=ArtifactCompatibilityStatus.COMPATIBLE,
        artifact_id="artifact-1",
        requested_lookup_key_hash="hash-1",
        artifact_lookup_key_hash="hash-1",
    )
    assert result.reasons == ()


def test_artifact_compatibility_result_compatible_with_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="COMPATIBLE requires matching lookup key hashes"):
        ArtifactCompatibilityResult(
            status=ArtifactCompatibilityStatus.COMPATIBLE,
            artifact_id="artifact-1",
            requested_lookup_key_hash="hash-1",
            artifact_lookup_key_hash="hash-2",
        )


def test_artifact_compatibility_result_compatible_with_reasons_rejected() -> None:
    with pytest.raises(ValueError, match="COMPATIBLE requires empty reasons"):
        ArtifactCompatibilityResult(
            status=ArtifactCompatibilityStatus.COMPATIBLE,
            artifact_id="artifact-1",
            requested_lookup_key_hash="hash-1",
            artifact_lookup_key_hash="hash-1",
            reasons=(ArtifactCompatibilityReason.ARTIFACT_TYPE_MISMATCH,),
        )


def test_artifact_compatibility_result_incompatible_without_reasons_rejected() -> None:
    with pytest.raises(ValueError, match="INCOMPATIBLE requires non-empty reasons"):
        ArtifactCompatibilityResult(
            status=ArtifactCompatibilityStatus.INCOMPATIBLE,
            artifact_id="artifact-1",
            requested_lookup_key_hash="hash-1",
            artifact_lookup_key_hash="hash-2",
        )


def test_artifact_compatibility_result_incompatible_with_reasons_accepted() -> None:
    result = ArtifactCompatibilityResult(
        status=ArtifactCompatibilityStatus.INCOMPATIBLE,
        artifact_id="artifact-1",
        requested_lookup_key_hash="hash-1",
        artifact_lookup_key_hash="hash-2",
        reasons=(ArtifactCompatibilityReason.POLICY_VERSION_MISMATCH,),
    )
    assert result.reasons == (ArtifactCompatibilityReason.POLICY_VERSION_MISMATCH,)


def test_safe_metadata_accepts_valid_nested_json_safe_metadata() -> None:
    policy = ContextOptimizationPolicy(
        policy_version="policy-v1",
        validation_contract_version="validation-v1",
        safe_metadata={"nested": {"count": 2, "enabled": True}},
    )
    assert policy.safe_metadata["nested"]["count"] == 2


def test_safe_metadata_input_mutation_does_not_mutate_contract() -> None:
    metadata = {"nested": {"count": 1}}
    policy = ContextOptimizationPolicy(
        policy_version="policy-v1",
        validation_contract_version="validation-v1",
        safe_metadata=metadata,
    )
    metadata["nested"]["count"] = 99
    assert policy.safe_metadata["nested"]["count"] == 1
    assert isinstance(policy.safe_metadata, MappingProxyType)


def test_safe_metadata_rejects_bytes() -> None:
    with pytest.raises(ValueError, match="safe_metadata must not contain bytes"):
        ContextOptimizationPolicy(
            policy_version="policy-v1",
            validation_contract_version="validation-v1",
            safe_metadata={"payload": b"raw"},
        )


class _ArbitraryObject:
    pass


def test_safe_metadata_rejects_arbitrary_object() -> None:
    with pytest.raises(ValueError, match="safe_metadata must contain only JSON-serializable values"):
        ContextOptimizationPolicy(
            policy_version="policy-v1",
            validation_contract_version="validation-v1",
            safe_metadata={"obj": _ArbitraryObject()},
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_safe_metadata_rejects_non_finite_floats(value: float) -> None:
    with pytest.raises(ValueError, match="safe_metadata must not contain non-finite floats"):
        ContextOptimizationPolicy(
            policy_version="policy-v1",
            validation_contract_version="validation-v1",
            safe_metadata={"score": value},
        )


@pytest.mark.parametrize(
    "key",
    [
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
        "CONTENT",
        "Prompt",
    ],
)
def test_safe_metadata_rejects_raw_content_keys(key: str) -> None:
    with pytest.raises(ValueError, match="safe_metadata must not contain forbidden key"):
        ContextOptimizationPolicy(
            policy_version="policy-v1",
            validation_contract_version="validation-v1",
            safe_metadata={key: "secret"},
        )


# --- strict enum type validation ---


def test_artifact_lookup_key_rejects_raw_artifact_type_string() -> None:
    with pytest.raises(ValueError, match="artifact_type must be OptimizationArtifactType"):
        _lookup_key_with_refs(artifact_type="message_sequence")


def test_context_optimization_policy_rejects_raw_mode_string() -> None:
    with pytest.raises(ValueError, match="mode must be ContextOptimizationMode"):
        ContextOptimizationPolicy(
            policy_version="policy-v1",
            validation_contract_version="validation-v1",
            mode="durable_compaction",  # type: ignore[arg-type]
        )


def test_context_optimization_policy_rejects_raw_ephemeral_persistence_string() -> None:
    with pytest.raises(
        ValueError,
        match="ephemeral_artifact_persistence must be EphemeralArtifactPersistencePolicy",
    ):
        ContextOptimizationPolicy(
            policy_version="policy-v1",
            validation_contract_version="validation-v1",
            ephemeral_artifact_persistence="do_not_persist_ephemeral_artifact",  # type: ignore[arg-type]
        )


def test_context_optimization_policy_rejects_invalid_allowed_artifact_type_item() -> None:
    with pytest.raises(ValueError, match="allowed_artifact_types item must be OptimizationArtifactType"):
        ContextOptimizationPolicy(
            policy_version="policy-v1",
            validation_contract_version="validation-v1",
            allowed_artifact_types=("message_sequence",),  # type: ignore[arg-type]
        )


def test_optimization_execution_guard_rejects_raw_execution_scope_string() -> None:
    with pytest.raises(ValueError, match="execution_scope must be ModelCallExecutionScope"):
        OptimizationExecutionGuard(
            execution_scope="primary_model_call",  # type: ignore[arg-type]
            operation_id="op-primary",
            parent_operation_id=None,
            optimization_depth=0,
        )


def test_artifact_validation_summary_rejects_raw_status_string() -> None:
    with pytest.raises(ValueError, match="status must be ArtifactValidationStatus"):
        _validation_summary(status="passed")  # type: ignore[arg-type]


def test_artifact_compatibility_result_rejects_raw_status_string() -> None:
    with pytest.raises(ValueError, match="status must be ArtifactCompatibilityStatus"):
        ArtifactCompatibilityResult(
            status="incompatible",  # type: ignore[arg-type]
            artifact_id="artifact-1",
            requested_lookup_key_hash="hash-1",
            artifact_lookup_key_hash="hash-2",
            reasons=(ArtifactCompatibilityReason.POLICY_VERSION_MISMATCH,),
        )


def test_artifact_compatibility_result_rejects_raw_reason_string() -> None:
    with pytest.raises(ValueError, match="reasons item must be ArtifactCompatibilityReason"):
        ArtifactCompatibilityResult(
            status=ArtifactCompatibilityStatus.INCOMPATIBLE,
            artifact_id="artifact-1",
            requested_lookup_key_hash="hash-1",
            artifact_lookup_key_hash="hash-2",
            reasons=("policy_version_mismatch",),  # type: ignore[arg-type]
        )


def test_reusable_optimization_artifact_rejects_raw_status_string() -> None:
    with pytest.raises(ValueError, match="status must be ReusableArtifactStatus"):
        _reusable_artifact(status="validated")  # type: ignore[arg-type]


def test_artifact_lookup_key_rejects_dict_compression_target() -> None:
    with pytest.raises(ValueError, match="compression_target must be ArtifactCompressionTarget"):
        _lookup_key_with_refs(compression_target={"target_tokens": 100})  # type: ignore[arg-type]


def test_artifact_lookup_key_rejects_dict_source_range() -> None:
    with pytest.raises(ValueError, match="source_range must be ArtifactSourceRange"):
        _lookup_key_with_range(source_range={"start_sequence": 0, "end_sequence": 5})  # type: ignore[arg-type]


def test_reusable_optimization_artifact_rejects_dict_lookup_key() -> None:
    with pytest.raises(ValueError, match="lookup_key must be ArtifactLookupKey"):
        _reusable_artifact(lookup_key={"tenant_id": "tenant-1"})  # type: ignore[arg-type]


def test_reusable_optimization_artifact_rejects_dict_validation() -> None:
    with pytest.raises(ValueError, match="validation must be ArtifactValidationSummary"):
        _reusable_artifact(validation={"status": "passed"})  # type: ignore[arg-type]


# --- strict numeric type validation ---


def test_artifact_source_range_rejects_bool_start_sequence() -> None:
    with pytest.raises(ValueError, match="start_sequence must be an integer"):
        ArtifactSourceRange(start_sequence=False, end_sequence=0)  # type: ignore[arg-type]


def test_artifact_source_range_rejects_bool_end_sequence() -> None:
    with pytest.raises(ValueError, match="end_sequence must be an integer"):
        ArtifactSourceRange(start_sequence=0, end_sequence=True)  # type: ignore[arg-type]


def test_artifact_compression_target_rejects_bool_target_tokens() -> None:
    with pytest.raises(ValueError, match="target_tokens must be an integer"):
        ArtifactCompressionTarget(target_tokens=True)  # type: ignore[arg-type]


def test_artifact_compression_target_rejects_float_target_tokens() -> None:
    with pytest.raises(ValueError, match="target_tokens must be an integer"):
        ArtifactCompressionTarget(target_tokens=1.0)  # type: ignore[arg-type]


def test_artifact_compression_target_rejects_string_target_tokens() -> None:
    with pytest.raises(ValueError, match="target_tokens must be an integer"):
        ArtifactCompressionTarget(target_tokens="100")  # type: ignore[arg-type]


def test_context_optimization_policy_rejects_bool_recent_tail_min_messages() -> None:
    with pytest.raises(ValueError, match="recent_tail_min_messages must be an integer"):
        ContextOptimizationPolicy(
            policy_version="policy-v1",
            validation_contract_version="validation-v1",
            recent_tail_min_messages=True,  # type: ignore[arg-type]
        )


def test_context_optimization_policy_rejects_bool_reservation_lease_seconds() -> None:
    with pytest.raises(ValueError, match="reservation_lease_seconds must be an integer"):
        ContextOptimizationPolicy(
            policy_version="policy-v1",
            validation_contract_version="validation-v1",
            reservation_lease_seconds=True,  # type: ignore[arg-type]
        )


def test_optimization_execution_guard_rejects_bool_optimization_depth() -> None:
    with pytest.raises(ValueError, match="optimization_depth must be an integer"):
        OptimizationExecutionGuard(
            execution_scope=ModelCallExecutionScope.PRIMARY_MODEL_CALL,
            operation_id="op-primary",
            parent_operation_id=None,
            optimization_depth=False,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "score",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
        "0.5",
    ],
)
def test_context_optimization_policy_rejects_invalid_quality_score_types(score: object) -> None:
    with pytest.raises(ValueError, match="minimum_quality_score"):
        ContextOptimizationPolicy(
            policy_version="policy-v1",
            validation_contract_version="validation-v1",
            minimum_quality_score=score,  # type: ignore[arg-type]
        )


# --- recursive safe metadata validation ---


@pytest.mark.parametrize(
    "metadata",
    [
        {"details": {"prompt": "secret"}},
        {"nested": {"RAW_SUMMARY": "secret"}},
        {"level1": {"level2": {"Messages": []}}},
    ],
)
def test_safe_metadata_rejects_nested_forbidden_keys(metadata: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="safe_metadata must not contain forbidden key"):
        ContextOptimizationPolicy(
            policy_version="policy-v1",
            validation_contract_version="validation-v1",
            safe_metadata=metadata,
        )


@pytest.mark.parametrize(
    "metadata",
    [
        {"nested": {1: "value"}},
        {"nested": {True: "value"}},
    ],
)
def test_safe_metadata_rejects_nested_non_string_keys(metadata: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="safe_metadata keys must be strings"):
        ContextOptimizationPolicy(
            policy_version="policy-v1",
            validation_contract_version="validation-v1",
            safe_metadata=metadata,
        )


def test_safe_metadata_nested_mappings_and_sequences_are_immutable() -> None:
    policy = ContextOptimizationPolicy(
        policy_version="policy-v1",
        validation_contract_version="validation-v1",
        safe_metadata={
            "outer": {
                "count": 2,
                "flags": [True, False],
                "nested": {"name": "safe"},
            },
        },
    )
    assert isinstance(policy.safe_metadata, MappingProxyType)
    assert isinstance(policy.safe_metadata["outer"], MappingProxyType)
    assert isinstance(policy.safe_metadata["outer"]["flags"], tuple)
    with pytest.raises(TypeError):
        policy.safe_metadata["outer"]["value"] = "mutated"  # type: ignore[index]
