# © Artur Czarnecki. All rights reserved.

"""Repository contract validation tests (CTX-UCL-2)."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from intergrax.runtime.context_lifecycle import (
    ArtifactCreationCoordinationResult,
    ArtifactCreationCoordinationStatus,
    ArtifactCompressionTarget,
    ArtifactLookupKey,
    ArtifactValidationStatus,
    ArtifactValidationSummary,
    InMemoryOptimizationArtifactRepository,
    OptimizationArtifactRepository,
    OptimizationArtifactRepositoryCapabilities,
    OptimizationArtifactType,
    ReusableOptimizationArtifact,
    StoredOptimizationArtifact,
    build_optimization_artifact_reference,
    compute_artifact_content_hash,
    optimization_artifact_reference_to_safe_dict,
    optimization_artifact_repository_capabilities_to_safe_dict,
    stored_optimization_artifact_to_safe_dict,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


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
        "lookup_key": _lookup_key(),
        "artifact_content_hash": compute_artifact_content_hash(b"payload-bytes"),
        "created_at": datetime(2026, 8, 2, 12, 0, 0, tzinfo=UTC),
        "created_by_executor": "executor.message_sequence",
        "validation": _validation_summary(),
    }
    defaults.update(overrides)
    return ReusableOptimizationArtifact(**defaults)  # type: ignore[arg-type]


def _stored_artifact(payload: bytes = b"payload-bytes", **overrides: object) -> StoredOptimizationArtifact:
    metadata_overrides = overrides.pop("metadata", None)
    metadata = _reusable_artifact(
        artifact_content_hash=compute_artifact_content_hash(payload),
        **(metadata_overrides or {}),
    )
    return StoredOptimizationArtifact(
        metadata=metadata,
        payload=payload,
        media_type="application/octet-stream",
    )


def test_protocol_is_runtime_checkable() -> None:
    repository = InMemoryOptimizationArtifactRepository()
    assert isinstance(repository, OptimizationArtifactRepository)
    repository.close()


def test_in_memory_satisfies_protocol() -> None:
    repository = InMemoryOptimizationArtifactRepository()
    assert isinstance(repository, OptimizationArtifactRepository)
    repository.close()


def test_capabilities_exact_values() -> None:
    repository = InMemoryOptimizationArtifactRepository()
    caps = repository.capabilities
    assert caps == OptimizationArtifactRepositoryCapabilities(
        backend_id="in_memory",
        durable=False,
        shared_across_processes=False,
        supports_single_flight=True,
        supports_bounded_wait=True,
        reference_only=True,
    )
    repository.close()


def test_in_memory_capabilities_non_durable() -> None:
    repository = InMemoryOptimizationArtifactRepository()
    caps = optimization_artifact_repository_capabilities_to_safe_dict(repository.capabilities)
    assert caps["durable"] is False
    assert caps["reference_only"] is True
    assert caps["shared_across_processes"] is False
    repository.close()


def test_stored_artifact_rejects_bytearray() -> None:
    with pytest.raises(ValueError, match="payload must be bytes"):
        StoredOptimizationArtifact(
            metadata=_reusable_artifact(),
            payload=bytearray(b"payload-bytes"),  # type: ignore[arg-type]
            media_type="application/octet-stream",
        )


def test_stored_artifact_rejects_empty_payload() -> None:
    with pytest.raises(ValueError, match="payload must be non-empty"):
        StoredOptimizationArtifact(
            metadata=_reusable_artifact(artifact_content_hash=compute_artifact_content_hash(b"")),
            payload=b"",
            media_type="application/octet-stream",
        )


def test_stored_artifact_rejects_hash_mismatch() -> None:
    with pytest.raises(ValueError, match="payload SHA-256"):
        StoredOptimizationArtifact(
            metadata=_reusable_artifact(artifact_content_hash="wrong-hash"),
            payload=b"payload-bytes",
            media_type="application/octet-stream",
        )


def test_stored_artifact_payload_hidden_from_repr() -> None:
    stored = _stored_artifact(b"secret-payload")
    rendered = repr(stored)
    assert "secret-payload" not in rendered
    assert "payload" not in rendered.lower() or "payload_size" not in rendered


def test_optimization_artifact_reference_invariants() -> None:
    stored = _stored_artifact()
    reference = build_optimization_artifact_reference(stored)
    assert reference.tenant_id == "tenant-1"
    assert reference.artifact_id == "artifact-1"
    assert reference.artifact_type is OptimizationArtifactType.MESSAGE_SEQUENCE


def test_coordination_result_status_invariants() -> None:
    stored = _stored_artifact()
    reference = build_optimization_artifact_reference(stored)
    ArtifactCreationCoordinationResult(
        status=ArtifactCreationCoordinationStatus.ARTIFACT_AVAILABLE,
        artifact_lookup_key_hash="hash",
        state_version=1,
        artifact_reference=reference,
    )
    with pytest.raises(ValueError, match="ACQUIRED requires reservation"):
        ArtifactCreationCoordinationResult(
            status=ArtifactCreationCoordinationStatus.ACQUIRED,
            artifact_lookup_key_hash="hash",
            state_version=1,
        )


def test_safe_serialization_never_exposes_payload() -> None:
    secret = b"unique-secret-payload-value"
    stored = _stored_artifact(secret)
    safe = stored_optimization_artifact_to_safe_dict(stored)
    serialized = json.dumps(safe)
    assert "unique-secret-payload-value" not in serialized
    assert safe["raw_content_included"] is False
    assert safe["payload_size_bytes"] == len(secret)
    reference = build_optimization_artifact_reference(stored)
    ref_safe = optimization_artifact_reference_to_safe_dict(reference)
    assert "tenant_id" not in ref_safe


def test_public_package_imports() -> None:
    from intergrax.runtime.context_lifecycle import (
        ArtifactCreationCoordinationResult,
        OptimizationArtifactRepository,
        StoredOptimizationArtifact,
        compute_artifact_content_hash,
        stored_optimization_artifact_to_safe_dict,
    )

    assert ArtifactCreationCoordinationResult is not None
    assert OptimizationArtifactRepository is not None
    assert StoredOptimizationArtifact is not None
    assert compute_artifact_content_hash(b"x") == compute_artifact_content_hash(b"x")
    assert stored_optimization_artifact_to_safe_dict is not None
