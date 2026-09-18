# © Artur Czarnecki. All rights reserved.

"""MP-5F-B3A — canonical UCL workspace ownership tests."""

from __future__ import annotations

import pytest

from intergrax.runtime.context_lifecycle.contracts import (
    UclArtifactOwnership,
    UclArtifactOwnershipKind,
)
from intergrax.runtime.context_lifecycle.serialization import (
    ucl_artifact_ownership_from_canonical_dict,
    ucl_artifact_ownership_to_canonical_dict,
)
from intergrax.runtime.context_lifecycle import (
    InMemoryOptimizationArtifactRepository,
    compute_artifact_lookup_key_hash,
)
from tests.unit.runtime.context_lifecycle.test_repository_contracts import (
    _artifact_ownership,
    _lookup_key,
    _ownership_scope,
    _stored_artifact,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _publish(
    repository: InMemoryOptimizationArtifactRepository,
    key,
    *,
    workspace_id: str,
    artifact_id: str,
) -> None:
    ownership = _ownership_scope(key, workspace_id=workspace_id)
    metadata_overrides = {
        "artifact_id": artifact_id,
        "lookup_key": key,
        "ownership": UclArtifactOwnership.for_workspace(ownership),
    }
    result = repository.try_acquire_creation_reservation(
        key,
        ownership=ownership,
        owner_operation_id="operation-1",
        lease_seconds=60,
    )
    assert result.reservation is not None
    repository.store_validated_artifact(
        reservation=result.reservation,
        artifact=_stored_artifact(**metadata_overrides),
    )


def test_workspace_ownership_serialization_round_trip() -> None:
    ownership = _artifact_ownership()
    payload = ucl_artifact_ownership_to_canonical_dict(ownership)
    restored = ucl_artifact_ownership_from_canonical_dict(payload)
    assert restored == ownership
    assert restored.kind is UclArtifactOwnershipKind.WORKSPACE
    assert restored.scope is not None
    assert restored.scope.workspace_id == "workspace-1"


def test_same_context_scope_different_workspace_isolated() -> None:
    repository = InMemoryOptimizationArtifactRepository()
    key = _lookup_key(context_scope_id="ctx-shared")
    _publish(repository, key, workspace_id="workspace-a", artifact_id="artifact-a")
    _publish(repository, key, workspace_id="workspace-b", artifact_id="artifact-b")

    found_a = repository.lookup(key, ownership=_ownership_scope(key, workspace_id="workspace-a"))
    found_b = repository.lookup(key, ownership=_ownership_scope(key, workspace_id="workspace-b"))
    assert found_a is not None
    assert found_b is not None
    assert found_a.metadata.artifact_id == "artifact-a"
    assert found_b.metadata.artifact_id == "artifact-b"
    assert found_a.metadata.workspace_id == "workspace-a"
    assert found_b.metadata.workspace_id == "workspace-b"
    repository.close()


def test_cross_workspace_reservation_isolated() -> None:
    repository = InMemoryOptimizationArtifactRepository()
    key = _lookup_key(context_scope_id="ctx-shared")
    result_a = repository.try_acquire_creation_reservation(
        key,
        ownership=_ownership_scope(key, workspace_id="workspace-a"),
        owner_operation_id="owner-a",
        lease_seconds=60,
    )
    result_b = repository.try_acquire_creation_reservation(
        key,
        ownership=_ownership_scope(key, workspace_id="workspace-b"),
        owner_operation_id="owner-b",
        lease_seconds=60,
    )
    assert result_a.status.value == "acquired"
    assert result_b.status.value == "acquired"
    assert result_a.reservation is not None
    assert result_b.reservation is not None
    assert result_a.reservation.reservation_id != result_b.reservation.reservation_id
    repository.close()


def test_context_scope_may_differ_from_workspace() -> None:
    repository = InMemoryOptimizationArtifactRepository()
    key = _lookup_key(context_scope_id="ctx-1")
    ownership = _ownership_scope(key, workspace_id="workspace-not-ctx")
    assert ownership.workspace_id != key.context_scope_id
    result = repository.try_acquire_creation_reservation(
        key,
        ownership=ownership,
        owner_operation_id="owner-1",
        lease_seconds=60,
    )
    assert result.reservation is not None
    reference = repository.store_validated_artifact(
        reservation=result.reservation,
        artifact=_stored_artifact(
            lookup_key=key,
            ownership=UclArtifactOwnership.for_workspace(ownership),
        ),
    )
    assert reference.workspace_id == "workspace-not-ctx"
    assert reference.context_scope_id == "ctx-1"
    repository.close()


def test_lookup_hash_unchanged_when_workspace_differs() -> None:
    key_a = _lookup_key(context_scope_id="ctx-1")
    key_b = _lookup_key(context_scope_id="ctx-2")
    assert compute_artifact_lookup_key_hash(key_a) == compute_artifact_lookup_key_hash(
        _lookup_key(context_scope_id="ctx-1")
    )
    assert compute_artifact_lookup_key_hash(key_a) != compute_artifact_lookup_key_hash(key_b)
