# © Artur Czarnecki. All rights reserved.

"""MP-5F-B3A-C1R1 — workspace-scoped reference resolution hardening tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.runtime.context_lifecycle import (
    InMemoryOptimizationArtifactRepository,
    OptimizationArtifactReference,
    OptimizationArtifactRepository,
    ReusableArtifactStatus,
    SQLiteOptimizationArtifactRepository,
    build_optimization_artifact_reference,
)
from intergrax.runtime.context_lifecycle.contracts import UclArtifactOwnership
from tests.unit.runtime.context_lifecycle.test_repository_contracts import (
    _lookup_key,
    _ownership_scope,
    _stored_artifact,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.fixture(params=("memory", "sqlite"))
def reference_repository(
    request: pytest.FixtureRequest, tmp_path: Path
) -> OptimizationArtifactRepository:
    if request.param == "memory":
        repo: OptimizationArtifactRepository = InMemoryOptimizationArtifactRepository()
    else:
        repo = SQLiteOptimizationArtifactRepository(str(tmp_path / "c1r1-ref.sqlite"))
    yield repo
    repo.close()


def _publish_in_workspace(
    repository: OptimizationArtifactRepository,
    *,
    workspace_id: str,
    artifact_id: str = "artifact-1",
    tenant_id: str = "tenant-1",
    context_scope_id: str = "scope-1",
) -> OptimizationArtifactReference:
    key = _lookup_key(tenant_id=tenant_id, context_scope_id=context_scope_id)
    ownership = _ownership_scope(key, workspace_id=workspace_id)
    result = repository.try_acquire_creation_reservation(
        key,
        ownership=ownership,
        owner_operation_id="operation-1",
        lease_seconds=60,
    )
    assert result.reservation is not None
    metadata_overrides = {
        "artifact_id": artifact_id,
        "lookup_key": key,
        "ownership": UclArtifactOwnership.for_workspace(ownership),
    }
    return repository.store_validated_artifact(
        reservation=result.reservation,
        artifact=_stored_artifact(**metadata_overrides),
    )


def _forged_reference(
    reference: OptimizationArtifactReference,
    **overrides: object,
) -> OptimizationArtifactReference:
    fields = {
        "tenant_id": reference.tenant_id,
        "artifact_id": reference.artifact_id,
        "artifact_lookup_key_hash": reference.artifact_lookup_key_hash,
        "artifact_content_hash": reference.artifact_content_hash,
        "artifact_type": reference.artifact_type,
        "context_scope_id": reference.context_scope_id,
        "workspace_id": reference.workspace_id,
    }
    fields.update(overrides)
    return OptimizationArtifactReference(**fields)  # type: ignore[arg-type]


def test_valid_resolve_returns_artifact(reference_repository: OptimizationArtifactRepository) -> None:
    reference = _publish_in_workspace(reference_repository, workspace_id="workspace-a")
    resolved = reference_repository.resolve(reference)
    assert resolved is not None
    assert resolved.metadata.artifact_id == reference.artifact_id
    assert resolved.metadata.workspace_id == "workspace-a"


def test_forged_workspace_resolve_returns_none(
    reference_repository: OptimizationArtifactRepository,
) -> None:
    reference = _publish_in_workspace(reference_repository, workspace_id="workspace-a")
    forged = _forged_reference(reference, workspace_id="workspace-b")
    assert reference_repository.resolve(forged) is None
    assert reference_repository.resolve(reference) is not None


def test_forged_tenant_resolve_returns_none(
    reference_repository: OptimizationArtifactRepository,
) -> None:
    reference = _publish_in_workspace(reference_repository, workspace_id="workspace-a")
    forged = _forged_reference(reference, tenant_id="tenant-2")
    assert reference_repository.resolve(forged) is None


def test_forged_context_scope_resolve_returns_none(
    reference_repository: OptimizationArtifactRepository,
) -> None:
    reference = _publish_in_workspace(reference_repository, workspace_id="workspace-a")
    forged = _forged_reference(reference, context_scope_id="other-scope")
    assert reference_repository.resolve(forged) is None


def test_valid_invalidate(reference_repository: OptimizationArtifactRepository) -> None:
    reference = _publish_in_workspace(reference_repository, workspace_id="workspace-a")
    updated = reference_repository.invalidate_artifact(reference, reason="stale")
    assert updated is not None
    assert updated.metadata.status is ReusableArtifactStatus.INVALIDATED


def test_wrong_workspace_invalidate_no_mutation(
    reference_repository: OptimizationArtifactRepository,
) -> None:
    reference = _publish_in_workspace(reference_repository, workspace_id="workspace-a")
    forged = _forged_reference(reference, workspace_id="workspace-b")
    assert reference_repository.invalidate_artifact(forged, reason="stale") is None
    resolved = reference_repository.resolve(reference)
    assert resolved is not None
    assert resolved.metadata.status is ReusableArtifactStatus.VALIDATED


def test_valid_retire(reference_repository: OptimizationArtifactRepository) -> None:
    reference = _publish_in_workspace(reference_repository, workspace_id="workspace-a")
    updated = reference_repository.retire_artifact(reference, reason="retired")
    assert updated is not None
    assert updated.metadata.status is ReusableArtifactStatus.RETIRED


def test_wrong_workspace_retire_no_mutation(
    reference_repository: OptimizationArtifactRepository,
) -> None:
    reference = _publish_in_workspace(reference_repository, workspace_id="workspace-a")
    forged = _forged_reference(reference, workspace_id="workspace-b")
    assert reference_repository.retire_artifact(forged, reason="retired") is None
    resolved = reference_repository.resolve(reference)
    assert resolved is not None
    assert resolved.metadata.status is ReusableArtifactStatus.VALIDATED


def test_build_reference_uses_canonical_workspace(
    reference_repository: OptimizationArtifactRepository,
) -> None:
    reference = _publish_in_workspace(reference_repository, workspace_id="workspace-a")
    resolved = reference_repository.resolve(reference)
    assert resolved is not None
    rebuilt = build_optimization_artifact_reference(resolved)
    assert rebuilt == reference
