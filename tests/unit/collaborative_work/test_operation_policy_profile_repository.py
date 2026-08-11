# © Artur Czarnecki. All rights reserved.

"""COLLAB-WORK-1G — operation policy profile repository tests."""

from __future__ import annotations

import pytest

from intergrax.collaborative_work.in_memory_repository import (
    InMemoryCollaborativeOperationPolicyProfileRepository,
)
from intergrax.collaborative_work.repository import (
    CollaborativeOperationPolicyProfileAlreadyExists,
    CollaborativeOperationPolicyProfileIdempotencyConflict,
    CollaborativeOperationPolicyProfileNotFound,
    CollaborativeOperationPolicyProfileRevisionConflict,
    CollaborativeOperationPolicyProfileScopeKey,
    CreateCollaborativeOperationPolicyProfileCommand,
    INITIAL_RECORD_REVISION,
    UpdateCollaborativeOperationPolicyProfileCommand,
)
from intergrax.contracts.collaborative_work import (
    CollaborativeOperationPolicyProfile,
    CollaborativeOperationPolicyProfileStatus,
    OperationPolicyRequirement,
    PolicyLayerApplicability,
)

pytestmark = pytest.mark.unit

_TENANT = "tenant-a"
_OTHER_TENANT = "tenant-b"
_WORKSPACE = "workspace-a"
_OTHER_WORKSPACE = "workspace-b"
_OPERATION = "collaborative.document.delete"
_OTHER_OPERATION = "collaborative.document.read"
_SCOPE = "document.delete"


def _repo() -> InMemoryCollaborativeOperationPolicyProfileRepository:
    return InMemoryCollaborativeOperationPolicyProfileRepository()


def _create_command(**overrides: object) -> CreateCollaborativeOperationPolicyProfileCommand:
    payload = {
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "operation_id": _OPERATION,
        "authority_scope": _SCOPE,
        "workspace_policy_applicability": PolicyLayerApplicability.REQUIRED,
        "resource_policy_applicability": PolicyLayerApplicability.REQUIRED,
        "runtime_policy_applicability": PolicyLayerApplicability.REQUIRED,
        "resource_requirement": OperationPolicyRequirement.REQUIRED,
        "meaningful_side_effect_requirement": OperationPolicyRequirement.REQUIRED,
        "status": CollaborativeOperationPolicyProfileStatus.ACTIVE,
    }
    payload.update(overrides)
    return CreateCollaborativeOperationPolicyProfileCommand(**payload)


def test_profile_create_and_get() -> None:
    repo = _repo()
    created = repo.create(_create_command())
    loaded = repo.get_for_operation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation_id=_OPERATION,
    )
    assert loaded == created
    assert loaded.revision == INITIAL_RECORD_REVISION


def test_profile_scoped_isolation() -> None:
    repo = _repo()
    repo.create(_create_command())
    assert repo.get_for_operation(
        tenant_id=_OTHER_TENANT,
        workspace_id=_WORKSPACE,
        operation_id=_OPERATION,
    ) is None
    assert repo.get_for_operation(
        tenant_id=_TENANT,
        workspace_id=_OTHER_WORKSPACE,
        operation_id=_OPERATION,
    ) is None
    assert repo.get_for_operation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation_id=_OTHER_OPERATION,
    ) is None


def test_profile_duplicate_operation_rejected() -> None:
    repo = _repo()
    repo.create(_create_command())
    with pytest.raises(CollaborativeOperationPolicyProfileAlreadyExists):
        repo.create(_create_command(idempotency_key=None))


def test_profile_revision_increment() -> None:
    repo = _repo()
    created = repo.create(_create_command())
    updated = repo.update(
        UpdateCollaborativeOperationPolicyProfileCommand(
            scope=CollaborativeOperationPolicyProfileScopeKey(
                tenant_id=_TENANT,
                workspace_id=_WORKSPACE,
                operation_id=_OPERATION,
            ),
            expected_revision=INITIAL_RECORD_REVISION,
            authority_scope=_SCOPE,
            workspace_policy_applicability=PolicyLayerApplicability.NOT_APPLICABLE,
            resource_policy_applicability=PolicyLayerApplicability.NOT_APPLICABLE,
            runtime_policy_applicability=PolicyLayerApplicability.NOT_APPLICABLE,
            resource_requirement=OperationPolicyRequirement.NOT_APPLICABLE,
            meaningful_side_effect_requirement=OperationPolicyRequirement.NOT_APPLICABLE,
            status=CollaborativeOperationPolicyProfileStatus.ACTIVE,
        )
    )
    assert created.revision == INITIAL_RECORD_REVISION
    assert updated.revision == INITIAL_RECORD_REVISION + 1


def test_profile_stale_revision_conflict_preserves_state() -> None:
    repo = _repo()
    repo.create(_create_command())
    with pytest.raises(CollaborativeOperationPolicyProfileRevisionConflict):
        repo.update(
            UpdateCollaborativeOperationPolicyProfileCommand(
                scope=CollaborativeOperationPolicyProfileScopeKey(
                    tenant_id=_TENANT,
                    workspace_id=_WORKSPACE,
                    operation_id=_OPERATION,
                ),
                expected_revision=99,
                authority_scope=_SCOPE,
                workspace_policy_applicability=PolicyLayerApplicability.REQUIRED,
                resource_policy_applicability=PolicyLayerApplicability.REQUIRED,
                runtime_policy_applicability=PolicyLayerApplicability.REQUIRED,
                resource_requirement=OperationPolicyRequirement.REQUIRED,
                meaningful_side_effect_requirement=OperationPolicyRequirement.REQUIRED,
                status=CollaborativeOperationPolicyProfileStatus.DISABLED,
            )
        )
    current = repo.get_for_operation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation_id=_OPERATION,
    )
    assert current is not None
    assert current.status is CollaborativeOperationPolicyProfileStatus.ACTIVE
    assert current.revision == INITIAL_RECORD_REVISION


def test_profile_idempotent_create_replay() -> None:
    repo = _repo()
    command = _create_command(idempotency_key="idem-profile-1")
    first = repo.create(command)
    second = repo.create(command)
    assert second == first


def test_profile_semantic_idempotency_conflict() -> None:
    repo = _repo()
    repo.create(_create_command(idempotency_key="idem-profile-2"))
    with pytest.raises(CollaborativeOperationPolicyProfileIdempotencyConflict):
        repo.create(
            _create_command(
                authority_scope="document.read",
                idempotency_key="idem-profile-2",
            )
        )


def test_profile_delayed_replay_returns_original_create_snapshot() -> None:
    repo = _repo()
    command = _create_command(idempotency_key="idem-profile-3")
    original = repo.create(command)
    repo.update(
        UpdateCollaborativeOperationPolicyProfileCommand(
            scope=CollaborativeOperationPolicyProfileScopeKey(
                tenant_id=_TENANT,
                workspace_id=_WORKSPACE,
                operation_id=_OPERATION,
            ),
            expected_revision=INITIAL_RECORD_REVISION,
            authority_scope=_SCOPE,
            workspace_policy_applicability=PolicyLayerApplicability.NOT_APPLICABLE,
            resource_policy_applicability=PolicyLayerApplicability.NOT_APPLICABLE,
            runtime_policy_applicability=PolicyLayerApplicability.NOT_APPLICABLE,
            resource_requirement=OperationPolicyRequirement.NOT_APPLICABLE,
            meaningful_side_effect_requirement=OperationPolicyRequirement.NOT_APPLICABLE,
            status=CollaborativeOperationPolicyProfileStatus.ACTIVE,
        )
    )
    replay = repo.create(command)
    assert replay.revision == original.revision
    assert replay.workspace_policy_applicability == original.workspace_policy_applicability


def test_profile_immutable_operation_identity() -> None:
    repo = _repo()
    created = repo.create(_create_command())
    with pytest.raises(CollaborativeOperationPolicyProfileNotFound):
        repo.update(
            UpdateCollaborativeOperationPolicyProfileCommand(
                scope=CollaborativeOperationPolicyProfileScopeKey(
                    tenant_id=_TENANT,
                    workspace_id=_WORKSPACE,
                    operation_id=_OTHER_OPERATION,
                ),
                expected_revision=INITIAL_RECORD_REVISION,
                authority_scope=_SCOPE,
                workspace_policy_applicability=PolicyLayerApplicability.REQUIRED,
                resource_policy_applicability=PolicyLayerApplicability.REQUIRED,
                runtime_policy_applicability=PolicyLayerApplicability.REQUIRED,
                resource_requirement=OperationPolicyRequirement.REQUIRED,
                meaningful_side_effect_requirement=OperationPolicyRequirement.REQUIRED,
                status=CollaborativeOperationPolicyProfileStatus.ACTIVE,
            )
        )
    assert created.operation_id == _OPERATION


def test_profile_unknown_applicability_cannot_be_authored() -> None:
    with pytest.raises(ValueError, match="UNKNOWN"):
        CollaborativeOperationPolicyProfile(
            operation_id=_OPERATION,
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            authority_scope=_SCOPE,
            workspace_policy_applicability=PolicyLayerApplicability.UNKNOWN,
            resource_policy_applicability=PolicyLayerApplicability.REQUIRED,
            runtime_policy_applicability=PolicyLayerApplicability.REQUIRED,
            resource_requirement=OperationPolicyRequirement.REQUIRED,
            meaningful_side_effect_requirement=OperationPolicyRequirement.REQUIRED,
            revision=0,
        )


def test_profile_contradictory_meaningful_side_effect_rejected() -> None:
    with pytest.raises(ValueError, match="runtime policy REQUIRED"):
        CollaborativeOperationPolicyProfile(
            operation_id=_OPERATION,
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            authority_scope=_SCOPE,
            workspace_policy_applicability=PolicyLayerApplicability.REQUIRED,
            resource_policy_applicability=PolicyLayerApplicability.REQUIRED,
            runtime_policy_applicability=PolicyLayerApplicability.NOT_APPLICABLE,
            resource_requirement=OperationPolicyRequirement.REQUIRED,
            meaningful_side_effect_requirement=OperationPolicyRequirement.REQUIRED,
            revision=0,
        )
