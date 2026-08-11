# © Artur Czarnecki. All rights reserved.

"""Repository tests for Collaborative Work membership and delegation (COLLAB-WORK-1B)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.collaborative_work.in_memory_repository import (
    InMemoryAuthorityDelegationRepository,
    InMemoryWorkspaceMembershipRepository,
)
from intergrax.collaborative_work.repository import (
    AuthorityDelegationAlreadyExists,
    AuthorityDelegationIdempotencyConflict,
    AuthorityDelegationNotFound,
    AuthorityDelegationRepository,
    AuthorityDelegationRevisionConflict,
    AuthorityDelegationScopeKey,
    CreateAuthorityDelegationCommand,
    CreateWorkspaceMembershipCommand,
    INITIAL_RECORD_REVISION,
    UpdateAuthorityDelegationCommand,
    UpdateWorkspaceMembershipCommand,
    WorkspaceMembershipAlreadyExists,
    WorkspaceMembershipIdempotencyConflict,
    WorkspaceMembershipNotFound,
    WorkspaceMembershipRepository,
    WorkspaceMembershipRevisionConflict,
    WorkspaceMembershipScopeKey,
)
from intergrax.contracts.collaborative_work import (
    DelegationStatus,
    MembershipStatus,
    WorkspaceMembershipRole,
)

pytestmark = pytest.mark.unit

_TENANT_A = "tenant-a"
_TENANT_B = "tenant-b"
_WORKSPACE_A = "workspace-a"
_WORKSPACE_B = "workspace-b"
_VALID_FROM = datetime(2026, 1, 1, tzinfo=UTC)
_VALID_UNTIL = datetime(2026, 12, 31, tzinfo=UTC)


def _membership_repo() -> InMemoryWorkspaceMembershipRepository:
    return InMemoryWorkspaceMembershipRepository()


def _delegation_repo() -> InMemoryAuthorityDelegationRepository:
    return InMemoryAuthorityDelegationRepository()


def _create_membership_command(**overrides: object) -> CreateWorkspaceMembershipCommand:
    payload = {
        "tenant_id": _TENANT_A,
        "workspace_id": _WORKSPACE_A,
        "membership_id": "membership-1",
        "principal_id": "principal-1",
        "role": WorkspaceMembershipRole.MEMBER,
        "status": MembershipStatus.ACTIVE,
    }
    payload.update(overrides)
    return CreateWorkspaceMembershipCommand(**payload)


def _create_delegation_command(**overrides: object) -> CreateAuthorityDelegationCommand:
    payload = {
        "tenant_id": _TENANT_A,
        "workspace_id": _WORKSPACE_A,
        "delegation_id": "delegation-1",
        "delegator_principal_id": "principal-delegator",
        "delegate_principal_id": "principal-delegate",
        "authority_scopes": ("workspace.read",),
        "status": DelegationStatus.ACTIVE,
    }
    payload.update(overrides)
    return CreateAuthorityDelegationCommand(**payload)


@pytest.mark.parametrize(
    "repo_factory",
    [_membership_repo, _delegation_repo],
)
def test_repository_protocol_is_satisfied(repo_factory: object) -> None:
    repo = repo_factory()
    if isinstance(repo, InMemoryWorkspaceMembershipRepository):
        assert isinstance(repo, WorkspaceMembershipRepository)
    else:
        assert isinstance(repo, AuthorityDelegationRepository)


def test_membership_create_and_get() -> None:
    repo = _membership_repo()
    created = repo.create(_create_membership_command())
    assert created.revision == INITIAL_RECORD_REVISION
    loaded = repo.get(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE_A,
        membership_id="membership-1",
    )
    assert loaded == created


def test_membership_duplicate_create_raises() -> None:
    repo = _membership_repo()
    repo.create(_create_membership_command())
    with pytest.raises(WorkspaceMembershipAlreadyExists):
        repo.create(_create_membership_command())


def test_membership_scoped_isolation_read() -> None:
    repo = _membership_repo()
    repo.create(_create_membership_command())
    assert (
        repo.get(
            tenant_id=_TENANT_B,
            workspace_id=_WORKSPACE_A,
            membership_id="membership-1",
        )
        is None
    )
    assert (
        repo.get(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_B,
            membership_id="membership-1",
        )
        is None
    )


def test_membership_update_success_and_revision_increment() -> None:
    repo = _membership_repo()
    created = repo.create(_create_membership_command())
    updated = repo.update(
        UpdateWorkspaceMembershipCommand(
            scope=WorkspaceMembershipScopeKey(
                tenant_id=_TENANT_A,
                workspace_id=_WORKSPACE_A,
                membership_id="membership-1",
            ),
            expected_revision=created.revision,
            role=WorkspaceMembershipRole.ADMIN,
            status=MembershipStatus.SUSPENDED,
        )
    )
    assert updated.revision == created.revision + 1
    assert updated.role is WorkspaceMembershipRole.ADMIN
    assert updated.status is MembershipStatus.SUSPENDED
    assert updated.principal_id == created.principal_id


def test_membership_stale_update_conflict_preserves_state() -> None:
    repo = _membership_repo()
    created = repo.create(_create_membership_command())
    first_update = repo.update(
        UpdateWorkspaceMembershipCommand(
            scope=WorkspaceMembershipScopeKey(
                tenant_id=_TENANT_A,
                workspace_id=_WORKSPACE_A,
                membership_id="membership-1",
            ),
            expected_revision=created.revision,
            role=WorkspaceMembershipRole.ADMIN,
            status=MembershipStatus.ACTIVE,
        )
    )
    with pytest.raises(WorkspaceMembershipRevisionConflict):
        repo.update(
            UpdateWorkspaceMembershipCommand(
                scope=WorkspaceMembershipScopeKey(
                    tenant_id=_TENANT_A,
                    workspace_id=_WORKSPACE_A,
                    membership_id="membership-1",
                ),
                expected_revision=created.revision,
                role=WorkspaceMembershipRole.OBSERVER,
                status=MembershipStatus.REVOKED,
            )
        )
    current = repo.get(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE_A,
        membership_id="membership-1",
    )
    assert current == first_update


def test_membership_cross_scope_update_is_not_found() -> None:
    repo = _membership_repo()
    created = repo.create(_create_membership_command())
    with pytest.raises(WorkspaceMembershipNotFound):
        repo.update(
            UpdateWorkspaceMembershipCommand(
                scope=WorkspaceMembershipScopeKey(
                    tenant_id=_TENANT_B,
                    workspace_id=_WORKSPACE_A,
                    membership_id="membership-1",
                ),
                expected_revision=created.revision,
                role=WorkspaceMembershipRole.ADMIN,
                status=MembershipStatus.REVOKED,
            )
        )


def test_membership_idempotency_replay_and_conflict() -> None:
    repo = _membership_repo()
    command = _create_membership_command(idempotency_key="idem-1")
    first = repo.create(command)
    second = repo.create(command)
    assert second == first
    with pytest.raises(WorkspaceMembershipIdempotencyConflict):
        repo.create(
            _create_membership_command(
                idempotency_key="idem-1",
                principal_id="principal-other",
            )
        )


def test_delegation_create_and_get() -> None:
    repo = _delegation_repo()
    created = repo.create(_create_delegation_command())
    assert created.revision == INITIAL_RECORD_REVISION
    loaded = repo.get(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE_A,
        delegation_id="delegation-1",
    )
    assert loaded == created


def test_delegation_duplicate_create_raises() -> None:
    repo = _delegation_repo()
    repo.create(_create_delegation_command())
    with pytest.raises(AuthorityDelegationAlreadyExists):
        repo.create(_create_delegation_command())


def test_delegation_scoped_isolation_read() -> None:
    repo = _delegation_repo()
    repo.create(_create_delegation_command())
    assert (
        repo.get(
            tenant_id=_TENANT_B,
            workspace_id=_WORKSPACE_A,
            delegation_id="delegation-1",
        )
        is None
    )
    assert (
        repo.get(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_B,
            delegation_id="delegation-1",
        )
        is None
    )


def test_delegation_update_success_and_revision_increment() -> None:
    repo = _delegation_repo()
    created = repo.create(_create_delegation_command())
    updated = repo.update(
        UpdateAuthorityDelegationCommand(
            scope=AuthorityDelegationScopeKey(
                tenant_id=_TENANT_A,
                workspace_id=_WORKSPACE_A,
                delegation_id="delegation-1",
            ),
            expected_revision=created.revision,
            authority_scopes=("workspace.write",),
            resource_scope="resource-1",
            valid_from=_VALID_FROM,
            valid_until=_VALID_UNTIL,
            status=DelegationStatus.REVOKED,
        )
    )
    assert updated.revision == created.revision + 1
    assert updated.authority_scopes == ("workspace.write",)
    assert updated.resource_scope == "resource-1"
    assert updated.status is DelegationStatus.REVOKED
    assert updated.delegator_principal_id == created.delegator_principal_id
    assert updated.delegate_principal_id == created.delegate_principal_id


def test_delegation_stale_update_conflict_preserves_state() -> None:
    repo = _delegation_repo()
    created = repo.create(_create_delegation_command())
    first_update = repo.update(
        UpdateAuthorityDelegationCommand(
            scope=AuthorityDelegationScopeKey(
                tenant_id=_TENANT_A,
                workspace_id=_WORKSPACE_A,
                delegation_id="delegation-1",
            ),
            expected_revision=created.revision,
            authority_scopes=("workspace.read", "workspace.write"),
            resource_scope=None,
            valid_from=None,
            valid_until=None,
            status=DelegationStatus.ACTIVE,
        )
    )
    with pytest.raises(AuthorityDelegationRevisionConflict):
        repo.update(
            UpdateAuthorityDelegationCommand(
                scope=AuthorityDelegationScopeKey(
                    tenant_id=_TENANT_A,
                    workspace_id=_WORKSPACE_A,
                    delegation_id="delegation-1",
                ),
                expected_revision=created.revision,
                authority_scopes=("workspace.admin",),
                resource_scope=None,
                valid_from=None,
                valid_until=None,
                status=DelegationStatus.REVOKED,
            )
        )
    current = repo.get(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE_A,
        delegation_id="delegation-1",
    )
    assert current == first_update


def test_delegation_cross_scope_update_is_not_found() -> None:
    repo = _delegation_repo()
    created = repo.create(_create_delegation_command())
    with pytest.raises(AuthorityDelegationNotFound):
        repo.update(
            UpdateAuthorityDelegationCommand(
                scope=AuthorityDelegationScopeKey(
                    tenant_id=_TENANT_A,
                    workspace_id=_WORKSPACE_B,
                    delegation_id="delegation-1",
                ),
                expected_revision=created.revision,
                authority_scopes=("workspace.read",),
                resource_scope=None,
                valid_from=None,
                valid_until=None,
                status=DelegationStatus.REVOKED,
            )
        )


def test_delegation_idempotency_replay_and_conflict() -> None:
    repo = _delegation_repo()
    command = _create_delegation_command(idempotency_key="delegation-idem")
    first = repo.create(command)
    second = repo.create(command)
    assert second == first
    with pytest.raises(AuthorityDelegationIdempotencyConflict):
        repo.create(
            _create_delegation_command(
                idempotency_key="delegation-idem",
                authority_scopes=("workspace.admin",),
            )
        )


def test_record_id_without_scope_does_not_authorize_access() -> None:
    membership_repo = _membership_repo()
    delegation_repo = _delegation_repo()
    membership_repo.create(_create_membership_command())
    delegation_repo.create(_create_delegation_command())
    assert (
        membership_repo.get(
            tenant_id=_TENANT_B,
            workspace_id=_WORKSPACE_B,
            membership_id="membership-1",
        )
        is None
    )
    assert (
        delegation_repo.get(
            tenant_id=_TENANT_B,
            workspace_id=_WORKSPACE_B,
            delegation_id="delegation-1",
        )
        is None
    )
