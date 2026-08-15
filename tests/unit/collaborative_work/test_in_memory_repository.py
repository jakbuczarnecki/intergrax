# © Artur Czarnecki. All rights reserved.

"""Repository tests for Collaborative Work membership and delegation (COLLAB-WORK-1B)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.collaborative_work.in_memory_repository import (
    InMemoryAuthorityDelegationRepository,
    InMemoryPrincipalAuthorityRepository,
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
    CreatePrincipalAuthorityGrantCommand,
    CreateWorkspaceMembershipCommand,
    INITIAL_RECORD_REVISION,
    PrincipalAuthorityGrantAlreadyExists,
    PrincipalAuthorityGrantIdempotencyConflict,
    PrincipalAuthorityGrantNotFound,
    PrincipalAuthorityRepository,
    PrincipalAuthorityGrantRevisionConflict,
    PrincipalAuthorityGrantScopeKey,
    UpdateAuthorityDelegationCommand,
    UpdatePrincipalAuthorityGrantCommand,
    UpdateWorkspaceMembershipCommand,
    WorkspaceMembershipAlreadyExists,
    WorkspaceMembershipIdempotencyConflict,
    WorkspaceMembershipNotFound,
    WorkspaceMembershipRepository,
    WorkspaceMembershipRevisionConflict,
    WorkspaceMembershipScopeKey,
)
from intergrax.contracts.collaborative_work import (
    AuthorityGrantStatus,
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


def _authority_repo() -> InMemoryPrincipalAuthorityRepository:
    return InMemoryPrincipalAuthorityRepository()


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


def _create_authority_grant_command(**overrides: object) -> CreatePrincipalAuthorityGrantCommand:
    payload = {
        "tenant_id": _TENANT_A,
        "workspace_id": _WORKSPACE_A,
        "authority_grant_id": "authority-grant-1",
        "principal_id": "principal-1",
        "authority_scopes": ("workspace.read", "workspace.write"),
        "status": AuthorityGrantStatus.ACTIVE,
    }
    payload.update(overrides)
    return CreatePrincipalAuthorityGrantCommand(**payload)


@pytest.mark.parametrize(
    "repo_factory",
    [_membership_repo, _delegation_repo, _authority_repo],
)
def test_repository_protocol_is_satisfied(repo_factory: object) -> None:
    repo = repo_factory()
    if isinstance(repo, InMemoryWorkspaceMembershipRepository):
        assert isinstance(repo, WorkspaceMembershipRepository)
    elif isinstance(repo, InMemoryPrincipalAuthorityRepository):
        assert isinstance(repo, PrincipalAuthorityRepository)
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


def test_membership_idempotency_replay_after_update_returns_original_create() -> None:
    repo = _membership_repo()
    command = _create_membership_command(idempotency_key="idem-delayed")
    created = repo.create(command)
    assert created.role is WorkspaceMembershipRole.MEMBER
    assert created.revision == INITIAL_RECORD_REVISION

    updated = repo.update(
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
    assert updated.revision == created.revision + 1
    assert updated.role is WorkspaceMembershipRole.ADMIN

    replayed = repo.create(command)
    assert replayed == created
    assert replayed.role is WorkspaceMembershipRole.MEMBER
    assert replayed.revision == INITIAL_RECORD_REVISION

    current = repo.get(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE_A,
        membership_id="membership-1",
    )
    assert current == updated
    assert current is not None
    assert current.role is WorkspaceMembershipRole.ADMIN
    assert current.revision == updated.revision


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


def test_delegation_idempotency_replay_after_update_returns_original_create() -> None:
    repo = _delegation_repo()
    command = _create_delegation_command(idempotency_key="delegation-idem-delayed")
    created = repo.create(command)
    assert created.authority_scopes == ("workspace.read",)
    assert created.revision == INITIAL_RECORD_REVISION

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
    assert updated.status is DelegationStatus.REVOKED

    replayed = repo.create(command)
    assert replayed == created
    assert replayed.authority_scopes == ("workspace.read",)
    assert replayed.revision == INITIAL_RECORD_REVISION

    current = repo.get(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE_A,
        delegation_id="delegation-1",
    )
    assert current == updated
    assert current is not None
    assert current.authority_scopes == ("workspace.write",)
    assert current.status is DelegationStatus.REVOKED
    assert current.revision == updated.revision


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


def test_authority_grant_create_and_get() -> None:
    repo = _authority_repo()
    created = repo.create(_create_authority_grant_command())
    assert created.revision == INITIAL_RECORD_REVISION
    loaded = repo.get(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE_A,
        authority_grant_id="authority-grant-1",
    )
    assert loaded == created


def test_authority_grant_get_for_principal() -> None:
    repo = _authority_repo()
    created = repo.create(_create_authority_grant_command())
    loaded = repo.get_for_principal(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE_A,
        principal_id="principal-1",
    )
    assert loaded == created


def test_authority_grant_duplicate_create_raises() -> None:
    repo = _authority_repo()
    repo.create(_create_authority_grant_command())
    with pytest.raises(PrincipalAuthorityGrantAlreadyExists):
        repo.create(_create_authority_grant_command())


def test_authority_grant_duplicate_principal_raises() -> None:
    repo = _authority_repo()
    repo.create(_create_authority_grant_command())
    with pytest.raises(PrincipalAuthorityGrantAlreadyExists):
        repo.create(
            _create_authority_grant_command(
                authority_grant_id="authority-grant-2",
            )
        )


def test_authority_grant_scoped_isolation_read() -> None:
    repo = _authority_repo()
    repo.create(_create_authority_grant_command())
    assert (
        repo.get(
            tenant_id=_TENANT_B,
            workspace_id=_WORKSPACE_B,
            authority_grant_id="authority-grant-1",
        )
        is None
    )
    assert (
        repo.get_for_principal(
            tenant_id=_TENANT_B,
            workspace_id=_WORKSPACE_B,
            principal_id="principal-1",
        )
        is None
    )


def test_authority_grant_update_increments_revision() -> None:
    repo = _authority_repo()
    created = repo.create(_create_authority_grant_command())
    updated = repo.update(
        UpdatePrincipalAuthorityGrantCommand(
            scope=PrincipalAuthorityGrantScopeKey(
                tenant_id=_TENANT_A,
                workspace_id=_WORKSPACE_A,
                authority_grant_id="authority-grant-1",
            ),
            expected_revision=created.revision,
            authority_scopes=("workspace.admin",),
            status=AuthorityGrantStatus.REVOKED,
        )
    )
    assert updated.revision == created.revision + 1
    assert updated.authority_scopes == ("workspace.admin",)
    assert updated.status is AuthorityGrantStatus.REVOKED


def test_authority_grant_stale_revision_conflict_preserves_state() -> None:
    repo = _authority_repo()
    created = repo.create(_create_authority_grant_command())
    with pytest.raises(PrincipalAuthorityGrantRevisionConflict):
        repo.update(
            UpdatePrincipalAuthorityGrantCommand(
                scope=PrincipalAuthorityGrantScopeKey(
                    tenant_id=_TENANT_A,
                    workspace_id=_WORKSPACE_A,
                    authority_grant_id="authority-grant-1",
                ),
                expected_revision=created.revision + 1,
                authority_scopes=("workspace.admin",),
                status=AuthorityGrantStatus.REVOKED,
            )
        )
    current = repo.get(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE_A,
        authority_grant_id="authority-grant-1",
    )
    assert current == created


def test_authority_grant_cross_scope_update_is_not_found() -> None:
    repo = _authority_repo()
    created = repo.create(_create_authority_grant_command())
    with pytest.raises(PrincipalAuthorityGrantNotFound):
        repo.update(
            UpdatePrincipalAuthorityGrantCommand(
                scope=PrincipalAuthorityGrantScopeKey(
                    tenant_id=_TENANT_A,
                    workspace_id=_WORKSPACE_B,
                    authority_grant_id="authority-grant-1",
                ),
                expected_revision=created.revision,
                authority_scopes=("workspace.read",),
                status=AuthorityGrantStatus.REVOKED,
            )
        )


def test_authority_grant_idempotency_replay_and_conflict() -> None:
    repo = _authority_repo()
    command = _create_authority_grant_command(idempotency_key="authority-idem")
    first = repo.create(command)
    second = repo.create(command)
    assert second == first
    with pytest.raises(PrincipalAuthorityGrantIdempotencyConflict):
        repo.create(
            _create_authority_grant_command(
                idempotency_key="authority-idem",
                authority_scopes=("workspace.admin",),
            )
        )


def test_authority_grant_idempotency_replay_after_update_returns_original_create() -> None:
    repo = _authority_repo()
    command = _create_authority_grant_command(idempotency_key="authority-idem-delayed")
    created = repo.create(command)
    assert created.authority_scopes == ("workspace.read", "workspace.write")
    assert created.revision == INITIAL_RECORD_REVISION

    updated = repo.update(
        UpdatePrincipalAuthorityGrantCommand(
            scope=PrincipalAuthorityGrantScopeKey(
                tenant_id=_TENANT_A,
                workspace_id=_WORKSPACE_A,
                authority_grant_id="authority-grant-1",
            ),
            expected_revision=created.revision,
            authority_scopes=("workspace.admin",),
            status=AuthorityGrantStatus.REVOKED,
        )
    )
    assert updated.revision == created.revision + 1

    replayed = repo.create(command)
    assert replayed == created
    assert replayed.authority_scopes == ("workspace.read", "workspace.write")
    assert replayed.revision == INITIAL_RECORD_REVISION

    current = repo.get(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE_A,
        authority_grant_id="authority-grant-1",
    )
    assert current == updated
