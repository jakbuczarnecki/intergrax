# © Artur Czarnecki. All rights reserved.

"""COLLAB-WORK-1H-R2 — canonical membership and delegator membership closure tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.collaborative_work.authority import CollaborativeWorkAuthorityResolver
from intergrax.collaborative_work.in_memory_repository import (
    InMemoryAuthorityDelegationRepository,
    InMemoryPrincipalAuthorityRepository,
    InMemoryWorkspaceMembershipRepository,
)
from intergrax.collaborative_work.persistence import open_sqlite_collaborative_work_repositories
from intergrax.collaborative_work.repository import (
    CreateAuthorityDelegationCommand,
    CreatePrincipalAuthorityGrantCommand,
    CreateWorkspaceMembershipCommand,
    UpdateWorkspaceMembershipCommand,
    WorkspaceMembershipAlreadyExists,
    WorkspaceMembershipScopeKey,
)
from intergrax.contracts.collaborative_work import (
    AuthorityGrantStatus,
    DelegationStatus,
    EffectiveAuthorityDenialReason,
    EffectiveAuthorityRequest,
    MembershipStatus,
    WorkspaceMembership,
    WorkspaceMembershipRole,
)
from intergrax.contracts.runtime_policy import PolicyAction

pytestmark = pytest.mark.unit

_TENANT = "tenant-a"
_WORKSPACE = "workspace-a"
_OTHER_WORKSPACE = "workspace-b"
_OTHER_TENANT = "tenant-b"
_PRINCIPAL = "principal-acting"
_DELEGATOR = "principal-delegator"


def _membership_command(
    *,
    tenant_id: str = _TENANT,
    workspace_id: str = _WORKSPACE,
    membership_id: str,
    principal_id: str = _PRINCIPAL,
) -> CreateWorkspaceMembershipCommand:
    return CreateWorkspaceMembershipCommand(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        membership_id=membership_id,
        principal_id=principal_id,
        role=WorkspaceMembershipRole.MEMBER,
        status=MembershipStatus.ACTIVE,
    )


@pytest.fixture(params=("memory", "sqlite"))
def membership_repo(request: pytest.FixtureRequest, tmp_path: Path):
    if request.param == "memory":
        yield InMemoryWorkspaceMembershipRepository()
        return
    bundle = open_sqlite_collaborative_work_repositories(str(tmp_path / "canonical.sqlite"))
    try:
        yield bundle.membership
    finally:
        bundle.close()


def test_first_membership_for_principal_workspace_succeeds(membership_repo: object) -> None:
    created = membership_repo.create(_membership_command(membership_id="membership-1"))  # type: ignore[attr-defined]
    assert created.principal_id == _PRINCIPAL


def test_second_membership_same_principal_workspace_rejected(membership_repo: object) -> None:
    membership_repo.create(_membership_command(membership_id="membership-1"))  # type: ignore[attr-defined]
    with pytest.raises(WorkspaceMembershipAlreadyExists):
        membership_repo.create(  # type: ignore[attr-defined]
            _membership_command(membership_id="membership-2", principal_id=_PRINCIPAL)
        )


def test_same_principal_other_workspace_allowed(membership_repo: object) -> None:
    membership_repo.create(_membership_command(membership_id="membership-1"))  # type: ignore[attr-defined]
    other = membership_repo.create(  # type: ignore[attr-defined]
        _membership_command(
            membership_id="membership-2",
            workspace_id=_OTHER_WORKSPACE,
        )
    )
    assert other.workspace_id == _OTHER_WORKSPACE


def test_same_principal_other_tenant_allowed(membership_repo: object) -> None:
    membership_repo.create(_membership_command(membership_id="membership-1"))  # type: ignore[attr-defined]
    other = membership_repo.create(  # type: ignore[attr-defined]
        _membership_command(
            membership_id="membership-2",
            tenant_id=_OTHER_TENANT,
        )
    )
    assert other.tenant_id == _OTHER_TENANT


def test_get_for_principal_returns_canonical_record(membership_repo: object) -> None:
    created = membership_repo.create(_membership_command(membership_id="membership-1"))  # type: ignore[attr-defined]
    loaded = membership_repo.get_for_principal(  # type: ignore[attr-defined]
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        principal_id=_PRINCIPAL,
    )
    assert loaded == created


def test_idempotent_replay_unchanged(membership_repo: object) -> None:
    command = _membership_command(membership_id="membership-1")
    command = CreateWorkspaceMembershipCommand(
        **{**command.model_dump(), "idempotency_key": "idem-1"}
    )
    first = membership_repo.create(command)  # type: ignore[attr-defined]
    second = membership_repo.create(command)  # type: ignore[attr-defined]
    assert second == first


def test_revoked_membership_denies_and_bypass_membership_id_fails(
    membership_repo: object,
) -> None:
    created = membership_repo.create(_membership_command(membership_id="membership-1"))  # type: ignore[attr-defined]
    membership_repo.update(  # type: ignore[attr-defined]
        UpdateWorkspaceMembershipCommand(
            scope=WorkspaceMembershipScopeKey(
                tenant_id=_TENANT,
                workspace_id=_WORKSPACE,
                membership_id="membership-1",
            ),
            expected_revision=created.revision,
            role=WorkspaceMembershipRole.MEMBER,
            status=MembershipStatus.REVOKED,
        )
    )

    resolver = CollaborativeWorkAuthorityResolver(
        membership_repository=membership_repo,  # type: ignore[arg-type]
        delegation_repository=InMemoryAuthorityDelegationRepository(),
        principal_authority_repository=InMemoryPrincipalAuthorityRepository(),
    )
    decision = resolver.resolve(
        EffectiveAuthorityRequest.model_validate(
            {
                "tenant_id": _TENANT,
                "workspace_id": _WORKSPACE,
                "acting_principal_id": _PRINCIPAL,
                "requested_authority_scopes": ("workspace.read",),
                "membership": WorkspaceMembership.model_validate(
                    {
                        "membership_id": "membership-alt",
                        "tenant_id": _TENANT,
                        "workspace_id": _WORKSPACE,
                        "principal_id": _PRINCIPAL,
                        "role": WorkspaceMembershipRole.MEMBER,
                        "status": MembershipStatus.ACTIVE,
                        "revision": 0,
                    }
                ),
            }
        )
    )
    assert decision.decision.action is PolicyAction.DENY
    assert decision.denial_reason is EffectiveAuthorityDenialReason.MISSING_MEMBERSHIP


def test_sqlite_membership_persists_principal_uniqueness(tmp_path: Path) -> None:
    db_path = tmp_path / "persist.sqlite"
    bundle = open_sqlite_collaborative_work_repositories(str(db_path))
    created = bundle.membership.create(_membership_command(membership_id="membership-1"))
    bundle.close()

    reopened = open_sqlite_collaborative_work_repositories(str(db_path))
    try:
        loaded = reopened.membership.get_for_principal(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            principal_id=_PRINCIPAL,
        )
        assert loaded == created
        with pytest.raises(WorkspaceMembershipAlreadyExists):
            reopened.membership.create(
                _membership_command(membership_id="membership-2", principal_id=_PRINCIPAL)
            )
    finally:
        reopened.close()


def test_missing_delegator_membership_denies() -> None:
    membership_repo = InMemoryWorkspaceMembershipRepository()
    membership_repo.create(
        _membership_command(membership_id="membership-acting", principal_id=_PRINCIPAL)
    )
    delegation_repo = InMemoryAuthorityDelegationRepository()
    delegation_repo.create(
        CreateAuthorityDelegationCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            delegation_id="delegation-1",
            delegator_principal_id=_DELEGATOR,
            delegate_principal_id=_PRINCIPAL,
            authority_scopes=("workspace.read",),
        )
    )
    resolver = CollaborativeWorkAuthorityResolver(
        membership_repository=membership_repo,
        delegation_repository=delegation_repo,
        principal_authority_repository=InMemoryPrincipalAuthorityRepository(),
    )
    decision = resolver.resolve(
        EffectiveAuthorityRequest.model_validate(
            {
                "tenant_id": _TENANT,
                "workspace_id": _WORKSPACE,
                "acting_principal_id": _PRINCIPAL,
                "delegator_principal_id": _DELEGATOR,
                "requested_authority_scopes": ("workspace.read",),
                "membership": WorkspaceMembership.model_validate(
                    {
                        "membership_id": "membership-acting",
                        "tenant_id": _TENANT,
                        "workspace_id": _WORKSPACE,
                        "principal_id": _PRINCIPAL,
                        "role": WorkspaceMembershipRole.MEMBER,
                        "status": MembershipStatus.ACTIVE,
                        "revision": 0,
                    }
                ),
                "delegation": {
                    "delegation_id": "delegation-1",
                    "tenant_id": _TENANT,
                    "workspace_id": _WORKSPACE,
                    "delegator_principal_id": _DELEGATOR,
                    "delegate_principal_id": _PRINCIPAL,
                    "authority_scopes": ("workspace.read",),
                    "status": DelegationStatus.ACTIVE,
                    "revision": 0,
                },
            }
        )
    )
    assert decision.denial_reason is EffectiveAuthorityDenialReason.MISSING_DELEGATOR_MEMBERSHIP


@pytest.mark.parametrize(
    "delegator_status",
    (MembershipStatus.REVOKED, MembershipStatus.SUSPENDED),
)
def test_inactive_delegator_membership_denies(delegator_status: MembershipStatus) -> None:
    membership_repo = InMemoryWorkspaceMembershipRepository()
    membership_repo.create(
        _membership_command(membership_id="membership-acting", principal_id=_PRINCIPAL)
    )
    membership_repo.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            membership_id="membership-delegator",
            principal_id=_DELEGATOR,
            role=WorkspaceMembershipRole.MEMBER,
            status=delegator_status,
        )
    )
    delegation_repo = InMemoryAuthorityDelegationRepository()
    delegation = delegation_repo.create(
        CreateAuthorityDelegationCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            delegation_id="delegation-1",
            delegator_principal_id=_DELEGATOR,
            delegate_principal_id=_PRINCIPAL,
            authority_scopes=("workspace.read",),
        )
    )
    authority_repo = InMemoryPrincipalAuthorityRepository()
    authority_repo.create(
        CreatePrincipalAuthorityGrantCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            authority_grant_id="grant-delegator",
            principal_id=_DELEGATOR,
            authority_scopes=("workspace.read",),
            status=AuthorityGrantStatus.ACTIVE,
        )
    )
    resolver = CollaborativeWorkAuthorityResolver(
        membership_repository=membership_repo,
        delegation_repository=delegation_repo,
        principal_authority_repository=authority_repo,
    )
    decision = resolver.resolve(
        EffectiveAuthorityRequest.model_validate(
            {
                "tenant_id": _TENANT,
                "workspace_id": _WORKSPACE,
                "acting_principal_id": _PRINCIPAL,
                "delegator_principal_id": _DELEGATOR,
                "requested_authority_scopes": ("workspace.read",),
                "membership": WorkspaceMembership.model_validate(
                    {
                        "membership_id": "membership-acting",
                        "tenant_id": _TENANT,
                        "workspace_id": _WORKSPACE,
                        "principal_id": _PRINCIPAL,
                        "role": WorkspaceMembershipRole.MEMBER,
                        "status": MembershipStatus.ACTIVE,
                        "revision": 0,
                    }
                ),
                "delegation": delegation,
            }
        )
    )
    assert decision.denial_reason is EffectiveAuthorityDenialReason.DELEGATOR_MEMBERSHIP_NOT_ACTIVE
