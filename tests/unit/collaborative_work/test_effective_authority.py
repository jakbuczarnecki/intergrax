# © Artur Czarnecki. All rights reserved.

"""COLLAB-WORK-1C / 1D — authoritative effective-authority resolver tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from intergrax.collaborative_work.authority import CollaborativeWorkAuthorityResolver
from intergrax.collaborative_work.in_memory_repository import (
    InMemoryAuthorityDelegationRepository,
    InMemoryPrincipalAuthorityRepository,
    InMemoryWorkspaceMembershipRepository,
)
from intergrax.collaborative_work.repository import (
    CreateAuthorityDelegationCommand,
    CreatePrincipalAuthorityGrantCommand,
    CreateWorkspaceMembershipCommand,
    UpdateWorkspaceMembershipCommand,
    WorkspaceMembershipScopeKey,
)
from intergrax.contracts.collaborative_work import (
    AuthorityDelegation,
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
_ACTING = "principal-acting"
_DELEGATOR = "principal-delegator"
_NOW = datetime(2026, 6, 15, 12, 0, tzinfo=UTC)


def _membership_repo() -> InMemoryWorkspaceMembershipRepository:
    return InMemoryWorkspaceMembershipRepository()


def _delegation_repo() -> InMemoryAuthorityDelegationRepository:
    return InMemoryAuthorityDelegationRepository()


def _authority_repo() -> InMemoryPrincipalAuthorityRepository:
    return InMemoryPrincipalAuthorityRepository()


def _resolver(
    *,
    membership_repo: InMemoryWorkspaceMembershipRepository | None = None,
    delegation_repo: InMemoryAuthorityDelegationRepository | None = None,
    authority_repo: InMemoryPrincipalAuthorityRepository | None = None,
    now: datetime = _NOW,
    clock: object | None = None,
) -> CollaborativeWorkAuthorityResolver:
    clock_fn = clock if clock is not None else (lambda: now)
    return CollaborativeWorkAuthorityResolver(
        membership_repository=membership_repo or _membership_repo(),
        delegation_repository=delegation_repo or _delegation_repo(),
        principal_authority_repository=authority_repo or _authority_repo(),
        clock=clock_fn,
    )


def _seed_membership(
    repo: InMemoryWorkspaceMembershipRepository,
    *,
    principal_id: str = _ACTING,
    membership_id: str = "membership-1",
    status: MembershipStatus = MembershipStatus.ACTIVE,
) -> WorkspaceMembership:
    return repo.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            membership_id=membership_id,
            principal_id=principal_id,
            role=WorkspaceMembershipRole.MEMBER,
            status=status,
        )
    )


def _seed_delegated_memberships(
    repo: InMemoryWorkspaceMembershipRepository,
    *,
    delegate_status: MembershipStatus = MembershipStatus.ACTIVE,
    delegator_status: MembershipStatus = MembershipStatus.ACTIVE,
) -> tuple[WorkspaceMembership, WorkspaceMembership]:
    delegate = _seed_membership(
        repo,
        principal_id=_ACTING,
        membership_id="membership-acting",
        status=delegate_status,
    )
    delegator = _seed_membership(
        repo,
        principal_id=_DELEGATOR,
        membership_id="membership-delegator",
        status=delegator_status,
    )
    return delegate, delegator


def _seed_delegation(
    repo: InMemoryAuthorityDelegationRepository,
    **overrides: object,
) -> AuthorityDelegation:
    payload = {
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "delegation_id": "delegation-1",
        "delegator_principal_id": _DELEGATOR,
        "delegate_principal_id": _ACTING,
        "authority_scopes": ("workspace.read", "workspace.write"),
        "status": DelegationStatus.ACTIVE,
    }
    payload.update(overrides)
    return repo.create(CreateAuthorityDelegationCommand(**payload))


def _seed_authority(
    repo: InMemoryPrincipalAuthorityRepository,
    *,
    principal_id: str = _ACTING,
    authority_scopes: tuple[str, ...] = ("workspace.read", "workspace.write"),
    status: AuthorityGrantStatus = AuthorityGrantStatus.ACTIVE,
) -> object:
    return repo.create(
        CreatePrincipalAuthorityGrantCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            authority_grant_id="authority-grant-1",
            principal_id=principal_id,
            authority_scopes=authority_scopes,
            status=status,
        )
    )


def _membership_locator(**overrides: object) -> WorkspaceMembership:
    payload = {
        "membership_id": "membership-1",
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "principal_id": _ACTING,
        "role": WorkspaceMembershipRole.MEMBER,
        "status": MembershipStatus.ACTIVE,
        "revision": 0,
    }
    payload.update(overrides)
    return WorkspaceMembership.model_validate(payload)


def _delegation_locator(**overrides: object) -> AuthorityDelegation:
    payload = {
        "delegation_id": "delegation-1",
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "delegator_principal_id": _DELEGATOR,
        "delegate_principal_id": _ACTING,
        "authority_scopes": ("workspace.read", "workspace.write"),
        "status": DelegationStatus.ACTIVE,
        "revision": 0,
    }
    payload.update(overrides)
    return AuthorityDelegation.model_validate(payload)


def _acting_membership_locator(**overrides: object) -> WorkspaceMembership:
    return _membership_locator(membership_id="membership-acting", **overrides)


def _request(**overrides: object) -> EffectiveAuthorityRequest:
    payload = {
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "acting_principal_id": _ACTING,
        "requested_authority_scopes": ("workspace.read",),
    }
    payload.update(overrides)
    return EffectiveAuthorityRequest.model_validate(payload)


def test_membership_repository_state_overrides_embedded_request_state() -> None:
    membership_repo = _membership_repo()
    _seed_membership(membership_repo, status=MembershipStatus.ACTIVE)

    resolver = _resolver(membership_repo=membership_repo)
    decision = resolver.resolve(
        _request(
            membership=_membership_locator(
                status=MembershipStatus.SUSPENDED,
                revision=0,
            )
        )
    )

    assert decision.decision.action is PolicyAction.DENY
    assert decision.denial_reason is EffectiveAuthorityDenialReason.MISSING_BASE_AUTHORITY


def test_membership_revoked_in_repository_denies_despite_stale_active_embed() -> None:
    membership_repo = _membership_repo()
    created = _seed_membership(membership_repo)
    membership_repo.update(
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

    resolver = _resolver(membership_repo=membership_repo)
    decision = resolver.resolve(
        _request(membership=_membership_locator(status=MembershipStatus.ACTIVE, revision=0))
    )

    assert decision.decision.action is PolicyAction.DENY
    assert decision.denial_reason is EffectiveAuthorityDenialReason.MEMBERSHIP_NOT_ACTIVE


def test_delegation_repository_state_overrides_embedded_request_state() -> None:
    membership_repo = _membership_repo()
    delegation_repo = _delegation_repo()
    _seed_delegated_memberships(membership_repo)
    _seed_delegation(delegation_repo, authority_scopes=("workspace.read",))

    resolver = _resolver(membership_repo=membership_repo, delegation_repo=delegation_repo)
    decision = resolver.resolve(
        _request(
            delegator_principal_id=_DELEGATOR,
            requested_authority_scopes=("workspace.write",),
            membership=_acting_membership_locator(),
            delegation=_delegation_locator(
                authority_scopes=("workspace.read", "workspace.write", "workspace.admin"),
            ),
        )
    )

    assert decision.decision.action is PolicyAction.DENY
    assert decision.denial_reason is EffectiveAuthorityDenialReason.INSUFFICIENT_DELEGATION_SCOPE


def test_missing_membership_locator_fail_closed() -> None:
    decision = _resolver().resolve(_request())
    assert decision.decision.action is PolicyAction.DENY
    assert decision.denial_reason is EffectiveAuthorityDenialReason.MISSING_MEMBERSHIP


def test_missing_authoritative_membership_record() -> None:
    decision = _resolver().resolve(_request(membership=_membership_locator()))
    assert decision.decision.action is PolicyAction.DENY
    assert decision.denial_reason is EffectiveAuthorityDenialReason.MISSING_MEMBERSHIP


def test_inactive_membership_denied() -> None:
    membership_repo = _membership_repo()
    _seed_membership(membership_repo, status=MembershipStatus.SUSPENDED)

    decision = _resolver(membership_repo=membership_repo).resolve(
        _request(membership=_membership_locator(status=MembershipStatus.ACTIVE))
    )
    assert decision.denial_reason is EffectiveAuthorityDenialReason.MEMBERSHIP_NOT_ACTIVE


def test_cross_scope_membership_locator_cannot_authorize() -> None:
    membership_repo = _membership_repo()
    _seed_membership(membership_repo, principal_id="other-principal")

    decision = _resolver(membership_repo=membership_repo).resolve(
        _request(membership=_membership_locator(principal_id=_ACTING))
    )
    assert decision.denial_reason is EffectiveAuthorityDenialReason.MISSING_MEMBERSHIP


def test_delegated_request_missing_delegation_locator() -> None:
    membership_repo = _membership_repo()
    _seed_delegated_memberships(membership_repo)

    decision = _resolver(membership_repo=membership_repo).resolve(
        _request(
            delegator_principal_id=_DELEGATOR,
            membership=_acting_membership_locator(),
        )
    )
    assert decision.denial_reason is EffectiveAuthorityDenialReason.MISSING_DELEGATION


def test_missing_authoritative_delegation_record() -> None:
    membership_repo = _membership_repo()
    _seed_delegated_memberships(membership_repo)

    decision = _resolver(membership_repo=membership_repo).resolve(
        _request(
            delegator_principal_id=_DELEGATOR,
            membership=_acting_membership_locator(),
            delegation=_delegation_locator(),
        )
    )
    assert decision.denial_reason is EffectiveAuthorityDenialReason.MISSING_DELEGATION


def test_inactive_delegation_denied() -> None:
    membership_repo = _membership_repo()
    delegation_repo = _delegation_repo()
    _seed_delegated_memberships(membership_repo)
    _seed_delegation(delegation_repo, status=DelegationStatus.REVOKED)

    decision = _resolver(membership_repo=membership_repo, delegation_repo=delegation_repo).resolve(
        _request(
            delegator_principal_id=_DELEGATOR,
            membership=_acting_membership_locator(),
            delegation=_delegation_locator(status=DelegationStatus.ACTIVE),
        )
    )
    assert decision.denial_reason is EffectiveAuthorityDenialReason.DELEGATION_NOT_ACTIVE


def test_delegation_not_yet_valid() -> None:
    membership_repo = _membership_repo()
    delegation_repo = _delegation_repo()
    _seed_delegated_memberships(membership_repo)
    _seed_delegation(
        delegation_repo,
        valid_from=_NOW + timedelta(days=1),
        valid_until=_NOW + timedelta(days=30),
    )

    decision = _resolver(membership_repo=membership_repo, delegation_repo=delegation_repo).resolve(
        _request(
            delegator_principal_id=_DELEGATOR,
            membership=_acting_membership_locator(),
            delegation=_delegation_locator(
                valid_from=_NOW + timedelta(days=1),
                valid_until=_NOW + timedelta(days=30),
            ),
        )
    )
    assert decision.denial_reason is EffectiveAuthorityDenialReason.DELEGATION_NOT_ACTIVE


def test_delegation_expired() -> None:
    membership_repo = _membership_repo()
    delegation_repo = _delegation_repo()
    _seed_delegated_memberships(membership_repo)
    _seed_delegation(
        delegation_repo,
        valid_from=_NOW - timedelta(days=30),
        valid_until=_NOW - timedelta(seconds=1),
    )

    decision = _resolver(membership_repo=membership_repo, delegation_repo=delegation_repo).resolve(
        _request(
            delegator_principal_id=_DELEGATOR,
            membership=_acting_membership_locator(),
            delegation=_delegation_locator(
                valid_from=_NOW - timedelta(days=30),
                valid_until=_NOW - timedelta(seconds=1),
            ),
        )
    )
    assert decision.denial_reason is EffectiveAuthorityDenialReason.DELEGATION_NOT_ACTIVE


def test_requested_scope_exceeds_delegation() -> None:
    membership_repo = _membership_repo()
    delegation_repo = _delegation_repo()
    _seed_delegated_memberships(membership_repo)
    _seed_delegation(delegation_repo, authority_scopes=("workspace.read",))

    decision = _resolver(membership_repo=membership_repo, delegation_repo=delegation_repo).resolve(
        _request(
            delegator_principal_id=_DELEGATOR,
            requested_authority_scopes=("workspace.write",),
            membership=_acting_membership_locator(),
            delegation=_delegation_locator(authority_scopes=("workspace.read", "workspace.write")),
        )
    )
    assert decision.denial_reason is EffectiveAuthorityDenialReason.INSUFFICIENT_DELEGATION_SCOPE


def test_delegate_mismatch_cannot_authorize() -> None:
    membership_repo = _membership_repo()
    delegation_repo = _delegation_repo()
    _seed_delegated_memberships(membership_repo)
    _seed_delegation(delegation_repo, delegate_principal_id="other-delegate")

    decision = _resolver(membership_repo=membership_repo, delegation_repo=delegation_repo).resolve(
        _request(
            delegator_principal_id=_DELEGATOR,
            membership=_acting_membership_locator(),
            delegation=_delegation_locator(delegate_principal_id=_ACTING),
        )
    )
    assert decision.denial_reason is EffectiveAuthorityDenialReason.MISSING_DELEGATION


def test_delegator_mismatch_cannot_authorize() -> None:
    membership_repo = _membership_repo()
    delegation_repo = _delegation_repo()
    _seed_delegated_memberships(membership_repo)
    _seed_delegation(delegation_repo, delegator_principal_id="other-delegator")

    decision = _resolver(membership_repo=membership_repo, delegation_repo=delegation_repo).resolve(
        _request(
            delegator_principal_id=_DELEGATOR,
            membership=_acting_membership_locator(),
            delegation=_delegation_locator(delegator_principal_id=_DELEGATOR),
        )
    )
    assert decision.denial_reason is EffectiveAuthorityDenialReason.MISSING_DELEGATION


def test_resource_scope_mismatch_denied() -> None:
    membership_repo = _membership_repo()
    delegation_repo = _delegation_repo()
    _seed_delegated_memberships(membership_repo)
    _seed_delegation(delegation_repo, resource_scope="artifact-a")

    decision = _resolver(membership_repo=membership_repo, delegation_repo=delegation_repo).resolve(
        _request(
            delegator_principal_id=_DELEGATOR,
            resource_scope="artifact-b",
            membership=_acting_membership_locator(),
            delegation=_delegation_locator(resource_scope="artifact-a"),
        )
    )
    assert decision.denial_reason is EffectiveAuthorityDenialReason.INSUFFICIENT_DELEGATION_SCOPE


def test_resource_limited_delegation_requires_request_resource_scope() -> None:
    membership_repo = _membership_repo()
    delegation_repo = _delegation_repo()
    _seed_delegated_memberships(membership_repo)
    _seed_delegation(delegation_repo, resource_scope="artifact-a")

    decision = _resolver(membership_repo=membership_repo, delegation_repo=delegation_repo).resolve(
        _request(
            delegator_principal_id=_DELEGATOR,
            membership=_acting_membership_locator(),
            delegation=_delegation_locator(resource_scope="artifact-a"),
        )
    )
    assert decision.denial_reason is EffectiveAuthorityDenialReason.INSUFFICIENT_DELEGATION_SCOPE


def test_valid_delegated_evidence_reaches_collaborative_slice_allow() -> None:
    membership_repo = _membership_repo()
    delegation_repo = _delegation_repo()
    authority_repo = _authority_repo()
    _seed_delegated_memberships(membership_repo)
    _seed_delegation(
        delegation_repo,
        resource_scope="artifact-a",
        valid_from=_NOW - timedelta(days=1),
        valid_until=_NOW + timedelta(days=1),
    )
    _seed_authority(authority_repo, principal_id=_DELEGATOR)

    decision = _resolver(
        membership_repo=membership_repo,
        delegation_repo=delegation_repo,
        authority_repo=authority_repo,
    ).resolve(
        _request(
            delegator_principal_id=_DELEGATOR,
            resource_scope="artifact-a",
            membership=_acting_membership_locator(),
            delegation=_delegation_locator(
                resource_scope="artifact-a",
                valid_from=_NOW - timedelta(days=1),
                valid_until=_NOW + timedelta(days=1),
            ),
        )
    )

    assert decision.decision.action is PolicyAction.ALLOW
    assert decision.denial_reason is None
    assert (
        decision.decision.policy_rule_id
        == "collaborative_work.effective_authority.collaborative_slice_allow"
    )
    assert "workspace, resource, and runtime/tool policy were not evaluated" in decision.decision.reason


def test_direct_acting_active_membership_missing_base_authority_denies() -> None:
    membership_repo = _membership_repo()
    _seed_membership(membership_repo)

    decision = _resolver(membership_repo=membership_repo).resolve(
        _request(membership=_membership_locator())
    )

    assert decision.decision.action is PolicyAction.DENY
    assert decision.denial_reason is EffectiveAuthorityDenialReason.MISSING_BASE_AUTHORITY


def test_direct_acting_active_membership_with_base_authority_allows() -> None:
    membership_repo = _membership_repo()
    authority_repo = _authority_repo()
    _seed_membership(membership_repo)
    _seed_authority(authority_repo)

    decision = _resolver(membership_repo=membership_repo, authority_repo=authority_repo).resolve(
        _request(membership=_membership_locator())
    )

    assert decision.decision.action is PolicyAction.ALLOW
    assert decision.denial_reason is None


def test_direct_acting_insufficient_base_scope_denies() -> None:
    membership_repo = _membership_repo()
    authority_repo = _authority_repo()
    _seed_membership(membership_repo)
    _seed_authority(authority_repo, authority_scopes=("workspace.read",))

    decision = _resolver(membership_repo=membership_repo, authority_repo=authority_repo).resolve(
        _request(
            requested_authority_scopes=("workspace.write",),
            membership=_membership_locator(),
        )
    )

    assert decision.decision.action is PolicyAction.DENY
    assert decision.denial_reason is EffectiveAuthorityDenialReason.INSUFFICIENT_BASE_AUTHORITY


def test_inactive_base_authority_denies() -> None:
    membership_repo = _membership_repo()
    authority_repo = _authority_repo()
    _seed_membership(membership_repo)
    _seed_authority(authority_repo, status=AuthorityGrantStatus.REVOKED)

    decision = _resolver(membership_repo=membership_repo, authority_repo=authority_repo).resolve(
        _request(membership=_membership_locator())
    )

    assert decision.denial_reason is EffectiveAuthorityDenialReason.BASE_AUTHORITY_NOT_ACTIVE


def test_admin_membership_without_base_authority_does_not_authorize() -> None:
    membership_repo = _membership_repo()
    membership_repo.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            membership_id="membership-1",
            principal_id=_ACTING,
            role=WorkspaceMembershipRole.ADMIN,
            status=MembershipStatus.ACTIVE,
        )
    )

    decision = _resolver(membership_repo=membership_repo).resolve(
        _request(
            membership=_membership_locator(role=WorkspaceMembershipRole.ADMIN),
        )
    )

    assert decision.decision.action is PolicyAction.DENY
    assert decision.denial_reason is EffectiveAuthorityDenialReason.MISSING_BASE_AUTHORITY


def test_delegation_scopes_alone_do_not_manufacture_delegator_authority() -> None:
    membership_repo = _membership_repo()
    delegation_repo = _delegation_repo()
    _seed_delegated_memberships(membership_repo)
    _seed_delegation(delegation_repo, authority_scopes=("workspace.admin",))

    decision = _resolver(membership_repo=membership_repo, delegation_repo=delegation_repo).resolve(
        _request(
            delegator_principal_id=_DELEGATOR,
            requested_authority_scopes=("workspace.admin",),
            membership=_acting_membership_locator(),
            delegation=_delegation_locator(authority_scopes=("workspace.admin",)),
        )
    )

    assert decision.decision.action is PolicyAction.DENY
    assert decision.denial_reason is EffectiveAuthorityDenialReason.MISSING_BASE_AUTHORITY


def test_delegation_scope_exceeds_delegator_base_authority_denies() -> None:
    membership_repo = _membership_repo()
    delegation_repo = _delegation_repo()
    authority_repo = _authority_repo()
    _seed_delegated_memberships(membership_repo)
    _seed_delegation(delegation_repo, authority_scopes=("workspace.read", "workspace.write"))
    _seed_authority(authority_repo, principal_id=_DELEGATOR, authority_scopes=("workspace.read",))

    decision = _resolver(
        membership_repo=membership_repo,
        delegation_repo=delegation_repo,
        authority_repo=authority_repo,
    ).resolve(
        _request(
            delegator_principal_id=_DELEGATOR,
            requested_authority_scopes=("workspace.write",),
            membership=_acting_membership_locator(),
            delegation=_delegation_locator(authority_scopes=("workspace.read", "workspace.write")),
        )
    )

    assert decision.decision.action is PolicyAction.DENY
    assert decision.denial_reason is EffectiveAuthorityDenialReason.INSUFFICIENT_BASE_AUTHORITY


def test_delegator_base_authority_without_delegation_scope_denies() -> None:
    membership_repo = _membership_repo()
    delegation_repo = _delegation_repo()
    authority_repo = _authority_repo()
    _seed_delegated_memberships(membership_repo)
    _seed_delegation(delegation_repo, authority_scopes=("workspace.read",))
    _seed_authority(
        authority_repo,
        principal_id=_DELEGATOR,
        authority_scopes=("workspace.read", "workspace.write"),
    )

    decision = _resolver(
        membership_repo=membership_repo,
        delegation_repo=delegation_repo,
        authority_repo=authority_repo,
    ).resolve(
        _request(
            delegator_principal_id=_DELEGATOR,
            requested_authority_scopes=("workspace.write",),
            membership=_acting_membership_locator(),
            delegation=_delegation_locator(authority_scopes=("workspace.read",)),
        )
    )

    assert decision.decision.action is PolicyAction.DENY
    assert decision.denial_reason is EffectiveAuthorityDenialReason.INSUFFICIENT_DELEGATION_SCOPE


def test_missing_delegator_base_authority_denies() -> None:
    membership_repo = _membership_repo()
    delegation_repo = _delegation_repo()
    _seed_delegated_memberships(membership_repo)
    _seed_delegation(delegation_repo)

    decision = _resolver(membership_repo=membership_repo, delegation_repo=delegation_repo).resolve(
        _request(
            delegator_principal_id=_DELEGATOR,
            membership=_acting_membership_locator(),
            delegation=_delegation_locator(),
        )
    )

    assert decision.denial_reason is EffectiveAuthorityDenialReason.MISSING_BASE_AUTHORITY


def test_delegate_own_base_authority_cannot_substitute_delegator_authority() -> None:
    membership_repo = _membership_repo()
    delegation_repo = _delegation_repo()
    authority_repo = _authority_repo()
    _seed_delegated_memberships(membership_repo)
    _seed_delegation(delegation_repo, authority_scopes=("workspace.admin",))
    _seed_authority(authority_repo, principal_id=_ACTING, authority_scopes=("workspace.admin",))

    decision = _resolver(
        membership_repo=membership_repo,
        delegation_repo=delegation_repo,
        authority_repo=authority_repo,
    ).resolve(
        _request(
            delegator_principal_id=_DELEGATOR,
            requested_authority_scopes=("workspace.admin",),
            membership=_acting_membership_locator(),
            delegation=_delegation_locator(authority_scopes=("workspace.admin",)),
        )
    )

    assert decision.decision.action is PolicyAction.DENY
    assert decision.denial_reason is EffectiveAuthorityDenialReason.MISSING_BASE_AUTHORITY


def test_resolver_does_not_require_policy_engine() -> None:
    import intergrax.collaborative_work.authority as authority_module

    assert "PolicyEngine" not in authority_module.__dict__
    decision = _resolver().resolve(_request())
    assert decision.decision.action is PolicyAction.DENY


def test_naive_clock_result_fail_closed_without_type_error() -> None:
    membership_repo = _membership_repo()
    delegation_repo = _delegation_repo()
    delegate_membership, _ = _seed_delegated_memberships(membership_repo)
    delegation = _seed_delegation(
        delegation_repo,
        valid_from=_NOW - timedelta(days=1),
        valid_until=_NOW + timedelta(days=1),
    )

    naive_now = datetime(2026, 6, 15, 12, 0)
    resolver = _resolver(
        membership_repo=membership_repo,
        delegation_repo=delegation_repo,
        clock=lambda: naive_now,
    )
    decision = resolver.resolve(
        _request(
            delegator_principal_id=_DELEGATOR,
            membership=_acting_membership_locator(),
            delegation=_delegation_locator(
                valid_from=_NOW - timedelta(days=1),
                valid_until=_NOW + timedelta(days=1),
            ),
        )
    )

    assert decision.decision.action is PolicyAction.DENY
    assert (
        decision.denial_reason
        is EffectiveAuthorityDenialReason.AUTHORITY_TEMPORAL_CONTEXT_UNAVAILABLE
    )
    assert membership_repo.get(
        tenant_id=delegate_membership.tenant_id,
        workspace_id=delegate_membership.workspace_id,
        membership_id=delegate_membership.membership_id,
    ) == delegate_membership
    assert delegation_repo.get(
        tenant_id=delegation.tenant_id,
        workspace_id=delegation.workspace_id,
        delegation_id=delegation.delegation_id,
    ) == delegation


def test_non_datetime_clock_result_fail_closed() -> None:
    membership_repo = _membership_repo()
    delegation_repo = _delegation_repo()
    _seed_delegated_memberships(membership_repo)
    _seed_delegation(
        delegation_repo,
        valid_from=_NOW - timedelta(days=1),
        valid_until=_NOW + timedelta(days=1),
    )

    resolver = _resolver(
        membership_repo=membership_repo,
        delegation_repo=delegation_repo,
        clock=lambda: "not-a-datetime",  # type: ignore[return-value]
    )
    decision = resolver.resolve(
        _request(
            delegator_principal_id=_DELEGATOR,
            membership=_acting_membership_locator(),
            delegation=_delegation_locator(
                valid_from=_NOW - timedelta(days=1),
                valid_until=_NOW + timedelta(days=1),
            ),
        )
    )

    assert decision.decision.action is PolicyAction.DENY
    assert (
        decision.denial_reason
        is EffectiveAuthorityDenialReason.AUTHORITY_TEMPORAL_CONTEXT_UNAVAILABLE
    )
