# © Artur Czarnecki. All rights reserved.

"""COLLAB-WORK-1C — authoritative effective-authority resolver tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from intergrax.collaborative_work.authority import CollaborativeWorkAuthorityResolver
from intergrax.collaborative_work.in_memory_repository import (
    InMemoryAuthorityDelegationRepository,
    InMemoryWorkspaceMembershipRepository,
)
from intergrax.collaborative_work.repository import (
    CreateAuthorityDelegationCommand,
    CreateWorkspaceMembershipCommand,
    UpdateWorkspaceMembershipCommand,
    WorkspaceMembershipScopeKey,
)
from intergrax.contracts.collaborative_work import (
    AuthorityDelegation,
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


def _resolver(
    *,
    membership_repo: InMemoryWorkspaceMembershipRepository | None = None,
    delegation_repo: InMemoryAuthorityDelegationRepository | None = None,
    now: datetime = _NOW,
) -> CollaborativeWorkAuthorityResolver:
    return CollaborativeWorkAuthorityResolver(
        membership_repository=membership_repo or _membership_repo(),
        delegation_repository=delegation_repo or _delegation_repo(),
        clock=lambda: now,
    )


def _seed_membership(
    repo: InMemoryWorkspaceMembershipRepository,
    *,
    principal_id: str = _ACTING,
    status: MembershipStatus = MembershipStatus.ACTIVE,
) -> WorkspaceMembership:
    return repo.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            membership_id="membership-1",
            principal_id=principal_id,
            role=WorkspaceMembershipRole.MEMBER,
            status=status,
        )
    )


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
    assert decision.denial_reason is EffectiveAuthorityDenialReason.SCOPE_ONLY_INSUFFICIENT


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
    _seed_membership(membership_repo)
    _seed_delegation(delegation_repo, authority_scopes=("workspace.read",))

    resolver = _resolver(membership_repo=membership_repo, delegation_repo=delegation_repo)
    decision = resolver.resolve(
        _request(
            delegator_principal_id=_DELEGATOR,
            requested_authority_scopes=("workspace.write",),
            membership=_membership_locator(),
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
    _seed_membership(membership_repo)

    decision = _resolver(membership_repo=membership_repo).resolve(
        _request(
            delegator_principal_id=_DELEGATOR,
            membership=_membership_locator(),
        )
    )
    assert decision.denial_reason is EffectiveAuthorityDenialReason.MISSING_DELEGATION


def test_missing_authoritative_delegation_record() -> None:
    membership_repo = _membership_repo()
    _seed_membership(membership_repo)

    decision = _resolver(membership_repo=membership_repo).resolve(
        _request(
            delegator_principal_id=_DELEGATOR,
            membership=_membership_locator(),
            delegation=_delegation_locator(),
        )
    )
    assert decision.denial_reason is EffectiveAuthorityDenialReason.MISSING_DELEGATION


def test_inactive_delegation_denied() -> None:
    membership_repo = _membership_repo()
    delegation_repo = _delegation_repo()
    _seed_membership(membership_repo)
    _seed_delegation(delegation_repo, status=DelegationStatus.REVOKED)

    decision = _resolver(membership_repo=membership_repo, delegation_repo=delegation_repo).resolve(
        _request(
            delegator_principal_id=_DELEGATOR,
            membership=_membership_locator(),
            delegation=_delegation_locator(status=DelegationStatus.ACTIVE),
        )
    )
    assert decision.denial_reason is EffectiveAuthorityDenialReason.DELEGATION_NOT_ACTIVE


def test_delegation_not_yet_valid() -> None:
    membership_repo = _membership_repo()
    delegation_repo = _delegation_repo()
    _seed_membership(membership_repo)
    _seed_delegation(
        delegation_repo,
        valid_from=_NOW + timedelta(days=1),
        valid_until=_NOW + timedelta(days=30),
    )

    decision = _resolver(membership_repo=membership_repo, delegation_repo=delegation_repo).resolve(
        _request(
            delegator_principal_id=_DELEGATOR,
            membership=_membership_locator(),
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
    _seed_membership(membership_repo)
    _seed_delegation(
        delegation_repo,
        valid_from=_NOW - timedelta(days=30),
        valid_until=_NOW - timedelta(seconds=1),
    )

    decision = _resolver(membership_repo=membership_repo, delegation_repo=delegation_repo).resolve(
        _request(
            delegator_principal_id=_DELEGATOR,
            membership=_membership_locator(),
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
    _seed_membership(membership_repo)
    _seed_delegation(delegation_repo, authority_scopes=("workspace.read",))

    decision = _resolver(membership_repo=membership_repo, delegation_repo=delegation_repo).resolve(
        _request(
            delegator_principal_id=_DELEGATOR,
            requested_authority_scopes=("workspace.write",),
            membership=_membership_locator(),
            delegation=_delegation_locator(authority_scopes=("workspace.read", "workspace.write")),
        )
    )
    assert decision.denial_reason is EffectiveAuthorityDenialReason.INSUFFICIENT_DELEGATION_SCOPE


def test_delegate_mismatch_cannot_authorize() -> None:
    membership_repo = _membership_repo()
    delegation_repo = _delegation_repo()
    _seed_membership(membership_repo)
    _seed_delegation(delegation_repo, delegate_principal_id="other-delegate")

    decision = _resolver(membership_repo=membership_repo, delegation_repo=delegation_repo).resolve(
        _request(
            delegator_principal_id=_DELEGATOR,
            membership=_membership_locator(),
            delegation=_delegation_locator(delegate_principal_id=_ACTING),
        )
    )
    assert decision.denial_reason is EffectiveAuthorityDenialReason.MISSING_DELEGATION


def test_delegator_mismatch_cannot_authorize() -> None:
    membership_repo = _membership_repo()
    delegation_repo = _delegation_repo()
    _seed_membership(membership_repo)
    _seed_delegation(delegation_repo, delegator_principal_id="other-delegator")

    decision = _resolver(membership_repo=membership_repo, delegation_repo=delegation_repo).resolve(
        _request(
            delegator_principal_id=_DELEGATOR,
            membership=_membership_locator(),
            delegation=_delegation_locator(delegator_principal_id=_DELEGATOR),
        )
    )
    assert decision.denial_reason is EffectiveAuthorityDenialReason.MISSING_DELEGATION


def test_resource_scope_mismatch_denied() -> None:
    membership_repo = _membership_repo()
    delegation_repo = _delegation_repo()
    _seed_membership(membership_repo)
    _seed_delegation(delegation_repo, resource_scope="artifact-a")

    decision = _resolver(membership_repo=membership_repo, delegation_repo=delegation_repo).resolve(
        _request(
            delegator_principal_id=_DELEGATOR,
            resource_scope="artifact-b",
            membership=_membership_locator(),
            delegation=_delegation_locator(resource_scope="artifact-a"),
        )
    )
    assert decision.denial_reason is EffectiveAuthorityDenialReason.INSUFFICIENT_DELEGATION_SCOPE


def test_resource_limited_delegation_requires_request_resource_scope() -> None:
    membership_repo = _membership_repo()
    delegation_repo = _delegation_repo()
    _seed_membership(membership_repo)
    _seed_delegation(delegation_repo, resource_scope="artifact-a")

    decision = _resolver(membership_repo=membership_repo, delegation_repo=delegation_repo).resolve(
        _request(
            delegator_principal_id=_DELEGATOR,
            membership=_membership_locator(),
            delegation=_delegation_locator(resource_scope="artifact-a"),
        )
    )
    assert decision.denial_reason is EffectiveAuthorityDenialReason.INSUFFICIENT_DELEGATION_SCOPE


def test_valid_delegated_evidence_reaches_maximum_safe_result() -> None:
    membership_repo = _membership_repo()
    delegation_repo = _delegation_repo()
    _seed_membership(membership_repo)
    _seed_delegation(
        delegation_repo,
        resource_scope="artifact-a",
        valid_from=_NOW - timedelta(days=1),
        valid_until=_NOW + timedelta(days=1),
    )

    decision = _resolver(membership_repo=membership_repo, delegation_repo=delegation_repo).resolve(
        _request(
            delegator_principal_id=_DELEGATOR,
            resource_scope="artifact-a",
            membership=_membership_locator(),
            delegation=_delegation_locator(
                resource_scope="artifact-a",
                valid_from=_NOW - timedelta(days=1),
                valid_until=_NOW + timedelta(days=1),
            ),
        )
    )

    assert decision.decision.action is PolicyAction.DENY
    assert decision.denial_reason is EffectiveAuthorityDenialReason.SCOPE_ONLY_INSUFFICIENT
    assert (
        decision.decision.policy_rule_id
        == "collaborative_work.effective_authority.scope_only_insufficient"
    )


def test_direct_acting_active_membership_still_scope_only_insufficient() -> None:
    membership_repo = _membership_repo()
    _seed_membership(membership_repo)

    decision = _resolver(membership_repo=membership_repo).resolve(
        _request(membership=_membership_locator())
    )

    assert decision.decision.action is PolicyAction.DENY
    assert decision.denial_reason is EffectiveAuthorityDenialReason.SCOPE_ONLY_INSUFFICIENT


def test_delegation_scopes_alone_do_not_manufacture_delegator_authority() -> None:
    membership_repo = _membership_repo()
    delegation_repo = _delegation_repo()
    _seed_membership(membership_repo)
    _seed_delegation(delegation_repo, authority_scopes=("workspace.admin",))

    decision = _resolver(membership_repo=membership_repo, delegation_repo=delegation_repo).resolve(
        _request(
            delegator_principal_id=_DELEGATOR,
            requested_authority_scopes=("workspace.admin",),
            membership=_membership_locator(),
            delegation=_delegation_locator(authority_scopes=("workspace.admin",)),
        )
    )

    assert decision.decision.action is PolicyAction.DENY
    assert decision.denial_reason is EffectiveAuthorityDenialReason.SCOPE_ONLY_INSUFFICIENT


def test_resolver_does_not_require_policy_engine() -> None:
    import intergrax.collaborative_work.authority as authority_module

    assert "PolicyEngine" not in authority_module.__dict__
    decision = _resolver().resolve(_request())
    assert decision.decision.action is PolicyAction.DENY
