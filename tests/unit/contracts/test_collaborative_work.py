# © Artur Czarnecki. All rights reserved.

"""COLLAB-WORK-1A — collaborative identity and authority contract tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.collaborative_work import (
    SCHEMA_COLLABORATIVE_PRINCIPAL_V1,
    SCHEMA_EFFECTIVE_AUTHORITY_DECISION_V1,
    SCHEMA_EFFECTIVE_AUTHORITY_REQUEST_V1,
    SCHEMA_WORKSPACE_MEMBERSHIP_V1,
    AuthorityDelegation,
    CollaborativePrincipal,
    DelegationStatus,
    EffectiveAuthorityDenialReason,
    EffectiveAuthorityDecision,
    EffectiveAuthorityRequest,
    MembershipStatus,
    PrincipalKind,
    WorkspaceMembership,
    WorkspaceMembershipRole,
    fail_closed_effective_authority_decision,
)
from intergrax.contracts.delegation import DelegationSpec
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision

_UTC = timezone.utc


def _principal(**overrides: object) -> CollaborativePrincipal:
    payload = {
        "principal_id": "principal-1",
        "principal_kind": PrincipalKind.HUMAN,
        "tenant_id": "tenant-1",
    }
    payload.update(overrides)
    return CollaborativePrincipal.model_validate(payload)


def _membership(**overrides: object) -> WorkspaceMembership:
    payload = {
        "membership_id": "membership-1",
        "tenant_id": "tenant-1",
        "workspace_id": "workspace-1",
        "principal_id": "principal-1",
        "role": WorkspaceMembershipRole.MEMBER,
        "status": MembershipStatus.ACTIVE,
        "revision": 0,
    }
    payload.update(overrides)
    return WorkspaceMembership.model_validate(payload)


def _delegation(**overrides: object) -> AuthorityDelegation:
    payload = {
        "delegation_id": "delegation-1",
        "tenant_id": "tenant-1",
        "workspace_id": "workspace-1",
        "delegator_principal_id": "principal-human",
        "delegate_principal_id": "principal-agent",
        "authority_scopes": ("workspace.read",),
        "status": DelegationStatus.ACTIVE,
        "revision": 1,
    }
    payload.update(overrides)
    return AuthorityDelegation.model_validate(payload)


@pytest.mark.unit
@pytest.mark.parametrize(
    "kind",
    [
        PrincipalKind.HUMAN,
        PrincipalKind.AGENT,
        PrincipalKind.SERVICE,
        PrincipalKind.EXTERNAL_AGENT,
    ],
)
def test_collaborative_principal_accepts_valid_kinds(kind: PrincipalKind) -> None:
    principal = _principal(principal_kind=kind)
    assert principal.principal_kind is kind
    assert principal.schema_version == SCHEMA_COLLABORATIVE_PRINCIPAL_V1


@pytest.mark.unit
def test_collaborative_principal_is_frozen_and_rejects_extra_fields() -> None:
    principal = _principal()
    with pytest.raises(ValidationError):
        principal.principal_id = "other"
    with pytest.raises(ValidationError):
        CollaborativePrincipal.model_validate(
            {
                "principal_id": "principal-1",
                "principal_kind": "human",
                "tenant_id": "tenant-1",
                "display_name": "Alice",
            }
        )


@pytest.mark.unit
@pytest.mark.parametrize("field_name", ["principal_id", "tenant_id"])
def test_collaborative_principal_rejects_empty_ids(field_name: str) -> None:
    with pytest.raises(ValidationError):
        _principal(**{field_name: "   "})


@pytest.mark.unit
def test_collaborative_principal_serializes_stably() -> None:
    principal = _principal()
    dumped = principal.model_dump(mode="json")
    assert dumped["schema_version"] == SCHEMA_COLLABORATIVE_PRINCIPAL_V1
    assert CollaborativePrincipal.model_validate(dumped) == principal


@pytest.mark.unit
def test_workspace_membership_requires_tenant_workspace_principal() -> None:
    membership = _membership()
    assert membership.tenant_id == "tenant-1"
    assert membership.workspace_id == "workspace-1"
    assert membership.principal_id == "principal-1"
    assert membership.schema_version == SCHEMA_WORKSPACE_MEMBERSHIP_V1


@pytest.mark.unit
@pytest.mark.parametrize(
    "field_name",
    ["membership_id", "tenant_id", "workspace_id", "principal_id"],
)
def test_workspace_membership_rejects_empty_ids(field_name: str) -> None:
    with pytest.raises(ValidationError):
        _membership(**{field_name: ""})


@pytest.mark.unit
def test_workspace_membership_revision_validation_and_immutability() -> None:
    membership = _membership(revision=3)
    assert membership.revision == 3
    with pytest.raises(ValidationError):
        _membership(revision=-1)
    with pytest.raises(ValidationError):
        membership.revision = 4
    with pytest.raises(ValidationError):
        _membership(extra_field="nope")


@pytest.mark.unit
def test_authority_delegation_requires_delegator_and_delegate() -> None:
    delegation = _delegation()
    assert delegation.delegator_principal_id == "principal-human"
    assert delegation.delegate_principal_id == "principal-agent"


@pytest.mark.unit
def test_authority_delegation_rejects_self_delegation() -> None:
    with pytest.raises(ValidationError, match="delegator_principal_id must differ"):
        _delegation(
            delegator_principal_id="same-principal",
            delegate_principal_id="same-principal",
        )


@pytest.mark.unit
def test_authority_delegation_requires_non_empty_authority_scopes() -> None:
    with pytest.raises(ValidationError):
        _delegation(authority_scopes=())
    with pytest.raises(ValidationError):
        _delegation(authority_scopes=("  ",))


@pytest.mark.unit
def test_authority_delegation_rejects_invalid_validity_window() -> None:
    start = datetime(2026, 8, 11, 12, 0, tzinfo=_UTC)
    end = start - timedelta(hours=1)
    with pytest.raises(ValidationError, match="valid_until must be after valid_from"):
        _delegation(valid_from=start, valid_until=end)


@pytest.mark.unit
def test_authority_delegation_rejects_naive_validity_timestamps() -> None:
    naive = datetime(2026, 8, 11, 12, 0)
    with pytest.raises(ValidationError, match="timezone-aware"):
        _delegation(valid_from=naive)


@pytest.mark.unit
def test_authority_delegation_revision_and_immutability() -> None:
    delegation = _delegation(revision=2)
    assert delegation.revision == 2
    with pytest.raises(ValidationError):
        _delegation(revision=-1)
    with pytest.raises(ValidationError):
        delegation.revision = 3
    with pytest.raises(ValidationError):
        _delegation(extra_field="nope")


@pytest.mark.unit
def test_effective_authority_request_and_decision_invariants() -> None:
    request = EffectiveAuthorityRequest(
        tenant_id="tenant-1",
        workspace_id="workspace-1",
        acting_principal_id="principal-agent",
        requested_authority_scopes=("workspace.write",),
        delegator_principal_id="principal-human",
        membership=_membership(),
        delegation=_delegation(),
    )
    assert request.schema_version == SCHEMA_EFFECTIVE_AUTHORITY_REQUEST_V1

    decision = EffectiveAuthorityDecision(
        decision=PolicyDecision(action=PolicyAction.REQUIRE_HUMAN, reason="approval needed"),
    )
    assert decision.schema_version == SCHEMA_EFFECTIVE_AUTHORITY_DECISION_V1
    assert decision.decision.action is PolicyAction.REQUIRE_HUMAN


@pytest.mark.unit
def test_effective_authority_decision_rejects_denial_reason_on_allow() -> None:
    with pytest.raises(ValidationError, match="denial_reason must be omitted"):
        EffectiveAuthorityDecision(
            decision=PolicyDecision(action=PolicyAction.ALLOW),
            denial_reason=EffectiveAuthorityDenialReason.MISSING_MEMBERSHIP,
        )


@pytest.mark.unit
def test_fail_closed_decision_uses_policy_deny() -> None:
    decision = fail_closed_effective_authority_decision(
        reason="membership proof required",
        denial_reason=EffectiveAuthorityDenialReason.MISSING_MEMBERSHIP,
    )
    assert decision.decision.action is PolicyAction.DENY
    assert decision.denial_reason is EffectiveAuthorityDenialReason.MISSING_MEMBERSHIP


@pytest.mark.unit
def test_tenant_or_workspace_ids_alone_do_not_authorize() -> None:
    request = EffectiveAuthorityRequest(
        tenant_id="tenant-1",
        workspace_id="workspace-1",
        acting_principal_id="principal-1",
        requested_authority_scopes=("workspace.write",),
    )
    decision = fail_closed_effective_authority_decision(
        reason="tenant_id and workspace_id alone do not authorize",
        denial_reason=EffectiveAuthorityDenialReason.SCOPE_ONLY_INSUFFICIENT,
    )
    assert request.membership is None
    assert request.delegation is None
    assert decision.decision.action is PolicyAction.DENY


@pytest.mark.unit
def test_collaborative_principal_is_not_request_identity() -> None:
    assert CollaborativePrincipal is not RequestIdentity
    assert CollaborativePrincipal.model_fields.keys() != RequestIdentity.model_fields.keys()
    assert "principal_type" not in CollaborativePrincipal.model_fields
    assert "user_id" not in CollaborativePrincipal.model_fields
    assert PrincipalKind.HUMAN.value != PrincipalType.USER.value


@pytest.mark.unit
def test_authority_delegation_is_not_delegation_spec() -> None:
    assert AuthorityDelegation is not DelegationSpec
    assert AuthorityDelegation.model_fields.keys().isdisjoint(
        {"child_agent_id", "objective", "permission_scopes"}
    )
    assert "child_agent_id" in DelegationSpec.model_fields
    assert "child_agent_id" not in AuthorityDelegation.model_fields
