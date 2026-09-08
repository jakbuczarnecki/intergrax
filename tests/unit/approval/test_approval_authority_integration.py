# © Artur Czarnecki. All rights reserved.

"""MP-4D — Approval authority integration tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

from intergrax.approval.errors import ApprovalAuthorizationDenied
from intergrax.approval.service import (
    TRUSTED_OPERATION_APPROVAL_ACTION,
    TRUSTED_OPERATION_APPROVAL_CREATE,
    ApprovalService,
)
from intergrax.collaborative_work.authority import CollaborativeWorkAuthorityResolver
from intergrax.collaborative_work.enforcement_gate import (
    CollaborativeWorkEnforcementGate,
)
from intergrax.collaborative_work.in_memory_repository import (
    InMemoryAuthorityDelegationRepository,
    InMemoryCollaborativeOperationPolicyProfileRepository,
    InMemoryCollaborativePolicyRepository,
    InMemoryPrincipalAuthorityRepository,
    InMemoryWorkspaceMembershipRepository,
)
from intergrax.collaborative_work.policy_source import CollaborativePolicyEvaluator
from intergrax.collaborative_work.repository import (
    CreateAuthorityDelegationCommand,
    CreateCollaborativeOperationPolicyProfileCommand,
    CreatePrincipalAuthorityGrantCommand,
    CreateWorkspaceMembershipCommand,
)
from intergrax.contracts.approval import (
    ApprovalLifecycleState,
    ApprovalRequest,
    CreateApprovalRequest,
    ExecuteHumanApprovalActionRequest,
    HumanApprovalActionType,
    approval_resource_scope,
    mint_approval_id,
)
from intergrax.contracts.collaborative_work import (
    AuthorityDelegation,
    CollaborativeOperationPolicyProfileStatus,
    DelegationStatus,
    MembershipResolutionMode,
    MembershipStatus,
    OperationPolicyRequirement,
    PolicyLayerApplicability,
    WorkspaceMembership,
    WorkspaceMembershipRole,
)
from intergrax.contracts.decision import mint_decision_id
from intergrax.contracts.meaningful_side_effect import MeaningfulSideEffectRequest
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision

pytestmark = pytest.mark.unit

_TENANT = "tenant-a"
_WORKSPACE = "workspace-a"
_ACTING = "principal-acting"
_DELEGATOR = "principal-delegator"
_AUTHORITY_SCOPE = "approval.manage"
_DECISION_ID = str(mint_decision_id())
_APPROVAL_ID = str(mint_approval_id())
_NOW = datetime(2026, 9, 8, 12, 0, tzinfo=UTC)
_LATER = _NOW + timedelta(minutes=5)


class _UnusedRuntimeEvaluator:
    def evaluate_meaningful_side_effect(
        self,
        request: MeaningfulSideEffectRequest,
    ) -> PolicyDecision:
        _ = request
        return PolicyDecision(
            action=PolicyAction.DENY,
            reason="runtime evaluator must not run for approval mutations",
            policy_rule_id="test.unexpected_runtime",
        )


@dataclass(frozen=True, slots=True)
class _ServiceFixture:
    service: ApprovalService
    membership_repo: InMemoryWorkspaceMembershipRepository
    authority_repo: InMemoryPrincipalAuthorityRepository
    profile_repo: InMemoryCollaborativeOperationPolicyProfileRepository
    delegation_repo: InMemoryAuthorityDelegationRepository


def _profile_command(
    *, operation_id: str, **overrides: object
) -> CreateCollaborativeOperationPolicyProfileCommand:
    payload = {
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "operation_id": operation_id,
        "authority_scope": _AUTHORITY_SCOPE,
        "workspace_policy_applicability": PolicyLayerApplicability.NOT_APPLICABLE,
        "resource_policy_applicability": PolicyLayerApplicability.NOT_APPLICABLE,
        "runtime_policy_applicability": PolicyLayerApplicability.NOT_APPLICABLE,
        "resource_requirement": OperationPolicyRequirement.NOT_APPLICABLE,
        "meaningful_side_effect_requirement": OperationPolicyRequirement.NOT_APPLICABLE,
        "status": CollaborativeOperationPolicyProfileStatus.ACTIVE,
    }
    payload.update(overrides)
    return CreateCollaborativeOperationPolicyProfileCommand(**payload)


def _seed_approval_profiles(
    profile_repo: InMemoryCollaborativeOperationPolicyProfileRepository,
) -> None:
    for operation_id in (
        TRUSTED_OPERATION_APPROVAL_CREATE,
        TRUSTED_OPERATION_APPROVAL_ACTION,
    ):
        profile_repo.create(_profile_command(operation_id=operation_id))


def _seed_membership(
    repo: InMemoryWorkspaceMembershipRepository,
    *,
    principal_id: str = _ACTING,
    membership_id: str = "membership-acting",
) -> WorkspaceMembership:
    return repo.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            membership_id=membership_id,
            principal_id=principal_id,
            role=WorkspaceMembershipRole.MEMBER,
            status=MembershipStatus.ACTIVE,
        )
    )


def _seed_authority(
    repo: InMemoryPrincipalAuthorityRepository,
    *,
    principal_id: str = _ACTING,
    authority_scopes: tuple[str, ...] = (_AUTHORITY_SCOPE,),
    grant_id: str = "authority-grant-acting",
) -> object:
    return repo.create(
        CreatePrincipalAuthorityGrantCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            authority_grant_id=grant_id,
            principal_id=principal_id,
            authority_scopes=authority_scopes,
        )
    )


def _service_fixture(
    *,
    seed_membership: bool = True,
    seed_authority: bool = True,
    seed_profiles: bool = True,
) -> _ServiceFixture:
    membership_repo = InMemoryWorkspaceMembershipRepository()
    authority_repo = InMemoryPrincipalAuthorityRepository()
    policy_repo = InMemoryCollaborativePolicyRepository()
    profile_repo = InMemoryCollaborativeOperationPolicyProfileRepository()
    delegation_repo = InMemoryAuthorityDelegationRepository()

    if seed_membership:
        _seed_membership(membership_repo)
    if seed_authority:
        _seed_authority(authority_repo)
    if seed_profiles:
        _seed_approval_profiles(profile_repo)

    gate = CollaborativeWorkEnforcementGate(
        profile_repository=profile_repo,
        authority_resolver=CollaborativeWorkAuthorityResolver(
            membership_repository=membership_repo,
            delegation_repository=delegation_repo,
            principal_authority_repository=authority_repo,
            clock=lambda: _NOW,
        ),
        policy_evaluator=CollaborativePolicyEvaluator(policy_repo),
        runtime_policy_evaluator=_UnusedRuntimeEvaluator(),
    )
    service = ApprovalService(enforcement_gate=gate, clock=lambda: _LATER)
    return _ServiceFixture(
        service=service,
        membership_repo=membership_repo,
        authority_repo=authority_repo,
        profile_repo=profile_repo,
        delegation_repo=delegation_repo,
    )


def _create_approval_request(**overrides: object) -> CreateApprovalRequest:
    payload = {
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "decision_id": _DECISION_ID,
        "approval_id": _APPROVAL_ID,
        "acting_principal_id": _ACTING,
        "membership_resolution_mode": MembershipResolutionMode.CANONICAL_PRINCIPAL,
    }
    payload.update(overrides)
    return CreateApprovalRequest.model_validate(payload)


def _existing_approval(**overrides: object) -> ApprovalRequest:
    payload = {
        "approval_id": _APPROVAL_ID,
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "decision_id": _DECISION_ID,
        "requested_by_principal_id": _ACTING,
        "requested_at": _NOW,
        "lifecycle_state": ApprovalLifecycleState.REQUESTED,
    }
    payload.update(overrides)
    return ApprovalRequest.model_validate(payload)


def _action_request(**overrides: object) -> ExecuteHumanApprovalActionRequest:
    payload = {
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "approval_id": _APPROVAL_ID,
        "acting_principal_id": _ACTING,
        "action": HumanApprovalActionType.START_REVIEW,
        "membership_resolution_mode": MembershipResolutionMode.CANONICAL_PRINCIPAL,
    }
    payload.update(overrides)
    return ExecuteHumanApprovalActionRequest.model_validate(payload)


def test_create_approval_request_allow() -> None:
    fixture = _service_fixture()
    created = fixture.service.create_approval_request(_create_approval_request())
    assert created.approval_id == _APPROVAL_ID
    assert created.decision_id == _DECISION_ID
    assert created.requested_by_principal_id == _ACTING
    assert created.lifecycle_state is ApprovalLifecycleState.REQUESTED
    assert created.requested_at == _LATER


def test_create_approval_request_deny_raises_without_domain_object() -> None:
    fixture = _service_fixture(seed_authority=False)
    with pytest.raises(ApprovalAuthorizationDenied) as exc:
        fixture.service.create_approval_request(_create_approval_request())
    assert (
        exc.value.enforcement_result.operation_id == TRUSTED_OPERATION_APPROVAL_CREATE
    )
    assert exc.value.enforcement_result.composition.decision.action is PolicyAction.DENY


def test_create_approval_request_missing_membership_denied() -> None:
    fixture = _service_fixture(seed_membership=False)
    with pytest.raises(ApprovalAuthorizationDenied):
        fixture.service.create_approval_request(_create_approval_request())


def test_execute_human_approval_action_allow() -> None:
    fixture = _service_fixture()
    action = fixture.service.execute_human_approval_action(
        existing_approval=_existing_approval(),
        request=_action_request(),
    )
    assert action.approval_id == _APPROVAL_ID
    assert action.acting_principal_id == _ACTING
    assert action.action is HumanApprovalActionType.START_REVIEW
    assert action.timestamp == _LATER


def test_execute_human_approval_action_deny_without_action_fact() -> None:
    fixture = _service_fixture(seed_authority=False)
    with pytest.raises(ApprovalAuthorizationDenied) as exc:
        fixture.service.execute_human_approval_action(
            existing_approval=_existing_approval(),
            request=_action_request(),
        )
    assert (
        exc.value.enforcement_result.operation_id == TRUSTED_OPERATION_APPROVAL_ACTION
    )
    assert exc.value.enforcement_result.composition.decision.action is PolicyAction.DENY


def test_delegated_human_approval_action_preserves_acting_principal() -> None:
    fixture = _service_fixture(seed_membership=False, seed_authority=False)
    _seed_membership(fixture.membership_repo, principal_id=_ACTING)
    _seed_membership(
        fixture.membership_repo,
        principal_id=_DELEGATOR,
        membership_id="membership-delegator",
    )
    resource_scope = approval_resource_scope(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        approval_id=_APPROVAL_ID,
    )
    fixture.delegation_repo.create(
        CreateAuthorityDelegationCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            delegation_id="delegation-1",
            delegator_principal_id=_DELEGATOR,
            delegate_principal_id=_ACTING,
            authority_scopes=(_AUTHORITY_SCOPE,),
            resource_scope=resource_scope,
        )
    )
    _seed_authority(
        fixture.authority_repo,
        principal_id=_DELEGATOR,
        grant_id="authority-grant-delegator",
    )
    delegation = AuthorityDelegation.model_validate(
        {
            "delegation_id": "delegation-1",
            "tenant_id": _TENANT,
            "workspace_id": _WORKSPACE,
            "delegator_principal_id": _DELEGATOR,
            "delegate_principal_id": _ACTING,
            "authority_scopes": (_AUTHORITY_SCOPE,),
            "resource_scope": resource_scope,
            "status": DelegationStatus.ACTIVE,
            "revision": 0,
        }
    )
    membership = WorkspaceMembership.model_validate(
        {
            "membership_id": "membership-acting",
            "tenant_id": _TENANT,
            "workspace_id": _WORKSPACE,
            "principal_id": _ACTING,
            "role": WorkspaceMembershipRole.MEMBER,
            "status": MembershipStatus.ACTIVE,
            "revision": 0,
        }
    )
    action = fixture.service.execute_human_approval_action(
        existing_approval=_existing_approval(),
        request=_action_request(
            membership_resolution_mode=MembershipResolutionMode.LOCATOR,
            membership=membership,
            delegator_principal_id=_DELEGATOR,
            delegation=delegation,
        ),
    )
    assert action.acting_principal_id == _ACTING


def test_create_approval_request_no_mutation_on_deny() -> None:
    fixture = _service_fixture(seed_authority=False)
    mutations: list[ApprovalRequest] = []

    def _capture_create(request: CreateApprovalRequest) -> ApprovalRequest:
        created = fixture.service.create_approval_request(request)
        mutations.append(created)
        return created

    with pytest.raises(ApprovalAuthorizationDenied):
        _capture_create(_create_approval_request())
    assert mutations == []
