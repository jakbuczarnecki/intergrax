# © Artur Czarnecki. All rights reserved.

"""Production adoption boundary tests for Collaborative Work enforcement."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.collaborative_work.authority import CollaborativeWorkAuthorityResolver
from intergrax.collaborative_work.enforcement_gate import CollaborativeWorkEnforcementGate
from intergrax.collaborative_work.in_memory_repository import (
    InMemoryAuthorityDelegationRepository,
    InMemoryCollaborativeOperationPolicyProfileRepository,
    InMemoryCollaborativePolicyRepository,
    InMemoryPrincipalAuthorityRepository,
    InMemoryWorkspaceMembershipRepository,
)
from intergrax.collaborative_work.policy_source import CollaborativePolicyEvaluator
from intergrax.collaborative_work.repository import (
    CreateCollaborativeOperationPolicyProfileCommand,
    CreateCollaborativePolicyRuleCommand,
    CreatePrincipalAuthorityGrantCommand,
    CreateWorkspaceMembershipCommand,
)
from intergrax.contracts.collaborative_work import (
    AuthorityGrantStatus,
    CollaborativeOperationPolicyProfileStatus,
    CollaborativePolicyRuleStatus,
    CollaborativeWorkEnforcementRequest,
    MembershipStatus,
    OperationPolicyRequirement,
    PolicyCompositionLayer,
    PolicyLayerApplicability,
    WorkspaceMembership,
    WorkspaceMembershipRole,
)
from intergrax.contracts.meaningful_side_effect import (
    MeaningfulSideEffectKind,
    MeaningfulSideEffectRequest,
)
from intergrax.contracts.meaningful_side_effect_policy import MeaningfulSideEffectPolicyRule
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.runtime.policy.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationBoundary,
    MeaningfulSideEffectAuthorizationResult,
)
from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine

pytestmark = pytest.mark.unit

_TENANT = "tenant-a"
_WORKSPACE = "workspace-a"
_ACTING = "principal-acting"
_OPERATION = "collaborative.document.delete"
_SCOPE = "document.delete"
_RESOURCE = "document-123"
_NOW = datetime(2026, 6, 15, 12, 0, tzinfo=UTC)

def _seed_gate(
    *,
    runtime_policy: RuntimePolicyEngine | None = None,
    seed_profile: bool = True,
) -> tuple[MeaningfulSideEffectAuthorizationBoundary, WorkspaceMembership]:
    membership_repo = InMemoryWorkspaceMembershipRepository()
    authority_repo = InMemoryPrincipalAuthorityRepository()
    policy_repo = InMemoryCollaborativePolicyRepository()
    profile_repo = InMemoryCollaborativeOperationPolicyProfileRepository()

    membership = membership_repo.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            membership_id="membership-1",
            principal_id=_ACTING,
            role=WorkspaceMembershipRole.MEMBER,
            status=MembershipStatus.ACTIVE,
        )
    )
    authority_repo.create(
        CreatePrincipalAuthorityGrantCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            authority_grant_id="grant-1",
            principal_id=_ACTING,
            authority_scopes=(_SCOPE,),
            status=AuthorityGrantStatus.ACTIVE,
        )
    )
    policy_repo.create(
        CreateCollaborativePolicyRuleCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            policy_rule_id="workspace-allow",
            layer=PolicyCompositionLayer.WORKSPACE_POLICY,
            authority_scope=_SCOPE,
            action=PolicyAction.ALLOW,
            status=CollaborativePolicyRuleStatus.ACTIVE,
        )
    )
    policy_repo.create(
        CreateCollaborativePolicyRuleCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            policy_rule_id="resource-allow",
            layer=PolicyCompositionLayer.RESOURCE_POLICY,
            authority_scope=_SCOPE,
            resource_scope=_RESOURCE,
            action=PolicyAction.ALLOW,
            status=CollaborativePolicyRuleStatus.ACTIVE,
        )
    )
    if seed_profile:
        profile_repo.create(
            CreateCollaborativeOperationPolicyProfileCommand(
                tenant_id=_TENANT,
                workspace_id=_WORKSPACE,
                operation_id=_OPERATION,
                authority_scope=_SCOPE,
                workspace_policy_applicability=PolicyLayerApplicability.REQUIRED,
                resource_policy_applicability=PolicyLayerApplicability.REQUIRED,
                runtime_policy_applicability=PolicyLayerApplicability.REQUIRED,
                resource_requirement=OperationPolicyRequirement.REQUIRED,
                meaningful_side_effect_requirement=OperationPolicyRequirement.REQUIRED,
                status=CollaborativeOperationPolicyProfileStatus.ACTIVE,
            )
        )

    gate = CollaborativeWorkEnforcementGate(
        profile_repository=profile_repo,
        authority_resolver=CollaborativeWorkAuthorityResolver(
            membership_repository=membership_repo,
            delegation_repository=InMemoryAuthorityDelegationRepository(),
            principal_authority_repository=authority_repo,
            clock=lambda: _NOW,
        ),
        policy_evaluator=CollaborativePolicyEvaluator(policy_repo),
        runtime_policy_evaluator=runtime_policy
        or RuntimePolicyEngine(
            meaningful_side_effect_rules=(
                MeaningfulSideEffectPolicyRule(
                    rule_id="runtime.allow",
                    action=_OPERATION,
                    decision=PolicyAction.ALLOW,
                ),
            )
        ),
    )
    return MeaningfulSideEffectAuthorizationBoundary(enforcement_gate=gate), membership


def _enforcement_request(membership: WorkspaceMembership) -> CollaborativeWorkEnforcementRequest:
    return CollaborativeWorkEnforcementRequest(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation_id=_OPERATION,
        acting_principal_id=_ACTING,
        resource_scope=_RESOURCE,
        membership=WorkspaceMembership.model_validate(membership.model_dump()),
        meaningful_side_effect_request=MeaningfulSideEffectRequest(
            action=_OPERATION,
            kinds=(MeaningfulSideEffectKind.MUTATION,),
            task_id="task-1",
            run_id="run-1",
            principal_id=_ACTING,
            tenant_id=_TENANT,
            resource=_RESOURCE,
        ),
    )


def test_boundary_allow_permits_execution() -> None:
    boundary, membership = _seed_gate()
    executed: list[str] = []

    result = boundary.authorize_and_execute(
        _enforcement_request(membership),
        lambda: executed.append("side-effect") or "ok",
    )
    assert result == "ok"
    assert executed == ["side-effect"]


def test_boundary_deny_blocks_execution() -> None:
    boundary, membership = _seed_gate(seed_profile=False)
    executed: list[str] = []

    result = boundary.authorize_and_execute(
        _enforcement_request(membership),
        lambda: executed.append("side-effect"),
    )
    assert isinstance(result, MeaningfulSideEffectAuthorizationResult)
    assert result.permitted is False
    assert result.decision.action is PolicyAction.DENY
    assert executed == []


def test_boundary_require_human_does_not_become_allow() -> None:
    boundary, membership = _seed_gate(
        runtime_policy=RuntimePolicyEngine(
            meaningful_side_effect_rules=(
                MeaningfulSideEffectPolicyRule(
                    rule_id="runtime.hitl",
                    action=_OPERATION,
                    decision=PolicyAction.REQUIRE_HUMAN,
                ),
            )
        )
    )
    authorization = boundary.authorize(_enforcement_request(membership))
    assert authorization.permitted is False
    assert authorization.decision.action is PolicyAction.REQUIRE_HUMAN
    assert authorization.requires_governed_continuation is True


def test_boundary_preserves_operation_and_principal_identities() -> None:
    boundary, membership = _seed_gate()
    authorization = boundary.authorize(_enforcement_request(membership))
    assert authorization.enforcement_result.operation_id == _OPERATION
    assert authorization.enforcement_result.authority_scope == _SCOPE


def test_missing_authoritative_state_fails_closed() -> None:
    empty_boundary = MeaningfulSideEffectAuthorizationBoundary(
        enforcement_gate=CollaborativeWorkEnforcementGate(
            profile_repository=InMemoryCollaborativeOperationPolicyProfileRepository(),
            authority_resolver=CollaborativeWorkAuthorityResolver(
                membership_repository=InMemoryWorkspaceMembershipRepository(),
                delegation_repository=InMemoryAuthorityDelegationRepository(),
                principal_authority_repository=InMemoryPrincipalAuthorityRepository(),
                clock=lambda: _NOW,
            ),
            policy_evaluator=CollaborativePolicyEvaluator(InMemoryCollaborativePolicyRepository()),
            runtime_policy_evaluator=RuntimePolicyEngine(),
        )
    )
    fake_membership = WorkspaceMembership(
        membership_id="membership-1",
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        principal_id=_ACTING,
        role=WorkspaceMembershipRole.MEMBER,
        status=MembershipStatus.ACTIVE,
        revision=0,
    )
    authorization = empty_boundary.authorize(_enforcement_request(fake_membership))
    assert authorization.permitted is False
    assert authorization.decision.action is PolicyAction.DENY
