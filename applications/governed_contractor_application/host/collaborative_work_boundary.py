# © Artur Czarnecki. All rights reserved.

"""Host composition helper — canonical side-effect authorization boundary for External Work."""

from __future__ import annotations

from datetime import UTC, datetime

from external_contractor_adapter.side_effect_actions import (
    ACTION_ACCEPT_QUOTE,
    ACTION_CANCEL_EXTERNAL_WORK,
    ACTION_CREATE_EXTERNAL_WORK,
)
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
    MembershipStatus,
    OperationPolicyRequirement,
    PolicyCompositionLayer,
    PolicyLayerApplicability,
    WorkspaceMembershipRole,
)
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.runtime.policy.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationBoundary,
)

_EXTERNAL_WORK_OPERATIONS = (
    ACTION_CREATE_EXTERNAL_WORK,
    ACTION_ACCEPT_QUOTE,
    ACTION_CANCEL_EXTERNAL_WORK,
)
_DEFAULT_SCOPE = "external_work.mutate"
_NOW = datetime(2026, 6, 15, 12, 0, tzinfo=UTC)


def build_external_work_authorization_boundary(
    runtime_policy_evaluator: object,
    *,
    tenant_id: str = "tenant-a",
    workspace_id: str = "workspace-a",
    principal_id: str = "u1",
) -> MeaningfulSideEffectAuthorizationBoundary:
    """Construct a canonical boundary for governed-contractor host/demo wiring."""
    membership_repo = InMemoryWorkspaceMembershipRepository()
    authority_repo = InMemoryPrincipalAuthorityRepository()
    policy_repo = InMemoryCollaborativePolicyRepository()
    profile_repo = InMemoryCollaborativeOperationPolicyProfileRepository()

    membership_repo.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            membership_id=f"membership-{principal_id}",
            principal_id=principal_id,
            role=WorkspaceMembershipRole.MEMBER,
            status=MembershipStatus.ACTIVE,
        )
    )
    authority_repo.create(
        CreatePrincipalAuthorityGrantCommand(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            authority_grant_id=f"grant-{principal_id}",
            principal_id=principal_id,
            authority_scopes=(_DEFAULT_SCOPE,),
            status=AuthorityGrantStatus.ACTIVE,
        )
    )
    policy_repo.create(
        CreateCollaborativePolicyRuleCommand(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            policy_rule_id="workspace-allow",
            layer=PolicyCompositionLayer.WORKSPACE_POLICY,
            authority_scope=_DEFAULT_SCOPE,
            action=PolicyAction.ALLOW,
            status=CollaborativePolicyRuleStatus.ACTIVE,
        )
    )
    for operation_id in _EXTERNAL_WORK_OPERATIONS:
        profile_repo.create(
            CreateCollaborativeOperationPolicyProfileCommand(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                operation_id=operation_id,
                authority_scope=_DEFAULT_SCOPE,
                workspace_policy_applicability=PolicyLayerApplicability.REQUIRED,
                resource_policy_applicability=PolicyLayerApplicability.NOT_APPLICABLE,
                runtime_policy_applicability=PolicyLayerApplicability.REQUIRED,
                resource_requirement=OperationPolicyRequirement.NOT_APPLICABLE,
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
        runtime_policy_evaluator=runtime_policy_evaluator,  # type: ignore[arg-type]
    )
    return MeaningfulSideEffectAuthorizationBoundary(enforcement_gate=gate)
