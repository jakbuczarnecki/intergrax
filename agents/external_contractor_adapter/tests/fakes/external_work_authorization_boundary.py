# © Artur Czarnecki. All rights reserved.

"""Test helpers — seed canonical MeaningfulSideEffectAuthorizationBoundary for External Work."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol

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
    WorkspaceMembership,
    WorkspaceMembershipRole,
)
from intergrax.contracts.meaningful_side_effect import MeaningfulSideEffectRequest
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.policy.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationBoundary,
)
from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine

_EXTERNAL_WORK_OPERATIONS = (
    ACTION_CREATE_EXTERNAL_WORK,
    ACTION_ACCEPT_QUOTE,
    ACTION_CANCEL_EXTERNAL_WORK,
)
_DEFAULT_SCOPE = "external_work.mutate"
_NOW = datetime(2026, 6, 15, 12, 0, tzinfo=UTC)


class MeaningfulSideEffectPolicyEvaluator(Protocol):
    def evaluate_meaningful_side_effect(
        self,
        request: MeaningfulSideEffectRequest,
    ) -> PolicyDecision: ...


def seed_external_work_authorization_boundary(
    *,
    tenant_id: str = "tenant-a",
    workspace_id: str = "workspace-a",
    principal_id: str = "u1",
    authority_scope: str = _DEFAULT_SCOPE,
    runtime_policy_evaluator: MeaningfulSideEffectPolicyEvaluator | None = None,
    operations: tuple[str, ...] = _EXTERNAL_WORK_OPERATIONS,
    seed_resource_policy: bool = False,
    seed_workspace_policy: bool = True,
    resource_allow_scopes: tuple[str, ...] = (),
    resource_deny_scopes: tuple[str, ...] = (),
    extra_principal_grants: dict[str, tuple[str, ...]] | None = None,
) -> MeaningfulSideEffectAuthorizationBoundary:
    """Seed in-memory collaborative-work state for External Work convergence tests."""
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
            authority_scopes=(authority_scope,),
            status=AuthorityGrantStatus.ACTIVE,
        )
    )
    for extra_principal, scopes in (extra_principal_grants or {}).items():
        membership_repo.create(
            CreateWorkspaceMembershipCommand(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                membership_id=f"membership-{extra_principal}",
                principal_id=extra_principal,
                role=WorkspaceMembershipRole.MEMBER,
                status=MembershipStatus.ACTIVE,
            )
        )
        authority_repo.create(
            CreatePrincipalAuthorityGrantCommand(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                authority_grant_id=f"grant-{extra_principal}",
                principal_id=extra_principal,
                authority_scopes=scopes,
                status=AuthorityGrantStatus.ACTIVE,
            )
        )

    if seed_workspace_policy:
        policy_repo.create(
            CreateCollaborativePolicyRuleCommand(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                policy_rule_id="workspace-allow",
                layer=PolicyCompositionLayer.WORKSPACE_POLICY,
                authority_scope=authority_scope,
                action=PolicyAction.ALLOW,
                status=CollaborativePolicyRuleStatus.ACTIVE,
            )
        )

    if seed_resource_policy:
        for index, resource_scope in enumerate(resource_allow_scopes):
            policy_repo.create(
                CreateCollaborativePolicyRuleCommand(
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    policy_rule_id=f"resource-allow-{index}",
                    layer=PolicyCompositionLayer.RESOURCE_POLICY,
                    authority_scope=authority_scope,
                    resource_scope=resource_scope,
                    action=PolicyAction.ALLOW,
                    status=CollaborativePolicyRuleStatus.ACTIVE,
                )
            )
        for index, resource_scope in enumerate(resource_deny_scopes):
            policy_repo.create(
                CreateCollaborativePolicyRuleCommand(
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    policy_rule_id=f"resource-deny-{index}",
                    layer=PolicyCompositionLayer.RESOURCE_POLICY,
                    authority_scope=authority_scope,
                    resource_scope=resource_scope,
                    action=PolicyAction.DENY,
                    status=CollaborativePolicyRuleStatus.ACTIVE,
                )
            )

    for operation_id in operations:
        profile_repo.create(
            CreateCollaborativeOperationPolicyProfileCommand(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                operation_id=operation_id,
                authority_scope=authority_scope,
                workspace_policy_applicability=(
                    PolicyLayerApplicability.REQUIRED
                    if seed_workspace_policy
                    else PolicyLayerApplicability.NOT_APPLICABLE
                ),
                resource_policy_applicability=(
                    PolicyLayerApplicability.REQUIRED
                    if seed_resource_policy
                    else PolicyLayerApplicability.NOT_APPLICABLE
                ),
                runtime_policy_applicability=PolicyLayerApplicability.REQUIRED,
                resource_requirement=(
                    OperationPolicyRequirement.REQUIRED
                    if seed_resource_policy
                    else OperationPolicyRequirement.NOT_APPLICABLE
                ),
                meaningful_side_effect_requirement=OperationPolicyRequirement.REQUIRED,
                status=CollaborativeOperationPolicyProfileStatus.ACTIVE,
            )
        )

    runtime = runtime_policy_evaluator or RuntimePolicyEngine()
    gate = CollaborativeWorkEnforcementGate(
        profile_repository=profile_repo,
        authority_resolver=CollaborativeWorkAuthorityResolver(
            membership_repository=membership_repo,
            delegation_repository=InMemoryAuthorityDelegationRepository(),
            principal_authority_repository=authority_repo,
            clock=lambda: _NOW,
        ),
        policy_evaluator=CollaborativePolicyEvaluator(policy_repo),
        runtime_policy_evaluator=runtime,
    )
    return MeaningfulSideEffectAuthorizationBoundary(enforcement_gate=gate)


def allow_external_work_boundary(
    *,
    runtime_policy_evaluator: MeaningfulSideEffectPolicyEvaluator | None = None,
    **kwargs: object,
) -> MeaningfulSideEffectAuthorizationBoundary:
    """Convenience wrapper used by migrated adapter unit tests."""
    return seed_external_work_authorization_boundary(
        runtime_policy_evaluator=runtime_policy_evaluator,
        **kwargs,  # type: ignore[arg-type]
    )


def workspace_membership_locator(
    *,
    tenant_id: str,
    workspace_id: str,
    principal_id: str,
) -> WorkspaceMembership:
    return WorkspaceMembership(
        membership_id=f"locator:{tenant_id}:{workspace_id}:{principal_id}",
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        principal_id=principal_id,
        role=WorkspaceMembershipRole.MEMBER,
        status=MembershipStatus.ACTIVE,
        revision=0,
    )
