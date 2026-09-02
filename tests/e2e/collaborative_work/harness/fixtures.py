# © Artur Czarnecki. All rights reserved.

"""Authoritative Collaborative Work state seeding for E2E scenarios."""

from __future__ import annotations


from intergrax.collaborative_work.persistence import CollaborativeWorkRepositories
from intergrax.collaborative_work.repository import (
    CreateAuthorityDelegationCommand,
    CreateCollaborativeOperationPolicyProfileCommand,
    CreateCollaborativePolicyRuleCommand,
    CreatePrincipalAuthorityGrantCommand,
    CreateWorkspaceMembershipCommand,
)
from intergrax.contracts.collaborative_work import (
    AuthorityDelegation,
    AuthorityGrantStatus,
    CollaborativeOperationPolicyProfileStatus,
    CollaborativePolicyRuleStatus,
    DelegationStatus,
    MembershipStatus,
    OperationPolicyRequirement,
    PolicyCompositionLayer,
    PolicyLayerApplicability,
    WorkspaceMembership,
    WorkspaceMembershipRole,
)
from intergrax.contracts.runtime_policy import PolicyAction

from tests.e2e.collaborative_work.harness.constants import (
    AUTHORITY_SCOPE_MUTATE,
    AUTHORITY_SCOPE_READ,
    FIXED_NOW,
    OPERATION_MUTATE,
    PRINCIPAL_ALICE,
    PRINCIPAL_DELEGATE,
    PRINCIPAL_DELEGATOR,
    RESOURCE_A,
    TENANT_A,
    TENANT_B,
    WORKSPACE_A,
    WORKSPACE_B,
)


def seed_membership(
    bundle: CollaborativeWorkRepositories,
    *,
    tenant_id: str = TENANT_A,
    workspace_id: str = WORKSPACE_A,
    membership_id: str,
    principal_id: str,
    status: MembershipStatus = MembershipStatus.ACTIVE,
) -> WorkspaceMembership:
    return bundle.membership.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            membership_id=membership_id,
            principal_id=principal_id,
            role=WorkspaceMembershipRole.MEMBER,
            status=status,
        )
    )


def seed_authority_grant(
    bundle: CollaborativeWorkRepositories,
    *,
    tenant_id: str = TENANT_A,
    workspace_id: str = WORKSPACE_A,
    grant_id: str,
    principal_id: str,
    authority_scopes: tuple[str, ...] = (AUTHORITY_SCOPE_MUTATE,),
    status: AuthorityGrantStatus = AuthorityGrantStatus.ACTIVE,
) -> None:
    bundle.principal_authority.create(
        CreatePrincipalAuthorityGrantCommand(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            authority_grant_id=grant_id,
            principal_id=principal_id,
            authority_scopes=authority_scopes,
            status=status,
        )
    )


def seed_workspace_policy(
    bundle: CollaborativeWorkRepositories,
    *,
    tenant_id: str = TENANT_A,
    workspace_id: str = WORKSPACE_A,
    policy_rule_id: str,
    authority_scope: str = AUTHORITY_SCOPE_MUTATE,
    action: PolicyAction = PolicyAction.ALLOW,
) -> None:
    bundle.policy.create(
        CreateCollaborativePolicyRuleCommand(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            policy_rule_id=policy_rule_id,
            layer=PolicyCompositionLayer.WORKSPACE_POLICY,
            authority_scope=authority_scope,
            action=action,
            status=CollaborativePolicyRuleStatus.ACTIVE,
        )
    )


def seed_resource_policy(
    bundle: CollaborativeWorkRepositories,
    *,
    tenant_id: str = TENANT_A,
    workspace_id: str = WORKSPACE_A,
    policy_rule_id: str,
    resource_scope: str,
    authority_scope: str = AUTHORITY_SCOPE_MUTATE,
    action: PolicyAction = PolicyAction.ALLOW,
) -> None:
    bundle.policy.create(
        CreateCollaborativePolicyRuleCommand(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            policy_rule_id=policy_rule_id,
            layer=PolicyCompositionLayer.RESOURCE_POLICY,
            authority_scope=authority_scope,
            resource_scope=resource_scope,
            action=action,
            status=CollaborativePolicyRuleStatus.ACTIVE,
        )
    )


def seed_operation_profile(
    bundle: CollaborativeWorkRepositories,
    *,
    tenant_id: str = TENANT_A,
    workspace_id: str = WORKSPACE_A,
    operation_id: str = OPERATION_MUTATE,
    authority_scope: str = AUTHORITY_SCOPE_MUTATE,
    workspace_required: bool = True,
    resource_required: bool = True,
    runtime_required: bool = True,
) -> None:
    bundle.operation_profile.create(
        CreateCollaborativeOperationPolicyProfileCommand(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            operation_id=operation_id,
            authority_scope=authority_scope,
            workspace_policy_applicability=(
                PolicyLayerApplicability.REQUIRED
                if workspace_required
                else PolicyLayerApplicability.NOT_APPLICABLE
            ),
            resource_policy_applicability=(
                PolicyLayerApplicability.REQUIRED
                if resource_required
                else PolicyLayerApplicability.NOT_APPLICABLE
            ),
            runtime_policy_applicability=(
                PolicyLayerApplicability.REQUIRED
                if runtime_required
                else PolicyLayerApplicability.NOT_APPLICABLE
            ),
            resource_requirement=(
                OperationPolicyRequirement.REQUIRED
                if resource_required
                else OperationPolicyRequirement.NOT_APPLICABLE
            ),
            meaningful_side_effect_requirement=(
                OperationPolicyRequirement.REQUIRED
                if runtime_required
                else OperationPolicyRequirement.NOT_APPLICABLE
            ),
            status=CollaborativeOperationPolicyProfileStatus.ACTIVE,
        )
    )


def seed_direct_allow_fixture(bundle: CollaborativeWorkRepositories) -> None:
    seed_membership(
        bundle,
        membership_id="membership-alice",
        principal_id=PRINCIPAL_ALICE,
    )
    seed_authority_grant(
        bundle,
        grant_id="grant-alice",
        principal_id=PRINCIPAL_ALICE,
    )
    seed_workspace_policy(bundle, policy_rule_id="workspace-allow")
    seed_resource_policy(
        bundle,
        policy_rule_id="resource-allow-a",
        resource_scope=RESOURCE_A,
    )
    seed_operation_profile(bundle)


def seed_missing_authority_fixture(bundle: CollaborativeWorkRepositories) -> None:
    seed_membership(
        bundle,
        membership_id="membership-alice",
        principal_id=PRINCIPAL_ALICE,
    )
    seed_workspace_policy(bundle, policy_rule_id="workspace-allow")
    seed_resource_policy(
        bundle,
        policy_rule_id="resource-allow-a",
        resource_scope=RESOURCE_A,
    )
    seed_operation_profile(bundle)


def seed_inactive_membership_fixture(bundle: CollaborativeWorkRepositories) -> None:
    seed_membership(
        bundle,
        membership_id="membership-alice",
        principal_id=PRINCIPAL_ALICE,
        status=MembershipStatus.SUSPENDED,
    )
    seed_authority_grant(
        bundle,
        grant_id="grant-alice",
        principal_id=PRINCIPAL_ALICE,
    )
    seed_workspace_policy(bundle, policy_rule_id="workspace-allow")
    seed_resource_policy(
        bundle,
        policy_rule_id="resource-allow-a",
        resource_scope=RESOURCE_A,
    )
    seed_operation_profile(bundle)


def seed_valid_delegation_fixture(
    bundle: CollaborativeWorkRepositories,
) -> AuthorityDelegation:
    seed_membership(
        bundle,
        membership_id="membership-delegator",
        principal_id=PRINCIPAL_DELEGATOR,
    )
    seed_membership(
        bundle,
        membership_id="membership-delegate",
        principal_id=PRINCIPAL_DELEGATE,
    )
    seed_authority_grant(
        bundle,
        grant_id="grant-delegator",
        principal_id=PRINCIPAL_DELEGATOR,
    )
    seed_workspace_policy(bundle, policy_rule_id="workspace-allow")
    seed_resource_policy(
        bundle,
        policy_rule_id="resource-allow-a",
        resource_scope=RESOURCE_A,
    )
    seed_operation_profile(bundle)
    return bundle.delegation.create(
        CreateAuthorityDelegationCommand(
            tenant_id=TENANT_A,
            workspace_id=WORKSPACE_A,
            delegation_id="delegation-valid",
            delegator_principal_id=PRINCIPAL_DELEGATOR,
            delegate_principal_id=PRINCIPAL_DELEGATE,
            authority_scopes=(AUTHORITY_SCOPE_MUTATE,),
            resource_scope=RESOURCE_A,
            valid_from=FIXED_NOW,
            status=DelegationStatus.ACTIVE,
        )
    )


def seed_delegation_amplification_fixture(
    bundle: CollaborativeWorkRepositories,
) -> AuthorityDelegation:
    seed_membership(
        bundle,
        membership_id="membership-delegator",
        principal_id=PRINCIPAL_DELEGATOR,
    )
    seed_membership(
        bundle,
        membership_id="membership-delegate",
        principal_id=PRINCIPAL_DELEGATE,
    )
    seed_authority_grant(
        bundle,
        grant_id="grant-delegator-read",
        principal_id=PRINCIPAL_DELEGATOR,
        authority_scopes=(AUTHORITY_SCOPE_READ,),
    )
    seed_workspace_policy(
        bundle,
        policy_rule_id="workspace-allow",
        authority_scope=AUTHORITY_SCOPE_MUTATE,
    )
    seed_resource_policy(
        bundle,
        policy_rule_id="resource-allow-a",
        resource_scope=RESOURCE_A,
        authority_scope=AUTHORITY_SCOPE_MUTATE,
    )
    seed_operation_profile(bundle)
    return bundle.delegation.create(
        CreateAuthorityDelegationCommand(
            tenant_id=TENANT_A,
            workspace_id=WORKSPACE_A,
            delegation_id="delegation-amplified",
            delegator_principal_id=PRINCIPAL_DELEGATOR,
            delegate_principal_id=PRINCIPAL_DELEGATE,
            authority_scopes=(AUTHORITY_SCOPE_MUTATE,),
            resource_scope=RESOURCE_A,
            valid_from=FIXED_NOW,
            status=DelegationStatus.ACTIVE,
        )
    )


def seed_resource_mismatch_fixture(bundle: CollaborativeWorkRepositories) -> None:
    seed_membership(
        bundle,
        membership_id="membership-alice",
        principal_id=PRINCIPAL_ALICE,
    )
    seed_authority_grant(
        bundle,
        grant_id="grant-alice",
        principal_id=PRINCIPAL_ALICE,
    )
    seed_workspace_policy(bundle, policy_rule_id="workspace-allow")
    seed_resource_policy(
        bundle,
        policy_rule_id="resource-allow-a",
        resource_scope=RESOURCE_A,
    )
    seed_operation_profile(bundle)


def seed_tenant_isolation_fixture(bundle: CollaborativeWorkRepositories) -> None:
    seed_membership(
        bundle,
        tenant_id=TENANT_A,
        workspace_id=WORKSPACE_A,
        membership_id="membership-a",
        principal_id=PRINCIPAL_ALICE,
    )
    seed_authority_grant(
        bundle,
        tenant_id=TENANT_A,
        workspace_id=WORKSPACE_A,
        grant_id="grant-a",
        principal_id=PRINCIPAL_ALICE,
    )
    seed_workspace_policy(
        bundle,
        tenant_id=TENANT_A,
        workspace_id=WORKSPACE_A,
        policy_rule_id="workspace-allow-a",
    )
    seed_resource_policy(
        bundle,
        tenant_id=TENANT_A,
        workspace_id=WORKSPACE_A,
        policy_rule_id="resource-allow-a",
        resource_scope=RESOURCE_A,
    )
    seed_operation_profile(
        bundle,
        tenant_id=TENANT_A,
        workspace_id=WORKSPACE_A,
    )
    seed_operation_profile(
        bundle,
        tenant_id=TENANT_B,
        workspace_id=WORKSPACE_B,
    )


def seed_policy_composition_deny_fixture(bundle: CollaborativeWorkRepositories) -> None:
    seed_membership(
        bundle,
        membership_id="membership-alice",
        principal_id=PRINCIPAL_ALICE,
    )
    seed_authority_grant(
        bundle,
        grant_id="grant-alice",
        principal_id=PRINCIPAL_ALICE,
    )
    seed_workspace_policy(bundle, policy_rule_id="workspace-allow")
    seed_resource_policy(
        bundle,
        policy_rule_id="resource-deny-a",
        resource_scope=RESOURCE_A,
        action=PolicyAction.DENY,
    )
    seed_operation_profile(bundle)


def seed_missing_policy_layer_fixture(bundle: CollaborativeWorkRepositories) -> None:
    seed_membership(
        bundle,
        membership_id="membership-alice",
        principal_id=PRINCIPAL_ALICE,
    )
    seed_authority_grant(
        bundle,
        grant_id="grant-alice",
        principal_id=PRINCIPAL_ALICE,
    )
    seed_resource_policy(
        bundle,
        policy_rule_id="resource-allow-a",
        resource_scope=RESOURCE_A,
    )
    seed_operation_profile(bundle, workspace_required=True)
