# © Artur Czarnecki. All rights reserved.

"""Composition-only fixture: materialize real platform MSE port for MP-7B proofs.

Private ``intergrax.collaborative_work`` imports are confined to this module.
The Tier-3 consumer never imports this graph — it receives only the public Protocol.
"""

from __future__ import annotations

from datetime import UTC, datetime

from intergrax.collaborative_work.in_memory_repository import (
    InMemoryAuthorityDelegationRepository,
    InMemoryCollaborativeOperationPolicyProfileRepository,
    InMemoryCollaborativePolicyRepository,
    InMemoryPrincipalAuthorityRepository,
    InMemoryWorkspaceMembershipRepository,
)
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
from intergrax.contracts.execution_identity import TaskId
from intergrax.contracts.meaningful_side_effect import (
    MeaningfulSideEffectKind,
    MeaningfulSideEffectRequest,
)
from intergrax.contracts.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationPort,
)
from intergrax.contracts.meaningful_side_effect_policy import (
    MeaningfulSideEffectPolicyRule,
)
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.runtime.governance.decision_requirement_policy import (
    PermissiveDecisionRequirementPolicy,
)
from intergrax.runtime.governance.orchestration_decision_bound_effect_composition import (
    build_production_orchestration_meaningful_side_effect_authorization_boundary,
)
from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine
from tests.unit.runtime.governance.gr3_test_support import (
    default_gr3_identity_bundle,
    default_gr3_inner_guard,
)

_TENANT = "mp7b-tenant"
_WORKSPACE = "mp7b-workspace"
_ACTING = "mp7b-principal"
_OPERATION = "mp7b.proof.mutation"
_SCOPE = "mp7b.proof.scope"
_RESOURCE = "mp7b-resource-1"
_NOW = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)


def _build_boundary(
    *,
    membership_repo: InMemoryWorkspaceMembershipRepository,
    authority_repo: InMemoryPrincipalAuthorityRepository,
    policy_repo: InMemoryCollaborativePolicyRepository,
    profile_repo: InMemoryCollaborativeOperationPolicyProfileRepository,
    task_id: TaskId,
) -> MeaningfulSideEffectAuthorizationPort:
    return build_production_orchestration_meaningful_side_effect_authorization_boundary(
        profile_repository=profile_repo,
        membership_repository=membership_repo,
        principal_authority_repository=authority_repo,
        delegation_repository=InMemoryAuthorityDelegationRepository(),
        collaborative_policy_repository=policy_repo,
        runtime_policy_evaluator=RuntimePolicyEngine(
            meaningful_side_effect_rules=(
                MeaningfulSideEffectPolicyRule(
                    rule_id="mp7b.runtime.allow",
                    action=_OPERATION,
                    decision=PolicyAction.ALLOW,
                ),
            )
        ),
        decision_requirement_policy=PermissiveDecisionRequirementPolicy(),
        inner_execution_guard=default_gr3_inner_guard(task_id),
        clock=lambda: _NOW,
        production_mode=True,
    )


def compose_deny_platform_port() -> tuple[
    MeaningfulSideEffectAuthorizationPort,
    CollaborativeWorkEnforcementRequest,
]:
    """Empty CW state → platform DENY via public port (real composition)."""
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    port = _build_boundary(
        membership_repo=InMemoryWorkspaceMembershipRepository(),
        authority_repo=InMemoryPrincipalAuthorityRepository(),
        policy_repo=InMemoryCollaborativePolicyRepository(),
        profile_repo=InMemoryCollaborativeOperationPolicyProfileRepository(),
        task_id=task_id,
    )
    # Embedded membership without authoritative store state must still DENY.
    fake_membership = WorkspaceMembership(
        membership_id="mp7b-fake-membership",
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        principal_id=_ACTING,
        role=WorkspaceMembershipRole.MEMBER,
        status=MembershipStatus.ACTIVE,
        revision=0,
    )
    request = CollaborativeWorkEnforcementRequest(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation_id=_OPERATION,
        acting_principal_id=_ACTING,
        resource_scope=_RESOURCE,
        membership=fake_membership,
        meaningful_side_effect_request=MeaningfulSideEffectRequest(
            action=_OPERATION,
            kinds=(MeaningfulSideEffectKind.MUTATION,),
            side_effect_scope_id="mp7b-side-effect-scope",
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
            principal_id=_ACTING,
            tenant_id=_TENANT,
            resource=_RESOURCE,
        ),
    )
    return port, request


def compose_allow_platform_port() -> tuple[
    MeaningfulSideEffectAuthorizationPort,
    CollaborativeWorkEnforcementRequest,
]:
    """Seeded CW state → platform ALLOW via public port (real composition)."""
    membership_repo = InMemoryWorkspaceMembershipRepository()
    authority_repo = InMemoryPrincipalAuthorityRepository()
    policy_repo = InMemoryCollaborativePolicyRepository()
    profile_repo = InMemoryCollaborativeOperationPolicyProfileRepository()
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()

    membership = membership_repo.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            membership_id="mp7b-membership-1",
            principal_id=_ACTING,
            role=WorkspaceMembershipRole.MEMBER,
            status=MembershipStatus.ACTIVE,
        )
    )
    authority_repo.create(
        CreatePrincipalAuthorityGrantCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            authority_grant_id="mp7b-grant-1",
            principal_id=_ACTING,
            authority_scopes=(_SCOPE,),
            status=AuthorityGrantStatus.ACTIVE,
        )
    )
    policy_repo.create(
        CreateCollaborativePolicyRuleCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            policy_rule_id="mp7b-ws-allow",
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
            policy_rule_id="mp7b-res-allow",
            layer=PolicyCompositionLayer.RESOURCE_POLICY,
            authority_scope=_SCOPE,
            resource_scope=_RESOURCE,
            action=PolicyAction.ALLOW,
            status=CollaborativePolicyRuleStatus.ACTIVE,
        )
    )
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

    port = _build_boundary(
        membership_repo=membership_repo,
        authority_repo=authority_repo,
        policy_repo=policy_repo,
        profile_repo=profile_repo,
        task_id=task_id,
    )
    request = CollaborativeWorkEnforcementRequest(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation_id=_OPERATION,
        acting_principal_id=_ACTING,
        resource_scope=_RESOURCE,
        membership=WorkspaceMembership.model_validate(membership.model_dump()),
        meaningful_side_effect_request=MeaningfulSideEffectRequest(
            action=_OPERATION,
            kinds=(MeaningfulSideEffectKind.MUTATION,),
            side_effect_scope_id="mp7b-side-effect-scope",
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
            principal_id=_ACTING,
            tenant_id=_TENANT,
            resource=_RESOURCE,
        ),
    )
    return port, request


__all__ = [
    "compose_allow_platform_port",
    "compose_deny_platform_port",
]
