# © Artur Czarnecki. All rights reserved.

"""Local/demo/test fixture — in-memory Collaborative Work state for non-production wiring."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from external_contractor_adapter.side_effect_actions import (
    ACTION_ACCEPT_QUOTE,
    ACTION_CANCEL_EXTERNAL_WORK,
    ACTION_CREATE_EXTERNAL_WORK,
)
from governed_contractor_application.host.collaborative_work_boundary import (
    build_external_work_authorization_boundary,
)
from intergrax.collaborative_work.enforcement_gate import MeaningfulSideEffectPolicyEvaluator
from intergrax.collaborative_work.in_memory_repository import (
    InMemoryAuthorityDelegationRepository,
    InMemoryCollaborativeOperationPolicyProfileRepository,
    InMemoryCollaborativePolicyRepository,
    InMemoryPrincipalAuthorityRepository,
    InMemoryWorkspaceMembershipRepository,
)
from intergrax.collaborative_work.persistence import CollaborativeWorkRepositories
from intergrax.collaborative_work.repository import (
    CreateCollaborativeOperationPolicyProfileCommand,
    CreateCollaborativePolicyRuleCommand,
    CreatePrincipalAuthorityGrantCommand,
    CreateWorkspaceMembershipCommand,
)
from intergrax.contracts.active_execution_task_scope import ActiveExecutionTaskScopePort
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
from intergrax.contracts.decision_requirement_policy import DecisionRequirementPolicy
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
_DEFAULT_FIXTURE_CLOCK = datetime(2026, 6, 15, 12, 0, tzinfo=UTC)


class _NoopCollaborativeWorkStore:
    def close(self) -> None:
        return None


def build_in_memory_collaborative_work_repositories() -> CollaborativeWorkRepositories:
    """Fresh in-memory MP-1 repository bundle for tests and offline tooling."""
    return CollaborativeWorkRepositories(
        membership=InMemoryWorkspaceMembershipRepository(),
        delegation=InMemoryAuthorityDelegationRepository(),
        principal_authority=InMemoryPrincipalAuthorityRepository(),
        policy=InMemoryCollaborativePolicyRepository(),
        operation_profile=InMemoryCollaborativeOperationPolicyProfileRepository(),
        store=_NoopCollaborativeWorkStore(),
    )


def seed_external_work_collaborative_governance_state(
    repositories: CollaborativeWorkRepositories,
    *,
    tenant_id: str,
    workspace_id: str,
    principal_id: str,
    authority_scope: str = _DEFAULT_SCOPE,
    operations: tuple[str, ...] = _EXTERNAL_WORK_OPERATIONS,
    workspace_policy_action: PolicyAction = PolicyAction.ALLOW,
    seed_workspace_policy: bool = True,
) -> None:
    """Populate authoritative membership, authority, workspace policy, and operation profiles."""
    repositories.membership.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            membership_id=f"membership-{principal_id}",
            principal_id=principal_id,
            role=WorkspaceMembershipRole.MEMBER,
            status=MembershipStatus.ACTIVE,
        )
    )
    repositories.principal_authority.create(
        CreatePrincipalAuthorityGrantCommand(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            authority_grant_id=f"grant-{principal_id}",
            principal_id=principal_id,
            authority_scopes=(authority_scope,),
            status=AuthorityGrantStatus.ACTIVE,
        )
    )
    if seed_workspace_policy:
        repositories.policy.create(
            CreateCollaborativePolicyRuleCommand(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                policy_rule_id="workspace-allow",
                layer=PolicyCompositionLayer.WORKSPACE_POLICY,
                authority_scope=authority_scope,
                action=workspace_policy_action,
                status=CollaborativePolicyRuleStatus.ACTIVE,
            )
        )
    for operation_id in operations:
        repositories.operation_profile.create(
            CreateCollaborativeOperationPolicyProfileCommand(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                operation_id=operation_id,
                authority_scope=authority_scope,
                workspace_policy_applicability=PolicyLayerApplicability.REQUIRED,
                resource_policy_applicability=PolicyLayerApplicability.NOT_APPLICABLE,
                runtime_policy_applicability=PolicyLayerApplicability.REQUIRED,
                resource_requirement=OperationPolicyRequirement.NOT_APPLICABLE,
                meaningful_side_effect_requirement=OperationPolicyRequirement.REQUIRED,
                status=CollaborativeOperationPolicyProfileStatus.ACTIVE,
            )
        )


def build_seeded_in_memory_external_work_authorization_boundary(
    runtime_policy_evaluator: MeaningfulSideEffectPolicyEvaluator,
    *,
    tenant_id: str,
    workspace_id: str,
    principal_id: str,
    decision_requirement_policy: DecisionRequirementPolicy | None = None,
    task_scope: ActiveExecutionTaskScopePort | None = None,
    authority_clock: Callable[[], datetime] | None = None,
) -> MeaningfulSideEffectAuthorizationBoundary:
    """Offline/demo helper — explicit in-memory seeding then contract-based boundary build."""
    repositories = build_in_memory_collaborative_work_repositories()
    seed_external_work_collaborative_governance_state(
        repositories,
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        principal_id=principal_id,
    )
    return build_external_work_authorization_boundary(
        runtime_policy_evaluator,
        collaborative_work_repositories=repositories,
        authority_clock=authority_clock or (lambda: _DEFAULT_FIXTURE_CLOCK),
        decision_requirement_policy=decision_requirement_policy,
        task_scope=task_scope,
    )
