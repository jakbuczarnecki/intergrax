# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Production composition for orchestration tool MSE boundary (GR-10-R9)."""

from __future__ import annotations

from datetime import UTC, datetime
from intergrax.collaborative_work.authority import CollaborativeWorkAuthorityResolver
from intergrax.collaborative_work.enforcement_gate import (
    CollaborativeWorkEnforcementGate,
    MeaningfulSideEffectPolicyEvaluator,
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
    PrincipalAuthorityRepository,
    WorkspaceMembershipRepository,
)
from intergrax.contracts.collaborative_work import (
    AuthorityGrantStatus,
    CollaborativeOperationPolicyProfile,
    CollaborativeOperationPolicyProfileStatus,
    MembershipStatus,
    OperationPolicyRequirement,
    PolicyLayerApplicability,
    PrincipalAuthorityGrant,
    WorkspaceMembership,
    WorkspaceMembershipRole,
)
from intergrax.contracts.meaningful_side_effect_policy import MeaningfulSideEffectPolicyRule
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.runtime.governance.meaningful_side_effect_authorization_composition import (
    build_default_canonical_inner_execution_guard,
    build_meaningful_side_effect_authorization_boundary,
)
from intergrax.runtime.nexus.tools.meaningful_side_effect_authorization_port import (
    MeaningfulSideEffectAuthorizationPort,
)
from intergrax.runtime.nexus.tools.tool_invocation_meaningful_side_effect import (
    ORCHESTRATION_TOOL_MSE_OPERATION_ID,
)
from intergrax.runtime.policy.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationBoundary,
)
from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine

_DEFAULT_AUTHORITY_SCOPE = "orchestration.tool_invocation"
_COMPOSITION_CLOCK = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)


class _PermissiveOrchestrationProfileRepository(
    InMemoryCollaborativeOperationPolicyProfileRepository,
):
    """Lab/production-default profile lookup for orchestration tool MSE operation id."""

    def get_for_operation(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        operation_id: str,
    ) -> CollaborativeOperationPolicyProfile | None:
        if operation_id != ORCHESTRATION_TOOL_MSE_OPERATION_ID:
            return super().get_for_operation(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                operation_id=operation_id,
            )
        return CollaborativeOperationPolicyProfile(
            tenant_id=tenant_id.strip(),
            workspace_id=workspace_id.strip(),
            operation_id=operation_id,
            authority_scope=_DEFAULT_AUTHORITY_SCOPE,
            workspace_policy_applicability=PolicyLayerApplicability.NOT_APPLICABLE,
            resource_policy_applicability=PolicyLayerApplicability.NOT_APPLICABLE,
            runtime_policy_applicability=PolicyLayerApplicability.REQUIRED,
            resource_requirement=OperationPolicyRequirement.NOT_APPLICABLE,
            meaningful_side_effect_requirement=OperationPolicyRequirement.REQUIRED,
            status=CollaborativeOperationPolicyProfileStatus.ACTIVE,
            revision=0,
        )


class _PrincipalAuthorityRepository(InMemoryPrincipalAuthorityRepository):
    """Synthesize base authority grants for orchestration tool MSE (composition default)."""

    def get_for_principal(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        principal_id: str,
    ) -> PrincipalAuthorityGrant | None:
        existing = super().get_for_principal(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            principal_id=principal_id,
        )
        if existing is not None:
            return existing
        return PrincipalAuthorityGrant(
            authority_grant_id=f"orchestration:auto:{tenant_id}:{workspace_id}:{principal_id}",
            tenant_id=tenant_id.strip(),
            workspace_id=workspace_id.strip(),
            principal_id=principal_id.strip(),
            authority_scopes=(_DEFAULT_AUTHORITY_SCOPE,),
            status=AuthorityGrantStatus.ACTIVE,
            revision=0,
        )


class _PrincipalMembershipRepository(InMemoryWorkspaceMembershipRepository):
    """Synthesize active membership for canonical principal resolution (composition default)."""

    def get_for_principal(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        principal_id: str,
    ) -> WorkspaceMembership | None:
        existing = super().get_for_principal(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            principal_id=principal_id,
        )
        if existing is not None:
            return existing
        return WorkspaceMembership(
            membership_id=f"orchestration:auto:{tenant_id}:{workspace_id}:{principal_id}",
            tenant_id=tenant_id.strip(),
            workspace_id=workspace_id.strip(),
            principal_id=principal_id.strip(),
            role=WorkspaceMembershipRole.MEMBER,
            status=MembershipStatus.ACTIVE,
            revision=0,
        )


def build_default_orchestration_meaningful_side_effect_authorization_boundary(
    *,
    runtime_policy_evaluator: MeaningfulSideEffectPolicyEvaluator | None = None,
) -> MeaningfulSideEffectAuthorizationBoundary:
    """Platform default for production orchestration tool invoker when host omits injection."""
    profile_repo = _PermissiveOrchestrationProfileRepository()
    authority_repo: PrincipalAuthorityRepository = _PrincipalAuthorityRepository()
    runtime = runtime_policy_evaluator or RuntimePolicyEngine(
        meaningful_side_effect_rules=(
            MeaningfulSideEffectPolicyRule(
                rule_id="orchestration.tool_invocation.allow",
                decision=PolicyAction.ALLOW,
            ),
        ),
    )
    membership_repo: WorkspaceMembershipRepository = _PrincipalMembershipRepository()
    gate = CollaborativeWorkEnforcementGate(
        profile_repository=profile_repo,
        authority_resolver=CollaborativeWorkAuthorityResolver(
            membership_repository=membership_repo,
            delegation_repository=InMemoryAuthorityDelegationRepository(),
            principal_authority_repository=authority_repo,
            clock=lambda: _COMPOSITION_CLOCK,
        ),
        policy_evaluator=CollaborativePolicyEvaluator(
            InMemoryCollaborativePolicyRepository(),
        ),
        runtime_policy_evaluator=runtime,
    )
    return build_meaningful_side_effect_authorization_boundary(
        enforcement_gate=gate,
        inner_execution_guard=build_default_canonical_inner_execution_guard(),
    )


def as_meaningful_side_effect_authorization_port(
    boundary: MeaningfulSideEffectAuthorizationBoundary,
) -> MeaningfulSideEffectAuthorizationPort:
    """Narrow concrete composition product to the Nexus port surface."""
    return boundary


__all__ = [
    "as_meaningful_side_effect_authorization_port",
    "build_default_orchestration_meaningful_side_effect_authorization_boundary",
]
