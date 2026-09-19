# © Artur Czarnecki. All rights reserved.

"""Explicit governance composition for orchestration MSE boundary (GR-10-R9-R1)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from intergrax.collaborative_work.authority import CollaborativeWorkAuthorityResolver
from intergrax.collaborative_work.enforcement_gate import (
    CollaborativeWorkEnforcementGate,
    MeaningfulSideEffectPolicyEvaluator,
)
from intergrax.collaborative_work.policy_source import CollaborativePolicyEvaluator
from intergrax.collaborative_work.repository import (
    AuthorityDelegationRepository,
    CollaborativeOperationPolicyProfileRepository,
    CollaborativePolicyRepository,
    PrincipalAuthorityRepository,
    WorkspaceMembershipRepository,
)
from intergrax.contracts.canonical_inner_governance import CanonicalInnerExecutionGuardPort
from intergrax.contracts.decision_requirement_policy import DecisionRequirementPolicy
from intergrax.runtime.governance.governance_evidence_recorder import GovernanceEvidenceRecorder
from intergrax.runtime.governance.meaningful_side_effect_authorization_composition import (
    build_default_canonical_inner_execution_guard,
    build_meaningful_side_effect_authorization_boundary,
)
from intergrax.runtime.policy.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationBoundary,
)
from intergrax.utils.time_provider import SystemTimeProvider


def build_orchestration_meaningful_side_effect_authorization_boundary(
    *,
    profile_repository: CollaborativeOperationPolicyProfileRepository,
    membership_repository: WorkspaceMembershipRepository,
    principal_authority_repository: PrincipalAuthorityRepository,
    delegation_repository: AuthorityDelegationRepository,
    collaborative_policy_repository: CollaborativePolicyRepository,
    runtime_policy_evaluator: MeaningfulSideEffectPolicyEvaluator,
    inner_execution_guard: CanonicalInnerExecutionGuardPort | None = None,
    decision_requirement_policy: DecisionRequirementPolicy | None = None,
    governance_evidence_recorder: GovernanceEvidenceRecorder | None = None,
    clock: Callable[[], datetime] | None = None,
) -> MeaningfulSideEffectAuthorizationBoundary:
    """Wire collaborative-work enforcement behind the shared MSE boundary (explicit ports only)."""
    resolved_clock = clock or SystemTimeProvider.utc_now
    gate = CollaborativeWorkEnforcementGate(
        profile_repository=profile_repository,
        authority_resolver=CollaborativeWorkAuthorityResolver(
            membership_repository=membership_repository,
            delegation_repository=delegation_repository,
            principal_authority_repository=principal_authority_repository,
            clock=resolved_clock,
        ),
        policy_evaluator=CollaborativePolicyEvaluator(collaborative_policy_repository),
        runtime_policy_evaluator=runtime_policy_evaluator,
    )
    guard = (
        inner_execution_guard
        if inner_execution_guard is not None
        else build_default_canonical_inner_execution_guard()
    )
    return build_meaningful_side_effect_authorization_boundary(
        enforcement_gate=gate,
        inner_execution_guard=guard,
        decision_requirement_policy=decision_requirement_policy,
        governance_evidence_recorder=governance_evidence_recorder,
    )


__all__ = [
    "build_orchestration_meaningful_side_effect_authorization_boundary",
]
