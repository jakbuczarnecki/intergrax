# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit DecisionRequirementPolicy binding for orchestration consequential MSE (GR-10-R10)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from intergrax.collaborative_work.repository import (
    AuthorityDelegationRepository,
    CollaborativeOperationPolicyProfileRepository,
    CollaborativePolicyRepository,
    PrincipalAuthorityRepository,
    WorkspaceMembershipRepository,
)
from intergrax.contracts.canonical_inner_governance import CanonicalInnerExecutionGuardPort
from intergrax.contracts.decision_requirement_policy import DecisionRequirementPolicy
from intergrax.runtime.governance.decision_requirement_policy import (
    PermissiveDecisionRequirementPolicy,
)
from intergrax.runtime.governance.governance_evidence_recorder import GovernanceEvidenceRecorder
from intergrax.runtime.governance.orchestration_meaningful_side_effect_composition import (
    build_orchestration_meaningful_side_effect_authorization_boundary,
)
from intergrax.collaborative_work.enforcement_gate import MeaningfulSideEffectPolicyEvaluator
from intergrax.runtime.policy.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationBoundary,
)


class OrchestrationDecisionBoundCompositionError(RuntimeError):
    """Fail closed when production orchestration cannot bind DecisionRequirementPolicy."""


def default_orchestration_decision_requirement_policy() -> DecisionRequirementPolicy:
    """Explicit production default — host rules may mark specific effects REQUIRED (GR-6-R1)."""
    return PermissiveDecisionRequirementPolicy()


def resolve_orchestration_decision_requirement_policy(
    explicit: DecisionRequirementPolicy | None,
    *,
    production_mode: bool,
) -> DecisionRequirementPolicy:
    """Resolve injectable policy — production composition always binds an explicit policy instance."""
    if explicit is not None:
        return explicit
    if production_mode:
        return default_orchestration_decision_requirement_policy()
    return PermissiveDecisionRequirementPolicy()


def build_production_orchestration_meaningful_side_effect_authorization_boundary(
    *,
    profile_repository: CollaborativeOperationPolicyProfileRepository,
    membership_repository: WorkspaceMembershipRepository,
    principal_authority_repository: PrincipalAuthorityRepository,
    delegation_repository: AuthorityDelegationRepository,
    collaborative_policy_repository: CollaborativePolicyRepository,
    runtime_policy_evaluator: MeaningfulSideEffectPolicyEvaluator,
    decision_requirement_policy: DecisionRequirementPolicy | None = None,
    inner_execution_guard: CanonicalInnerExecutionGuardPort | None = None,
    governance_evidence_recorder: GovernanceEvidenceRecorder | None = None,
    clock: Callable[[], datetime] | None = None,
    production_mode: bool = True,
) -> MeaningfulSideEffectAuthorizationBoundary:
    """Wire orchestration MSE with mandatory explicit DecisionRequirementPolicy (composition root)."""
    if not production_mode:
        raise OrchestrationDecisionBoundCompositionError(
            "production orchestration MSE boundary requires production_mode=True",
        )
    resolved_policy = resolve_orchestration_decision_requirement_policy(
        decision_requirement_policy,
        production_mode=True,
    )
    return build_orchestration_meaningful_side_effect_authorization_boundary(
        profile_repository=profile_repository,
        membership_repository=membership_repository,
        principal_authority_repository=principal_authority_repository,
        delegation_repository=delegation_repository,
        collaborative_policy_repository=collaborative_policy_repository,
        runtime_policy_evaluator=runtime_policy_evaluator,
        inner_execution_guard=inner_execution_guard,
        decision_requirement_policy=resolved_policy,
        governance_evidence_recorder=governance_evidence_recorder,
        clock=clock,
    )


__all__ = [
    "OrchestrationDecisionBoundCompositionError",
    "build_production_orchestration_meaningful_side_effect_authorization_boundary",
    "default_orchestration_decision_requirement_policy",
    "resolve_orchestration_decision_requirement_policy",
]
