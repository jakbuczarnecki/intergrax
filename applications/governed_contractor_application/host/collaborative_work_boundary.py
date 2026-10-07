# © Artur Czarnecki. All rights reserved.

"""Host composition helper — canonical side-effect authorization boundary for External Work."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from external_contractor_adapter.side_effect_actions import ACTION_ACCEPT_QUOTE
from intergrax.collaborative_work.authority import CollaborativeWorkAuthorityResolver
from intergrax.collaborative_work.enforcement_gate import CollaborativeWorkEnforcementGate
from intergrax.collaborative_work.persistence import CollaborativeWorkRepositories
from intergrax.collaborative_work.policy_source import CollaborativePolicyEvaluator
from intergrax.contracts.active_execution_task_scope import ActiveExecutionTaskScopePort
from intergrax.contracts.decision_requirement_policy import DecisionRequirementPolicy
from intergrax.contracts.meaningful_side_effect_policy import (
    MeaningfulSideEffectPolicyEvaluator,
)
from intergrax.runtime.governance.decision_requirement_policy import (
    decision_governed_side_effect_requirement_policy,
)
from intergrax.contracts.governed_execution_governance_evidence import (
    GovernanceEvidencePersistencePort,
)
from intergrax.runtime.governance.governance_evidence_composition import (
    build_governance_evidence_recorder,
)
from intergrax.runtime.governance.meaningful_side_effect_authorization_composition import (
    build_default_wired_meaningful_side_effect_authorization_boundary,
    build_meaningful_side_effect_authorization_boundary,
)
from intergrax.runtime.policy.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationBoundary,
)


def default_external_work_decision_requirement_policy() -> DecisionRequirementPolicy:
    """Production default — quote acceptance requires Decision provenance (GR-6-R2)."""
    return decision_governed_side_effect_requirement_policy(
        required_actions=frozenset({ACTION_ACCEPT_QUOTE}),
    )


def build_external_work_authorization_boundary(
    runtime_policy_evaluator: MeaningfulSideEffectPolicyEvaluator,
    *,
    collaborative_work_repositories: CollaborativeWorkRepositories,
    authority_clock: Callable[[], datetime] | None = None,
    decision_requirement_policy: DecisionRequirementPolicy | None = None,
    task_scope: ActiveExecutionTaskScopePort | None = None,
    governance_evidence_persistence: GovernanceEvidencePersistencePort | None = None,
) -> MeaningfulSideEffectAuthorizationBoundary:
    """Construct a canonical boundary from injected Collaborative Work repository contracts."""
    repos = collaborative_work_repositories
    resolved_clock = authority_clock or (lambda: datetime.now(UTC))
    gate = CollaborativeWorkEnforcementGate(
        profile_repository=repos.operation_profile,
        authority_resolver=CollaborativeWorkAuthorityResolver(
            membership_repository=repos.membership,
            delegation_repository=repos.delegation,
            principal_authority_repository=repos.principal_authority,
            clock=resolved_clock,
        ),
        policy_evaluator=CollaborativePolicyEvaluator(repos.policy),
        runtime_policy_evaluator=runtime_policy_evaluator,
    )
    resolved_decision_requirement_policy = (
        decision_requirement_policy
        if decision_requirement_policy is not None
        else default_external_work_decision_requirement_policy()
    )
    if governance_evidence_persistence is None:
        return build_default_wired_meaningful_side_effect_authorization_boundary(
            enforcement_gate=gate,
            decision_requirement_policy=resolved_decision_requirement_policy,
            task_scope=task_scope,
        )
    recorder = build_governance_evidence_recorder(
        persistence=governance_evidence_persistence,
    )
    from intergrax.runtime.governance.meaningful_side_effect_authorization_composition import (
        build_canonical_inner_execution_guard,
    )

    resolved_task_scope = task_scope
    if resolved_task_scope is None:
        from intergrax.runtime.task.active_task_registry import ActiveTaskRegistryTaskScopeResolver

        resolved_task_scope = ActiveTaskRegistryTaskScopeResolver()
    return build_meaningful_side_effect_authorization_boundary(
        enforcement_gate=gate,
        inner_execution_guard=build_canonical_inner_execution_guard(
            task_scope=resolved_task_scope,
        ),
        decision_requirement_policy=resolved_decision_requirement_policy,
        governance_evidence_recorder=recorder,
    )
