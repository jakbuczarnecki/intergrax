# © Artur Czarnecki. All rights reserved.

"""Canonical harness-host meaningful side-effect authorization composition (GR-10-R9 / P2D-R1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
)
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.collaborative_work.persistence import (
    CollaborativeWorkMaterializedRepositories,
    collaborative_work_core_repositories,
)
from intergrax.collaborative_work.persistence_provider import (
    resolve_collaborative_work_repositories,
)
from intergrax.contracts.active_execution_task_scope import ActiveExecutionTaskScopePort
from intergrax.contracts.decision_requirement_policy import DecisionRequirementPolicy
from intergrax.contracts.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationPort,
)
from intergrax.contracts.governed_execution_governance_evidence import (
    GovernanceEvidencePersistencePort,
)
from intergrax.contracts.meaningful_side_effect_policy import (
    MeaningfulSideEffectPolicyEvaluator,
)
from intergrax.runtime.governance.orchestration_governance_evidence_composition import (
    require_strict_orchestration_governance_evidence_persistence,
)
from intergrax.runtime.governance.meaningful_side_effect_authorization_composition import (
    build_canonical_inner_execution_guard,
)
from intergrax.runtime.governance.orchestration_decision_bound_effect_composition import (
    build_production_orchestration_meaningful_side_effect_authorization_boundary,
)
from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine


@dataclass(frozen=True, slots=True)
class HarnessMeaningfulSideEffectAuthorizationWiring:
    """Resolved MSE authorization plus host-owned Collaborative Work persistence (if any)."""

    authorization_port: MeaningfulSideEffectAuthorizationPort | None
    owned_collaborative_work_persistence: (
        CollaborativeWorkMaterializedRepositories | None
    ) = None


def _build_port_from_materialized_repositories(
    bundle: CollaborativeWorkMaterializedRepositories,
    *,
    decision_requirement_policy: DecisionRequirementPolicy | None = None,
    runtime_policy_evaluator: MeaningfulSideEffectPolicyEvaluator | None = None,
    active_execution_task_scope: ActiveExecutionTaskScopePort | None = None,
    governance_evidence_persistence: GovernanceEvidencePersistencePort | None = None,
    runtime_event_persistence: object | None = None,
    production_mode: bool = True,
) -> MeaningfulSideEffectAuthorizationPort:
    core = collaborative_work_core_repositories(bundle)
    evaluator: MeaningfulSideEffectPolicyEvaluator = (
        runtime_policy_evaluator
        if runtime_policy_evaluator is not None
        else RuntimePolicyEngine()
    )
    inner_guard = (
        build_canonical_inner_execution_guard(task_scope=active_execution_task_scope)
        if active_execution_task_scope is not None
        else None
    )
    resolved_evidence = require_strict_orchestration_governance_evidence_persistence(
        explicit=governance_evidence_persistence,
        runtime_event_persistence=runtime_event_persistence,
        production_mode=production_mode,
    )
    return build_production_orchestration_meaningful_side_effect_authorization_boundary(
        profile_repository=core.operation_profile,
        membership_repository=core.membership,
        principal_authority_repository=core.principal_authority,
        delegation_repository=core.delegation,
        collaborative_policy_repository=core.policy,
        runtime_policy_evaluator=evaluator,
        decision_requirement_policy=decision_requirement_policy,
        inner_execution_guard=inner_guard,
        governance_evidence_persistence=resolved_evidence,
        production_mode=production_mode,
    )


def _collaborative_work_integration_profile(
    environment: ApplicationEnvironmentProfile,
    *,
    collaborative_work_integration_profile: IntegrationProfile | None = None,
) -> IntegrationProfile:
    if collaborative_work_integration_profile is not None:
        return collaborative_work_integration_profile
    return environment.integration_profile


def build_harness_host_meaningful_side_effect_authorization_port(
    environment: ApplicationEnvironmentProfile,
    *,
    collaborative_work_repositories: CollaborativeWorkMaterializedRepositories
    | None = None,
    collaborative_work_integration_profile: IntegrationProfile | None = None,
    decision_requirement_policy: DecisionRequirementPolicy | None = None,
    runtime_policy_evaluator: MeaningfulSideEffectPolicyEvaluator | None = None,
    active_execution_task_scope: ActiveExecutionTaskScopePort | None = None,
    governance_evidence_persistence: GovernanceEvidencePersistencePort | None = None,
    runtime_event_persistence: object | None = None,
) -> MeaningfulSideEffectAuthorizationPort:
    """Build platform default MSE authorization for strict Tier-3 harness hosts."""
    if collaborative_work_repositories is not None:
        bundle = collaborative_work_repositories
    else:
        bundle = resolve_collaborative_work_repositories(
            _collaborative_work_integration_profile(
                environment,
                collaborative_work_integration_profile=collaborative_work_integration_profile,
            ),
        )
    production_mode = environment.execution_mode.value == "strict"
    return _build_port_from_materialized_repositories(
        bundle,
        decision_requirement_policy=decision_requirement_policy,
        runtime_policy_evaluator=runtime_policy_evaluator,
        active_execution_task_scope=active_execution_task_scope,
        governance_evidence_persistence=governance_evidence_persistence,
        runtime_event_persistence=runtime_event_persistence,
        production_mode=production_mode,
    )


def resolve_harness_host_meaningful_side_effect_authorization_wiring(
    environment: ApplicationEnvironmentProfile,
    *,
    explicit: MeaningfulSideEffectAuthorizationPort | None = None,
    collaborative_work_repositories: CollaborativeWorkMaterializedRepositories
    | None = None,
    collaborative_work_integration_profile: IntegrationProfile | None = None,
    decision_requirement_policy: DecisionRequirementPolicy | None = None,
    runtime_policy_evaluator: MeaningfulSideEffectPolicyEvaluator | None = None,
    active_execution_task_scope: ActiveExecutionTaskScopePort | None = None,
    governance_evidence_persistence: GovernanceEvidencePersistencePort | None = None,
    runtime_event_persistence: object | None = None,
) -> HarnessMeaningfulSideEffectAuthorizationWiring:
    """Resolve MSE wiring: injectable override, strict default, or absent in non-strict hosts."""
    if explicit is not None:
        return HarnessMeaningfulSideEffectAuthorizationWiring(
            authorization_port=explicit,
        )
    production_mode = environment.execution_mode.value == "strict"
    if not production_mode:
        return HarnessMeaningfulSideEffectAuthorizationWiring(authorization_port=None)
    if collaborative_work_repositories is not None:
        return HarnessMeaningfulSideEffectAuthorizationWiring(
            authorization_port=_build_port_from_materialized_repositories(
                collaborative_work_repositories,
                decision_requirement_policy=decision_requirement_policy,
                runtime_policy_evaluator=runtime_policy_evaluator,
                active_execution_task_scope=active_execution_task_scope,
                governance_evidence_persistence=governance_evidence_persistence,
                runtime_event_persistence=runtime_event_persistence,
                production_mode=production_mode,
            ),
        )
    bundle = resolve_collaborative_work_repositories(
        _collaborative_work_integration_profile(
            environment,
            collaborative_work_integration_profile=collaborative_work_integration_profile,
        ),
    )
    return HarnessMeaningfulSideEffectAuthorizationWiring(
        authorization_port=_build_port_from_materialized_repositories(
            bundle,
            decision_requirement_policy=decision_requirement_policy,
            runtime_policy_evaluator=runtime_policy_evaluator,
            active_execution_task_scope=active_execution_task_scope,
            governance_evidence_persistence=governance_evidence_persistence,
            runtime_event_persistence=runtime_event_persistence,
            production_mode=production_mode,
        ),
        owned_collaborative_work_persistence=bundle,
    )


def resolve_harness_host_meaningful_side_effect_authorization_port(
    environment: ApplicationEnvironmentProfile,
    *,
    explicit: MeaningfulSideEffectAuthorizationPort | None = None,
    collaborative_work_repositories: CollaborativeWorkMaterializedRepositories
    | None = None,
    collaborative_work_integration_profile: IntegrationProfile | None = None,
    decision_requirement_policy: DecisionRequirementPolicy | None = None,
    runtime_policy_evaluator: MeaningfulSideEffectPolicyEvaluator | None = None,
    active_execution_task_scope: ActiveExecutionTaskScopePort | None = None,
    governance_evidence_persistence: GovernanceEvidencePersistencePort | None = None,
    runtime_event_persistence: object | None = None,
) -> MeaningfulSideEffectAuthorizationPort | None:
    """Resolve MSE port: injectable override, strict default, or absent in non-strict hosts."""
    return resolve_harness_host_meaningful_side_effect_authorization_wiring(
        environment,
        explicit=explicit,
        collaborative_work_repositories=collaborative_work_repositories,
        collaborative_work_integration_profile=collaborative_work_integration_profile,
        decision_requirement_policy=decision_requirement_policy,
        runtime_policy_evaluator=runtime_policy_evaluator,
        active_execution_task_scope=active_execution_task_scope,
        governance_evidence_persistence=governance_evidence_persistence,
        runtime_event_persistence=runtime_event_persistence,
    ).authorization_port


__all__ = [
    "HarnessMeaningfulSideEffectAuthorizationWiring",
    "build_harness_host_meaningful_side_effect_authorization_port",
    "resolve_harness_host_meaningful_side_effect_authorization_port",
    "resolve_harness_host_meaningful_side_effect_authorization_wiring",
]
