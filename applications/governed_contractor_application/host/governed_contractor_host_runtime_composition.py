# © Artur Czarnecki. All rights reserved.

"""GR-6 / EBH-2D-D-R2 — host runtime composition (not declarative settings)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.collaborative_work.persistence import (
    CollaborativeWorkMaterializedRepositories,
    CollaborativeWorkRepositories,
    collaborative_work_core_repositories,
)
from intergrax.contracts.active_execution_task_scope import ActiveExecutionTaskScopePort
from intergrax.contracts.decision_requirement_policy import DecisionRequirementPolicy
from intergrax.contracts.execution_evidence.attestation import HostAttestor
from intergrax.integrations.contracts.external_work import ExternalWorkIntegration
from intergrax.runtime.policy.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationBoundary,
)
from intergrax.runtime.policy.runtime_policy_bundle_evaluator import (
    RuntimePolicyBundleEvaluator,
)

from governed_contractor_application.host.collaborative_work_boundary import (
    build_external_work_authorization_boundary,
    default_external_work_decision_requirement_policy,
)
from governed_contractor_application.host.settings import GovernedContractorBackendSettings
from intergrax.contracts.runtime_policy_bundle import ImmutableRuntimePolicyBundle


@dataclass(frozen=True, slots=True)
class GovernedContractorHostRuntimeComposition:
    """Runtime ports and services for governed_contractor_application (host-scoped)."""

    external_work_integration: ExternalWorkIntegration | None = None
    meaningful_side_effect_authorization_boundary: (
        MeaningfulSideEffectAuthorizationBoundary | None
    ) = None
    decision_requirement_policy: DecisionRequirementPolicy | None = None
    collaborative_work_repositories: CollaborativeWorkMaterializedRepositories | None = None
    active_execution_task_scope: ActiveExecutionTaskScopePort | None = None
    host_attestor: HostAttestor | None = None


def resolve_production_runtime_policy_bundle_from_settings(
    settings: GovernedContractorBackendSettings,
) -> ImmutableRuntimePolicyBundle:
    bundle = settings.runtime_policy_bundle
    if bundle is None:
        raise ValueError(
            "production external work requires settings.runtime_policy_bundle "
            "(ImmutableRuntimePolicyBundle); configure an explicit bundle at composition time",
        )
    return bundle


def _resolve_decision_requirement_policy(
    runtime: GovernedContractorHostRuntimeComposition,
    override: DecisionRequirementPolicy | None,
) -> DecisionRequirementPolicy:
    if override is not None:
        return override
    if runtime.decision_requirement_policy is not None:
        return runtime.decision_requirement_policy
    return default_external_work_decision_requirement_policy()


def resolve_production_collaborative_work_repositories_from_runtime(
    runtime: GovernedContractorHostRuntimeComposition,
) -> CollaborativeWorkRepositories:
    bundle = runtime.collaborative_work_repositories
    if bundle is None:
        raise ValueError(
            "production external work requires runtime.collaborative_work_repositories "
            "(CollaborativeWorkMaterializedRepositories); inject authoritative repository "
            "state at composition time",
        )
    return collaborative_work_core_repositories(bundle)


def compose_governed_contractor_host_runtime(
    settings: GovernedContractorBackendSettings,
    *,
    runtime: GovernedContractorHostRuntimeComposition | None = None,
    integration: ExternalWorkIntegration | None = None,
    task_scope: ActiveExecutionTaskScopePort | None = None,
    collaborative_work_repositories: CollaborativeWorkMaterializedRepositories | None = None,
    decision_requirement_policy: DecisionRequirementPolicy | None = None,
    host_attestor: HostAttestor | None = None,
) -> GovernedContractorHostRuntimeComposition:
    """Build or extend host runtime composition without mutating declarative settings."""
    base = runtime or GovernedContractorHostRuntimeComposition()
    resolved_integration = (
        integration
        if integration is not None
        else base.external_work_integration
    )
    if resolved_integration is None:
        return GovernedContractorHostRuntimeComposition(
            external_work_integration=base.external_work_integration,
            meaningful_side_effect_authorization_boundary=(
                base.meaningful_side_effect_authorization_boundary
            ),
            decision_requirement_policy=(
                decision_requirement_policy
                if decision_requirement_policy is not None
                else base.decision_requirement_policy
            ),
            collaborative_work_repositories=(
                collaborative_work_repositories
                if collaborative_work_repositories is not None
                else base.collaborative_work_repositories
            ),
            active_execution_task_scope=(
                task_scope if task_scope is not None else base.active_execution_task_scope
            ),
            host_attestor=host_attestor if host_attestor is not None else base.host_attestor,
        )

    if (
        base.meaningful_side_effect_authorization_boundary is not None
        and base.external_work_integration is resolved_integration
        and task_scope is None
        and collaborative_work_repositories is None
        and decision_requirement_policy is None
        and host_attestor is None
    ):
        return base

    bundle = resolve_production_runtime_policy_bundle_from_settings(settings)
    decision_policy = _resolve_decision_requirement_policy(
        base,
        decision_requirement_policy,
    )
    resolved_cw = (
        collaborative_work_repositories
        if collaborative_work_repositories is not None
        else base.collaborative_work_repositories
    )
    if resolved_cw is None:
        raise ValueError(
            "production external work requires runtime.collaborative_work_repositories "
            "(CollaborativeWorkMaterializedRepositories); inject at composition time",
        )
    resolved_task_scope = (
        task_scope if task_scope is not None else base.active_execution_task_scope
    )
    if resolved_task_scope is None:
        raise ValueError(
            "production external work requires active_execution_task_scope at composition time",
        )
    cw_core = collaborative_work_core_repositories(resolved_cw)
    policy_evaluator = RuntimePolicyBundleEvaluator(bundle)
    boundary = build_external_work_authorization_boundary(
        policy_evaluator,
        collaborative_work_repositories=cw_core,
        decision_requirement_policy=decision_policy,
        task_scope=resolved_task_scope,
    )
    return GovernedContractorHostRuntimeComposition(
        external_work_integration=resolved_integration,
        meaningful_side_effect_authorization_boundary=boundary,
        decision_requirement_policy=decision_policy,
        collaborative_work_repositories=resolved_cw,
        active_execution_task_scope=resolved_task_scope,
        host_attestor=host_attestor if host_attestor is not None else base.host_attestor,
    )


__all__ = [
    "GovernedContractorHostRuntimeComposition",
    "compose_governed_contractor_host_runtime",
    "resolve_production_collaborative_work_repositories_from_runtime",
]
