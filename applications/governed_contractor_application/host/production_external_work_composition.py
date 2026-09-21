# © Artur Czarnecki. All rights reserved.

"""GR-6-WIRE — production Decision → Governance → External Work composition root."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from external_contractor_adapter.external_work_adapter import ExternalWorkAdapter
from governed_contractor_application.host.collaborative_work_boundary import (
    default_external_work_decision_requirement_policy,
)
from governed_contractor_application.host.external_work_enterprise_reliability_bridge import (
    GovernedExternalWorkEnterpriseReliabilityBridge,
)
from governed_contractor_application.host.governed_contractor_host_runtime_composition import (
    GovernedContractorHostRuntimeComposition,
    resolve_production_collaborative_work_repositories_from_runtime,
    resolve_production_runtime_policy_bundle_from_settings,
)
from governed_contractor_application.host.orchestrator import GovernedExternalWorkOrchestrator
from governed_contractor_application.host.provider_invocation_lifecycle import (
    GovernedProviderInvocationDispatchGate,
)
from governed_contractor_application.host.settings import GovernedContractorBackendSettings
from governed_contractor_application.host.stores import (
    ContinuationStateStore,
    GovernedExecutionStore,
    PolicyBundleArtifactStore,
    ProofReceiptStore,
)
from intergrax.contracts.enterprise_reliability.provider_invocation_reliability_evidence import (
    ProviderInvocationReliabilityEvidenceObserver,
)
from intergrax.contracts.provider_invocation_store import ProviderInvocationStore
from intergrax.collaborative_work.persistence import CollaborativeWorkMaterializedRepositories
from intergrax.contracts.active_execution_task_scope import ActiveExecutionTaskScopePort
from intergrax.contracts.decision_requirement_policy import DecisionRequirementPolicy
from intergrax.contracts.execution_evidence.attestation import HostAttestor
from intergrax.contracts.external_work_provider_capabilities import (
    ExternalWorkProviderCapabilities,
)
from intergrax.contracts.runtime_policy_bundle import ImmutableRuntimePolicyBundle
from intergrax.integrations.contracts.external_work import ExternalWorkIntegration
from intergrax.runtime.policy.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationBoundary,
)
from intergrax.runtime.policy.runtime_policy_bundle_evaluator import (
    RuntimePolicyBundleEvaluator,
)

@dataclass(frozen=True, slots=True)
class GovernedExternalWorkProductionRuntime:
    """Injectable production stack for governed external-work lifecycle."""

    adapter: ExternalWorkAdapter
    orchestrator: GovernedExternalWorkOrchestrator
    authorization_boundary: MeaningfulSideEffectAuthorizationBoundary
    decision_requirement_policy: DecisionRequirementPolicy
    policy_bundle: ImmutableRuntimePolicyBundle
    policy_evaluator: RuntimePolicyBundleEvaluator


def resolve_production_runtime_policy_bundle_evaluator(
    settings: GovernedContractorBackendSettings,
) -> RuntimePolicyBundleEvaluator | None:
    """Bundle-backed runtime policy for harness orchestration MSE when configured."""
    bundle = settings.runtime_policy_bundle
    if bundle is None:
        return None
    return RuntimePolicyBundleEvaluator(bundle)


def resolve_production_runtime_policy_bundle(
    settings: GovernedContractorBackendSettings,
) -> ImmutableRuntimePolicyBundle:
    return resolve_production_runtime_policy_bundle_from_settings(settings)


def resolve_production_decision_requirement_policy(
    runtime: GovernedContractorHostRuntimeComposition,
) -> DecisionRequirementPolicy:
    policy = runtime.decision_requirement_policy
    if policy is None:
        return default_external_work_decision_requirement_policy()
    return policy


def resolve_production_collaborative_work_repositories(
    runtime: GovernedContractorHostRuntimeComposition,
):
    return resolve_production_collaborative_work_repositories_from_runtime(runtime)


def build_governed_external_work_production_runtime(
    integration: ExternalWorkIntegration,
    *,
    tenant_id: str,
    workspace_id: str,
    principal_id: str,
    task_scope: ActiveExecutionTaskScopePort,
    capabilities: ExternalWorkProviderCapabilities,
    decision_requirement_policy: DecisionRequirementPolicy | None = None,
    policy_bundle: ImmutableRuntimePolicyBundle,
    collaborative_work_repositories: CollaborativeWorkMaterializedRepositories,
    execution_store: GovernedExecutionStore,
    receipt_store: ProofReceiptStore,
    bundle_store: PolicyBundleArtifactStore,
    continuation_store: ContinuationStateStore,
    provider_invocation_store: ProviderInvocationStore,
    attestor: HostAttestor | None = None,
    clock: Callable[[], datetime] | None = None,
    reliability_bridge: GovernedExternalWorkEnterpriseReliabilityBridge | None = None,
    reliability_evidence_observer: ProviderInvocationReliabilityEvidenceObserver | None = None,
) -> GovernedExternalWorkProductionRuntime:
    """Construct orchestrator + adapter wired through canonical governance boundary."""
    from governed_contractor_application.host.collaborative_work_boundary import (
        build_external_work_authorization_boundary,
    )
    from intergrax.collaborative_work.persistence import collaborative_work_core_repositories

    resolved_clock = clock or (lambda: datetime.now(timezone.utc))
    bundle = policy_bundle
    resolved_policy = (
        decision_requirement_policy
        if decision_requirement_policy is not None
        else default_external_work_decision_requirement_policy()
    )
    policy_evaluator = RuntimePolicyBundleEvaluator(
        bundle,
        clock=clock,
    )
    cw_repositories = collaborative_work_core_repositories(collaborative_work_repositories)
    authorization_boundary = build_external_work_authorization_boundary(
        policy_evaluator,
        collaborative_work_repositories=cw_repositories,
        decision_requirement_policy=resolved_policy,
        task_scope=task_scope,
    )
    if provider_invocation_store is None:
        raise ValueError(
            "production external work requires provider_invocation_store "
            "(ProviderInvocationStore); inject durable invocation lifecycle at composition time",
        )
    if not provider_invocation_store.is_durable:
        raise ValueError(
            "production external work requires durable provider_invocation_store "
            "(is_durable must be True); non-durable stores are for tests, fixtures, and offline demo only",
        )
    invocation_dispatch = GovernedProviderInvocationDispatchGate(
        store=provider_invocation_store,
        clock=resolved_clock,
    )
    adapter = ExternalWorkAdapter(
        integration,
        authorization_boundary=authorization_boundary,
        invocation_dispatch=invocation_dispatch,
        reliability_aware_invocation_dispatch=invocation_dispatch,
        reliability_evidence_observer=reliability_evidence_observer,
        provider_capabilities=capabilities,
        clock=resolved_clock,
    )
    orchestrator = GovernedExternalWorkOrchestrator(
        adapter=adapter,
        policy=policy_evaluator,
        bundle=bundle,
        attestor=attestor,
        capabilities=capabilities,
        execution_store=execution_store,
        receipt_store=receipt_store,
        bundle_store=bundle_store,
        continuation_store=continuation_store,
        provider_invocation_store=provider_invocation_store,
        clock=resolved_clock,
        reliability_bridge=(
            reliability_bridge
            or GovernedExternalWorkEnterpriseReliabilityBridge.production()
        ),
        reliability_evidence_observer=reliability_evidence_observer,
    )
    return GovernedExternalWorkProductionRuntime(
        adapter=adapter,
        orchestrator=orchestrator,
        authorization_boundary=authorization_boundary,
        decision_requirement_policy=resolved_policy,
        policy_bundle=bundle,
        policy_evaluator=policy_evaluator,
    )


__all__ = [
    "GovernedExternalWorkProductionRuntime",
    "build_governed_external_work_production_runtime",
    "resolve_production_collaborative_work_repositories",
    "resolve_production_decision_requirement_policy",
    "resolve_production_runtime_policy_bundle",
    "resolve_production_runtime_policy_bundle_evaluator",
]
