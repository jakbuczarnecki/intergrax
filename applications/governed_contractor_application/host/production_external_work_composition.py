# © Artur Czarnecki. All rights reserved.

"""GR-6-WIRE — production Decision → Governance → External Work composition root."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone

from external_contractor_adapter.external_work_adapter import ExternalWorkAdapter
from governed_contractor_application.host.collaborative_work_boundary import (
    build_external_work_authorization_boundary,
    default_external_work_decision_requirement_policy,
)
from governed_contractor_application.host.offline_demo import build_demo_policy_bundle
from governed_contractor_application.host.orchestrator import GovernedExternalWorkOrchestrator
from governed_contractor_application.host.settings import GovernedContractorBackendSettings
from governed_contractor_application.host.stores import (
    InMemoryContinuationStateStore,
    InMemoryGovernedExecutionStore,
    InMemoryPolicyBundleArtifactStore,
    InMemoryProofReceiptStore,
)
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

_PRODUCTION_POLICY_ISSUED_AT = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)


@dataclass(frozen=True, slots=True)
class GovernedExternalWorkProductionRuntime:
    """Injectable production stack for governed external-work lifecycle."""

    adapter: ExternalWorkAdapter
    orchestrator: GovernedExternalWorkOrchestrator
    authorization_boundary: MeaningfulSideEffectAuthorizationBoundary
    decision_requirement_policy: DecisionRequirementPolicy
    policy_bundle: ImmutableRuntimePolicyBundle
    policy_evaluator: RuntimePolicyBundleEvaluator


def resolve_production_runtime_policy_bundle(
    settings: GovernedContractorBackendSettings,
) -> ImmutableRuntimePolicyBundle:
    bundle = settings.runtime_policy_bundle
    if bundle is None:
        return build_demo_policy_bundle(issued_at=_PRODUCTION_POLICY_ISSUED_AT)
    if not isinstance(bundle, ImmutableRuntimePolicyBundle):
        raise TypeError(
            "settings.runtime_policy_bundle must be ImmutableRuntimePolicyBundle",
        )
    return bundle


def resolve_production_decision_requirement_policy(
    settings: GovernedContractorBackendSettings,
) -> DecisionRequirementPolicy:
    policy = settings.decision_requirement_policy
    if policy is None:
        return default_external_work_decision_requirement_policy()
    return policy


def build_governed_external_work_production_runtime(
    integration: ExternalWorkIntegration,
    *,
    tenant_id: str,
    workspace_id: str,
    principal_id: str,
    task_scope: ActiveExecutionTaskScopePort,
    capabilities: ExternalWorkProviderCapabilities,
    decision_requirement_policy: DecisionRequirementPolicy | None = None,
    policy_bundle: ImmutableRuntimePolicyBundle | None = None,
    attestor: HostAttestor | None = None,
    clock: Callable[[], datetime] | None = None,
) -> GovernedExternalWorkProductionRuntime:
    """Construct orchestrator + adapter wired through canonical governance boundary."""
    bundle = policy_bundle or build_demo_policy_bundle(issued_at=_PRODUCTION_POLICY_ISSUED_AT)
    resolved_policy = (
        decision_requirement_policy
        if decision_requirement_policy is not None
        else default_external_work_decision_requirement_policy()
    )
    policy_evaluator = RuntimePolicyBundleEvaluator(
        bundle,
        clock=clock or (lambda: _PRODUCTION_POLICY_ISSUED_AT),
    )
    authorization_boundary = build_external_work_authorization_boundary(
        policy_evaluator,
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        principal_id=principal_id,
        decision_requirement_policy=resolved_policy,
        task_scope=task_scope,
    )
    adapter = ExternalWorkAdapter(
        integration,
        authorization_boundary=authorization_boundary,
    )
    orchestrator = GovernedExternalWorkOrchestrator(
        adapter=adapter,
        policy=policy_evaluator,
        bundle=bundle,
        attestor=attestor,
        capabilities=capabilities,
        execution_store=InMemoryGovernedExecutionStore(),
        receipt_store=InMemoryProofReceiptStore(),
        bundle_store=InMemoryPolicyBundleArtifactStore(),
        continuation_store=InMemoryContinuationStateStore(),
        clock=clock or (lambda: _PRODUCTION_POLICY_ISSUED_AT),
    )
    return GovernedExternalWorkProductionRuntime(
        adapter=adapter,
        orchestrator=orchestrator,
        authorization_boundary=authorization_boundary,
        decision_requirement_policy=resolved_policy,
        policy_bundle=bundle,
        policy_evaluator=policy_evaluator,
    )


def wire_governed_contractor_production_external_work_settings(
    settings: GovernedContractorBackendSettings,
    *,
    integration: ExternalWorkIntegration | None = None,
    task_scope: ActiveExecutionTaskScopePort | None = None,
    tenant_id: str = "tenant-a",
    workspace_id: str = "workspace-a",
    principal_id: str = "u1",
) -> GovernedContractorBackendSettings:
    """Apply production meaningful-side-effect boundary slots on host settings."""
    resolved_integration = integration
    if resolved_integration is None:
        raw = settings.external_work_integration
        if raw is not None and isinstance(raw, ExternalWorkIntegration):
            resolved_integration = raw
    if resolved_integration is None:
        return settings
    if (
        settings.meaningful_side_effect_authorization_boundary is not None
        and settings.external_work_integration is resolved_integration
    ):
        return settings

    bundle = resolve_production_runtime_policy_bundle(settings)
    decision_policy = resolve_production_decision_requirement_policy(settings)
    policy_evaluator = RuntimePolicyBundleEvaluator(bundle)
    boundary = build_external_work_authorization_boundary(
        policy_evaluator,
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        principal_id=principal_id,
        decision_requirement_policy=decision_policy,
        task_scope=task_scope,
    )
    return replace(
        settings,
        external_work_integration=resolved_integration,
        meaningful_side_effect_authorization_boundary=boundary,
        runtime_policy_bundle=bundle,
        decision_requirement_policy=decision_policy,
    )


__all__ = [
    "GovernedExternalWorkProductionRuntime",
    "build_governed_external_work_production_runtime",
    "resolve_production_decision_requirement_policy",
    "resolve_production_runtime_policy_bundle",
    "wire_governed_contractor_production_external_work_settings",
]
