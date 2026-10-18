# © Artur Czarnecki. All rights reserved.

"""Strict production host wiring for sanctioned external-work operation port."""

from __future__ import annotations

from dataclasses import dataclass

from governed_contractor_application.host.governed_contractor_host_runtime_composition import (
    GovernedContractorHostRuntimeComposition,
    resolve_production_runtime_policy_bundle_from_settings,
)
from governed_contractor_application.host.governed_external_work_host_stores import (
    GovernedExternalWorkHostStoreBundle,
    validate_production_governed_external_work_host_store_bundle,
)
from governed_contractor_application.host.production_external_work_composition import (
    GovernedExternalWorkProductionRuntime,
    build_governed_external_work_production_runtime,
    resolve_production_decision_requirement_policy,
)
from governed_contractor_application.host.sanctioned_external_work_operation_port import (
    GovernedSanctionedExternalWorkOperationPort,
)
from governed_contractor_application.host.settings import GovernedContractorBackendSettings
from intergrax.contracts.external_work_provider_capabilities import (
    ExternalWorkProviderCapabilities,
)
from intergrax.contracts.provider_invocation_store import ProviderInvocationStore
from intergrax.contracts.sanctioned_external_work_operation import (
    SanctionedExternalWorkOperationPort,
)
from intergrax.integrations.contracts.external_work import ExternalWorkIntegration


@dataclass(slots=True)
class GovernedExternalWorkHostSessionResources:
    """One process-scoped external-work stack — reused across strict host requests."""

    production_runtime: GovernedExternalWorkProductionRuntime
    operation_port: SanctionedExternalWorkOperationPort
    host_stores: GovernedExternalWorkHostStoreBundle


def build_strict_external_work_host_session_resources(
    integration: ExternalWorkIntegration,
    *,
    settings: GovernedContractorBackendSettings,
    host_runtime: GovernedContractorHostRuntimeComposition,
    provider_invocation_store: ProviderInvocationStore,
    execution_tenant_id: str,
    workspace_id: str,
    principal_id: str,
    host_stores: GovernedExternalWorkHostStoreBundle,
    capabilities: ExternalWorkProviderCapabilities,
) -> GovernedExternalWorkHostSessionResources:
    if host_runtime.collaborative_work_repositories is None:
        raise ValueError(
            "strict external work host wiring requires collaborative_work_repositories",
        )
    if host_runtime.active_execution_task_scope is None:
        raise ValueError(
            "strict external work host wiring requires active_execution_task_scope",
        )
    if not workspace_id.strip():
        raise ValueError("strict external work host wiring requires workspace_id")
    if not principal_id.strip():
        raise ValueError("strict external work host wiring requires principal_id")
    validate_production_governed_external_work_host_store_bundle(host_stores)
    policy_bundle = resolve_production_runtime_policy_bundle_from_settings(settings)
    production_runtime = build_governed_external_work_production_runtime(
        integration,
        tenant_id=execution_tenant_id,
        workspace_id=workspace_id,
        principal_id=principal_id,
        task_scope=host_runtime.active_execution_task_scope,
        capabilities=capabilities,
        decision_requirement_policy=resolve_production_decision_requirement_policy(
            host_runtime,
        ),
        policy_bundle=policy_bundle,
        collaborative_work_repositories=host_runtime.collaborative_work_repositories,
        execution_store=host_stores.execution_store,
        receipt_store=host_stores.receipt_store,
        bundle_store=host_stores.bundle_store,
        continuation_store=host_stores.continuation_store,
        provider_invocation_store=provider_invocation_store,
        attestor=host_runtime.host_attestor,
    )
    operation_port = GovernedSanctionedExternalWorkOperationPort(
        production_runtime.orchestrator,
    )
    return GovernedExternalWorkHostSessionResources(
        production_runtime=production_runtime,
        operation_port=operation_port,
        host_stores=host_stores,
    )


def attach_sanctioned_external_work_to_host_runtime(
    host_runtime: GovernedContractorHostRuntimeComposition,
    *,
    operation_port: SanctionedExternalWorkOperationPort,
) -> GovernedContractorHostRuntimeComposition:
    return GovernedContractorHostRuntimeComposition(
        external_work_integration=host_runtime.external_work_integration,
        meaningful_side_effect_authorization_boundary=(
            host_runtime.meaningful_side_effect_authorization_boundary
        ),
        decision_requirement_policy=host_runtime.decision_requirement_policy,
        collaborative_work_repositories=host_runtime.collaborative_work_repositories,
        active_execution_task_scope=host_runtime.active_execution_task_scope,
        host_attestor=host_runtime.host_attestor,
        sanctioned_external_work_operation_port=operation_port,
        require_sanctioned_external_work_mutations=True,
    )


__all__ = [
    "GovernedExternalWorkHostSessionResources",
    "attach_sanctioned_external_work_to_host_runtime",
    "build_strict_external_work_host_session_resources",
]
