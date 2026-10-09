# © Artur Czarnecki. All rights reserved.

"""Production composition for Marketplace qualified capability execution (TRACE-X-P5-R2-P3-R1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications._shared.integrations.persistence import (
    wire_execution_integration_configuration_pinning_store,
)
from intergrax.contracts.execution_bound_catalog_tool_invocation import (
    ExecutionBoundCatalogToolInvoker,
)
from intergrax.contracts.runtime_execution_policy_admission import (
    RuntimeExecutionPolicyAdmissionPort,
)
from intergrax.contracts.tools.marketplace_qualified_capability import (
    MarketplaceQualifiedToolStageRepository,
)
from intergrax.contracts.tools.qualified_marketplace_tool_execution_intent import (
    QualifiedMarketplaceToolExecutionIntentRepository,
)
from intergrax.contracts.tools.qualified_tool_invocation import (
    QualifiedToolInvocationMaterialProvider,
    QualifiedToolInvocationResolver,
)
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.integrations.configured_relational_store_execution_binding import (
    build_default_configured_relational_store_execution_binding,
)
from intergrax.integrations.contracts.document_store import DocumentStore
from intergrax.integrations.contracts.execution_integration_configuration_pinning import (
    ExecutionIntegrationConfigurationPinningStore,
)
from intergrax.integrations.execution_bound_integration_resolution import (
    ExecutionBoundIntegrationMaterializationPort,
    ExecutionBoundIntegrationResolution,
)
from intergrax.autonomous_work.execution_authority_admission import (
    WorkerExecutionAdmissionPort,
)
from intergrax.autonomous_work.worker_configured_capability_execution_fulfillment_service import (
    WorkerConfiguredCapabilityExecutionFulfillmentService,
)
from intergrax.runtime.execution.execution_bound_capability_execution_dispatch_service import (
    ExecutionBoundCapabilityExecutionDispatchService,
)
from intergrax.runtime.execution.qualified_capability_execution_composition import (
    build_execution_bound_capability_execution_dispatch_service,
    build_qualified_capability_execution_dispatch_service,
)
from intergrax.runtime.execution.worker_configured_capability_execution_adapter import (
    WorkerConfiguredCapabilityExecutionEngineAdapter,
)
from intergrax.tools.marketplace_configured_capability_binding_provider import (
    MarketplaceConfiguredCapabilityBindingProvider,
)
from intergrax.tools.marketplace_configured_tool_execution_intent_preparation import (
    MarketplaceConfiguredToolExecutionIntentPreparation,
)
from intergrax.runtime.execution.qualified_capability_execution_dispatch_service import (
    QualifiedCapabilityExecutionDispatchService,
)
from intergrax.runtime.execution.qualified_capability_execution_handlers import (
    QualifiedCapabilityExecutionBindingHandlerRegistry,
)
from intergrax.runtime.execution.qualified_capability_execution_runtime_delegate import (
    QualifiedCapabilityExecutionRuntimeDelegate,
)
from intergrax.tools.configured_integration_tool_invocation_projection import (
    DefaultConfiguredIntegrationToolInvocationProjectionPort,
)
from intergrax.tools.dynamic_acquisition import (
    DynamicToolAcquisitionPort,
    ToolHostActivationPort,
)
from intergrax.tools.marketplace_qualified_capability_execution_composition import (
    build_marketplace_tool_qualified_capability_execution_handler,
)
from intergrax.tools.marketplace_qualified_capability_execution_handler import (
    MarketplaceToolQualifiedCapabilityExecutionHandler,
)
from intergrax.tools.known_capability_realization import ToolPackageResolutionForIdentityPort
from intergrax.tools.qualified_marketplace_tool_activation_resolver import (
    QualifiedMarketplaceToolActivationResolver,
)
from intergrax.tools.qualified_tool_invocation_resolver import (
    DefaultQualifiedToolInvocationResolver,
)


class Uca6cMarketplaceQualifiedExecutionCompositionError(RuntimeError):
    """Fail closed when production Marketplace configured execution cannot be wired."""


def _require_exactly_one_pinning_backing(
    *,
    configuration_pinning_kv_store: DistributedKVStore | None,
    configuration_pinning_document_store: DocumentStore | None,
) -> None:
    if (
        configuration_pinning_kv_store is None
        and configuration_pinning_document_store is None
    ):
        raise Uca6cMarketplaceQualifiedExecutionCompositionError(
            "production marketplace qualified execution requires durable "
            "configuration_pinning_kv_store or configuration_pinning_document_store",
        )
    if (
        configuration_pinning_kv_store is not None
        and configuration_pinning_document_store is not None
    ):
        raise Uca6cMarketplaceQualifiedExecutionCompositionError(
            "production marketplace qualified execution accepts exactly one "
            "configuration pinning backing store",
        )


@dataclass(frozen=True, slots=True)
class ProductionMarketplaceConfiguredExecutionComposition:
    """Sanctioned cross-layer assembly — exposes pinning for composition evidence."""

    handler: MarketplaceToolQualifiedCapabilityExecutionHandler
    pinning_store: ExecutionIntegrationConfigurationPinningStore
    resolution: ExecutionBoundIntegrationResolution


def build_production_marketplace_configured_execution_composition(
    *,
    intent_repository: QualifiedMarketplaceToolExecutionIntentRepository,
    stage_repository: MarketplaceQualifiedToolStageRepository,
    activation_read: ToolHostActivationPort,
    acquisition: DynamicToolAcquisitionPort,
    host_profile_id: str,
    material_provider: QualifiedToolInvocationMaterialProvider,
    catalog_tool_invoker: ExecutionBoundCatalogToolInvoker,
    configuration_pinning_kv_store: DistributedKVStore | None = None,
    configuration_pinning_document_store: DocumentStore | None = None,
    invocation_resolver: QualifiedToolInvocationResolver | None = None,
    materialization: ExecutionBoundIntegrationMaterializationPort | None = None,
    activation_resolver: QualifiedMarketplaceToolActivationResolver | None = None,
    package_resolver: ToolPackageResolutionForIdentityPort | None = None,
) -> ProductionMarketplaceConfiguredExecutionComposition:
    """Wire durable P2 pinning → Pattern A → relational binding → configured projection → handler."""
    _require_exactly_one_pinning_backing(
        configuration_pinning_kv_store=configuration_pinning_kv_store,
        configuration_pinning_document_store=configuration_pinning_document_store,
    )
    pinning_store = wire_execution_integration_configuration_pinning_store(
        kv_store=configuration_pinning_kv_store,
        document_store=configuration_pinning_document_store,
    )
    resolution = ExecutionBoundIntegrationResolution(
        pinning_store=pinning_store,
        materialization=materialization,
    )
    relational_binding = build_default_configured_relational_store_execution_binding(
        resolution=resolution,
    )
    projection = DefaultConfiguredIntegrationToolInvocationProjectionPort(
        relational_binding=relational_binding,
    )
    if activation_resolver is None:
        handler = build_marketplace_tool_qualified_capability_execution_handler(
            intent_repository=intent_repository,
            stage_repository=stage_repository,
            activation_read=activation_read,
            acquisition=acquisition,
            host_profile_id=host_profile_id,
            material_provider=material_provider,
            catalog_tool_invoker=catalog_tool_invoker,
            invocation_resolver=invocation_resolver,
            configured_invocation_projection=projection,
            package_resolver=package_resolver,
        )
    else:
        handler = MarketplaceToolQualifiedCapabilityExecutionHandler(
            intent_repository=intent_repository,
            stage_repository=stage_repository,
            activation_resolver=activation_resolver,
            material_provider=material_provider,
            invocation_resolver=invocation_resolver
            or DefaultQualifiedToolInvocationResolver(),
            catalog_tool_invoker=catalog_tool_invoker,
            configured_invocation_projection=projection,
            package_resolver=package_resolver,
        )
    return ProductionMarketplaceConfiguredExecutionComposition(
        handler=handler,
        pinning_store=pinning_store,
        resolution=resolution,
    )


def build_production_marketplace_qualified_capability_execution_handler(
    *,
    intent_repository: QualifiedMarketplaceToolExecutionIntentRepository,
    stage_repository: MarketplaceQualifiedToolStageRepository,
    activation_read: ToolHostActivationPort,
    acquisition: DynamicToolAcquisitionPort,
    host_profile_id: str,
    material_provider: QualifiedToolInvocationMaterialProvider,
    catalog_tool_invoker: ExecutionBoundCatalogToolInvoker,
    configuration_pinning_kv_store: DistributedKVStore | None = None,
    configuration_pinning_document_store: DocumentStore | None = None,
    invocation_resolver: QualifiedToolInvocationResolver | None = None,
    materialization: ExecutionBoundIntegrationMaterializationPort | None = None,
    activation_resolver: QualifiedMarketplaceToolActivationResolver | None = None,
) -> MarketplaceToolQualifiedCapabilityExecutionHandler:
    return build_production_marketplace_configured_execution_composition(
        intent_repository=intent_repository,
        stage_repository=stage_repository,
        activation_read=activation_read,
        acquisition=acquisition,
        host_profile_id=host_profile_id,
        material_provider=material_provider,
        catalog_tool_invoker=catalog_tool_invoker,
        configuration_pinning_kv_store=configuration_pinning_kv_store,
        configuration_pinning_document_store=configuration_pinning_document_store,
        invocation_resolver=invocation_resolver,
        materialization=materialization,
        activation_resolver=activation_resolver,
    ).handler


def build_production_marketplace_configured_execution_bound_dispatch(
    *,
    composition: ProductionMarketplaceConfiguredExecutionComposition,
    runtime_policy_admission: RuntimeExecutionPolicyAdmissionPort,
) -> tuple[
    ExecutionBoundCapabilityExecutionDispatchService,
    object,
]:
    """Shared ExecutionBound ingress for CONFIGURE_EXISTING — same handler registry."""
    registry = QualifiedCapabilityExecutionBindingHandlerRegistry(
        (composition.handler,),
    )
    dispatch, delegate, _launcher = build_execution_bound_capability_execution_dispatch_service(
        handler_registry=registry,
        runtime_policy_admission=runtime_policy_admission,
    )
    return dispatch, delegate


def build_production_marketplace_configured_execution_fulfillment(
    *,
    intent_repository: QualifiedMarketplaceToolExecutionIntentRepository,
    execution_bound_dispatch: ExecutionBoundCapabilityExecutionDispatchService,
    authority_admission: WorkerExecutionAdmissionPort | None = None,
) -> WorkerConfiguredCapabilityExecutionFulfillmentService:
    """Wire configured binding, intent preparation, and ExecutionBound execution."""
    execution = WorkerConfiguredCapabilityExecutionEngineAdapter(
        dispatch=execution_bound_dispatch,
    )
    return WorkerConfiguredCapabilityExecutionFulfillmentService(
        binding=MarketplaceConfiguredCapabilityBindingProvider(),
        intent_preparation=MarketplaceConfiguredToolExecutionIntentPreparation(
            intent_repository=intent_repository,
        ),
        execution=execution,
        authority_admission=authority_admission,
    )


def build_production_marketplace_qualified_capability_execution_dispatch(
    *,
    composition: ProductionMarketplaceConfiguredExecutionComposition,
    runtime_policy_admission: RuntimeExecutionPolicyAdmissionPort,
) -> tuple[
    QualifiedCapabilityExecutionDispatchService,
    QualifiedCapabilityExecutionRuntimeDelegate,
]:
    """Register the configured-capable Marketplace handler in canonical qualified dispatch."""
    registry = QualifiedCapabilityExecutionBindingHandlerRegistry(
        (composition.handler,),
    )
    dispatch, delegate, _launcher = build_qualified_capability_execution_dispatch_service(
        handler_registry=registry,
        runtime_policy_admission=runtime_policy_admission,
    )
    return dispatch, delegate


__all__ = [
    "ProductionMarketplaceConfiguredExecutionComposition",
    "Uca6cMarketplaceQualifiedExecutionCompositionError",
    "build_production_marketplace_configured_execution_bound_dispatch",
    "build_production_marketplace_configured_execution_composition",
    "build_production_marketplace_configured_execution_fulfillment",
    "build_production_marketplace_qualified_capability_execution_dispatch",
    "build_production_marketplace_qualified_capability_execution_handler",
]
