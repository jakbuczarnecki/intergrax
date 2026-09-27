# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Composition for Marketplace qualified Tool EE execution handler (S24-GAP-02-P3)."""

from __future__ import annotations

from intergrax.contracts.execution_bound_catalog_tool_invocation import (
    ExecutionBoundCatalogToolInvoker,
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
from intergrax.tools.dynamic_acquisition import (
    DynamicToolAcquisitionPort,
    ToolHostActivationPort,
)
from intergrax.tools.marketplace_qualified_capability_execution_handler import (
    MarketplaceToolQualifiedCapabilityExecutionHandler,
)
from intergrax.tools.qualified_marketplace_tool_activation_resolver import (
    QualifiedMarketplaceToolActivationResolver,
)
from intergrax.tools.qualified_tool_invocation_resolver import (
    DefaultQualifiedToolInvocationResolver,
)


def build_marketplace_tool_qualified_capability_execution_handler(
    *,
    intent_repository: QualifiedMarketplaceToolExecutionIntentRepository,
    stage_repository: MarketplaceQualifiedToolStageRepository,
    activation_read: ToolHostActivationPort,
    acquisition: DynamicToolAcquisitionPort,
    host_profile_id: str,
    material_provider: QualifiedToolInvocationMaterialProvider,
    catalog_tool_invoker: ExecutionBoundCatalogToolInvoker,
    invocation_resolver: QualifiedToolInvocationResolver | None = None,
) -> MarketplaceToolQualifiedCapabilityExecutionHandler:
    activation_resolver = QualifiedMarketplaceToolActivationResolver(
        activation_read=activation_read,
        acquisition=acquisition,
        host_profile_id=host_profile_id,
    )
    return MarketplaceToolQualifiedCapabilityExecutionHandler(
        intent_repository=intent_repository,
        stage_repository=stage_repository,
        activation_resolver=activation_resolver,
        material_provider=material_provider,
        invocation_resolver=invocation_resolver or DefaultQualifiedToolInvocationResolver(),
        catalog_tool_invoker=catalog_tool_invoker,
    )


__all__ = ["build_marketplace_tool_qualified_capability_execution_handler"]
