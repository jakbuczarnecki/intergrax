# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Memgraph graph store."""

from __future__ import annotations

from intergrax.integrations.providers.graph_store.memgraph.bundle import (
    create_memgraph_graph_store_integration,
)
from intergrax.integrations.providers.graph_store.memgraph.integration import (
    MEMGRAPH_GRAPH_STORE_PROVIDER_ID,
    MemgraphGraphStoreIntegration,
    MemgraphGraphStoreIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.data import GraphStoreIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="graph_store",
    provider_id=MEMGRAPH_GRAPH_STORE_PROVIDER_ID,
    integration_class=MemgraphGraphStoreIntegration,
    contract_class=GraphStoreIntegrationContract,
    contract_factory=create_memgraph_graph_store_integration,
    display_name="Memgraph",
    config_class=MemgraphGraphStoreIntegrationConfig,
    capabilities=(
        PlatformIntegrationCapability.CONNECT,
        PlatformIntegrationCapability.READ,
        PlatformIntegrationCapability.WRITE,
        PlatformIntegrationCapability.HEALTH_CHECK,
    ),
    security_posture=PlatformIntegrationSecurityPosture(),
    supports_runtime_binding=True,
    supports_health_check=True,
    metadata={"source": "explicit_provider_declaration"},
)

CONTRACT_SPECS = (CONTRACT_SPEC,)

__all__ = ["CONTRACT_SPEC", "CONTRACT_SPECS"]
