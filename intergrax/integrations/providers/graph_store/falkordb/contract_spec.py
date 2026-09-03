# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Falkordb graph store."""

from __future__ import annotations

from intergrax.integrations.providers.graph_store.falkordb.bundle import (
    create_falkordb_graph_store_integration,
)
from intergrax.integrations.providers.graph_store.falkordb.integration import (
    FALKORDB_GRAPH_STORE_PROVIDER_ID,
    FalkordbGraphStoreIntegration,
    FalkordbGraphStoreIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.data import GraphStoreIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="graph_store",
    provider_id=FALKORDB_GRAPH_STORE_PROVIDER_ID,
    integration_class=FalkordbGraphStoreIntegration,
    contract_class=GraphStoreIntegrationContract,
    contract_factory=create_falkordb_graph_store_integration,
    display_name="Falkordb",
    config_class=FalkordbGraphStoreIntegrationConfig,
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
