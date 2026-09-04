# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for OrientDB graph store."""

from __future__ import annotations

from intergrax.integrations.providers.graph_store.orientdb.bundle import (
    create_orientdb_graph_store_integration,
)
from intergrax.integrations.providers.graph_store.orientdb.integration import (
    ORIENTDB_GRAPH_STORE_PROVIDER_ID,
    OrientDbGraphStoreIntegration,
    OrientDbGraphStoreIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.data import GraphStoreIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="graph_store",
    provider_id=ORIENTDB_GRAPH_STORE_PROVIDER_ID,
    integration_class=OrientDbGraphStoreIntegration,
    contract_class=GraphStoreIntegrationContract,
    contract_factory=create_orientdb_graph_store_integration,
    display_name="OrientDB",
    config_class=OrientDbGraphStoreIntegrationConfig,
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
