# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for ArangoDB graph store."""

from __future__ import annotations

from intergrax.integrations.providers.graph_store.arangodb.bundle import (
    create_arangodb_graph_store_integration,
)
from intergrax.integrations.providers.graph_store.arangodb.integration import (
    ARANGODB_GRAPH_STORE_PROVIDER_ID,
    ArangoDbGraphStoreIntegration,
    ArangoDbGraphStoreIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.data import GraphStoreIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="graph_store",
    provider_id=ARANGODB_GRAPH_STORE_PROVIDER_ID,
    integration_class=ArangoDbGraphStoreIntegration,
    contract_class=GraphStoreIntegrationContract,
    contract_factory=create_arangodb_graph_store_integration,
    display_name="ArangoDB",
    config_class=ArangoDbGraphStoreIntegrationConfig,
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
