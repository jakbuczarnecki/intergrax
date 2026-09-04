# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Neo4J graph store."""

from __future__ import annotations

from intergrax.integrations.providers.graph_store.neo4j.bundle import (
    create_neo4j_graph_store_integration,
)
from intergrax.integrations.providers.graph_store.neo4j.integration import (
    NEO4J_GRAPH_STORE_PROVIDER_ID,
    Neo4jGraphStoreIntegration,
    Neo4jGraphStoreIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.data import GraphStoreIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="graph_store",
    provider_id=NEO4J_GRAPH_STORE_PROVIDER_ID,
    integration_class=Neo4jGraphStoreIntegration,
    contract_class=GraphStoreIntegrationContract,
    contract_factory=create_neo4j_graph_store_integration,
    display_name="Neo4J",
    config_class=Neo4jGraphStoreIntegrationConfig,
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
