# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p3.factories import create_neo4j_graph_store

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.graph_store.neo4j.integration import (
    NEO4J_GRAPH_STORE_PROVIDER_ID,
    Neo4jGraphStoreIntegration,
    Neo4jGraphStoreIntegrationConfig,
    Neo4jGraphStoreClient,
)

__all__ = [
    "create_neo4j_graph_store",
    "create_neo4j_graph_store_integration",
]


def create_neo4j_graph_store_integration(
    *,
    client: Neo4jGraphStoreClient | None = None,
    enabled: bool = False,
) -> Neo4jGraphStoreIntegration:
    """
    Build a contract-based Neo4J graph store integration.

    The legacy facade (create_neo4j_graph_store) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Neo4J graph store integration requires an injected client when enabled=True",
        )
    if client is not None:
        return Neo4jGraphStoreIntegration.from_client(client, enabled=enabled)
    return Neo4jGraphStoreIntegration.for_provider(
        provider_id=NEO4J_GRAPH_STORE_PROVIDER_ID,
        display_name="Neo4J",
        config=Neo4jGraphStoreIntegrationConfig(enabled=enabled),
    )
