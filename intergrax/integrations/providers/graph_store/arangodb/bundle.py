# © Artur Czarnecki. All rights reserved.

from intergrax.integrations._shared.p5.factories import create_arangodb_graph_store as _legacy_create_arangodb_graph_store

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.graph_store.arangodb.integration import (
    ARANGODB_GRAPH_STORE_PROVIDER_ID,
    ArangoDbGraphStoreClient,
    ArangoDbGraphStoreIntegration,
    ArangoDbGraphStoreIntegrationConfig,
)

__all__ = [
    "create_arangodb_graph_store",
    "create_arangodb_graph_store_integration",
]


def create_arangodb_graph_store_integration(
    *,
    client: ArangoDbGraphStoreClient | None = None,
    enabled: bool = False,
) -> ArangoDbGraphStoreIntegration:
    """
    Build a contract-based ArangoDB graph store integration.

    The legacy facade (create_arangodb_graph_store) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "ArangoDB graph store integration requires an injected client when enabled=True",
        )
    if client is not None:
        return ArangoDbGraphStoreIntegration.from_client(client, enabled=enabled)
    return ArangoDbGraphStoreIntegration.for_provider(
        provider_id=ARANGODB_GRAPH_STORE_PROVIDER_ID,
        display_name="ArangoDB",
        config=ArangoDbGraphStoreIntegrationConfig(enabled=enabled),
    )


def create_arangodb_graph_store(**kwargs: object) -> ArangoDbGraphStoreIntegration:
    """Compatibility shim — constructs ArangoDbGraphStoreIntegration from legacy runtime."""
    runtime = _legacy_create_arangodb_graph_store(**kwargs)
    if isinstance(runtime, ArangoDbGraphStoreIntegration):
        return runtime
    return ArangoDbGraphStoreIntegration.from_client(runtime)
