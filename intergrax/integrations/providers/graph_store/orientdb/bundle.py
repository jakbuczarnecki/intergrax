# © Artur Czarnecki. All rights reserved.

from intergrax.integrations._shared.p5.factories import create_orientdb_graph_store as _legacy_create_orientdb_graph_store

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.graph_store.orientdb.integration import (
    ORIENTDB_GRAPH_STORE_PROVIDER_ID,
    OrientDbGraphStoreClient,
    OrientDbGraphStoreIntegration,
    OrientDbGraphStoreIntegrationConfig,
)

__all__ = [
    "create_orientdb_graph_store",
    "create_orientdb_graph_store_integration",
]


def create_orientdb_graph_store_integration(
    *,
    client: OrientDbGraphStoreClient | None = None,
    enabled: bool = False,
) -> OrientDbGraphStoreIntegration:
    """
    Build a contract-based OrientDB graph store integration.

    The legacy facade (create_orientdb_graph_store) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "OrientDB graph store integration requires an injected client when enabled=True",
        )
    if client is not None:
        return OrientDbGraphStoreIntegration.from_client(client, enabled=enabled)
    return OrientDbGraphStoreIntegration.for_provider(
        provider_id=ORIENTDB_GRAPH_STORE_PROVIDER_ID,
        display_name="OrientDB",
        config=OrientDbGraphStoreIntegrationConfig(enabled=enabled),
    )


def create_orientdb_graph_store(**kwargs: object) -> OrientDbGraphStoreIntegration:
    """Compatibility shim — constructs OrientDbGraphStoreIntegration from legacy runtime."""
    runtime = _legacy_create_orientdb_graph_store(**kwargs)
    if isinstance(runtime, OrientDbGraphStoreIntegration):
        return runtime
    return OrientDbGraphStoreIntegration.from_client(runtime)
