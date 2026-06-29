# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p5.factories import create_falkordb_graph_store as _legacy_create_falkordb_graph_store

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.graph_store.falkordb.integration import (
    FALKORDB_GRAPH_STORE_PROVIDER_ID,
    FalkordbGraphStoreIntegration,
    FalkordbGraphStoreIntegrationConfig,
    FalkordbGraphStoreClient,
)

__all__ = [
    "create_falkordb_graph_store",
    "create_falkordb_graph_store_integration",
]


def create_falkordb_graph_store_integration(
    *,
    client: FalkordbGraphStoreClient | None = None,
    enabled: bool = False,
) -> FalkordbGraphStoreIntegration:
    """
    Build a contract-based Falkordb graph store integration.

    The legacy facade (create_falkordb_graph_store) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Falkordb graph store integration requires an injected client when enabled=True",
        )
    if client is not None:
        return FalkordbGraphStoreIntegration.from_client(client, enabled=enabled)
    return FalkordbGraphStoreIntegration.for_provider(
        provider_id=FALKORDB_GRAPH_STORE_PROVIDER_ID,
        display_name="Falkordb",
        config=FalkordbGraphStoreIntegrationConfig(enabled=enabled),
    )


def create_falkordb_graph_store(**kwargs: object) -> FalkordbGraphStoreIntegration:
    """Compatibility shim — constructs FalkordbGraphStoreIntegration from legacy runtime."""
    runtime = _legacy_create_falkordb_graph_store(**kwargs)
    if isinstance(runtime, FalkordbGraphStoreIntegration):
        return runtime
    return FalkordbGraphStoreIntegration.from_runtime(runtime)
