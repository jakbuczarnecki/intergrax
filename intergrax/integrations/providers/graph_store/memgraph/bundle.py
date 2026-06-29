# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p5.factories import create_memgraph_graph_store as _legacy_create_memgraph_graph_store

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.graph_store.memgraph.integration import (
    MEMGRAPH_GRAPH_STORE_PROVIDER_ID,
    MemgraphGraphStoreIntegration,
    MemgraphGraphStoreIntegrationConfig,
    MemgraphGraphStoreClient,
)

__all__ = [
    "create_memgraph_graph_store",
    "create_memgraph_graph_store_integration",
]


def create_memgraph_graph_store_integration(
    *,
    client: MemgraphGraphStoreClient | None = None,
    enabled: bool = False,
) -> MemgraphGraphStoreIntegration:
    """
    Build a contract-based Memgraph graph store integration.

    The legacy facade (create_memgraph_graph_store) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Memgraph graph store integration requires an injected client when enabled=True",
        )
    if client is not None:
        return MemgraphGraphStoreIntegration.from_client(client, enabled=enabled)
    return MemgraphGraphStoreIntegration.for_provider(
        provider_id=MEMGRAPH_GRAPH_STORE_PROVIDER_ID,
        display_name="Memgraph",
        config=MemgraphGraphStoreIntegrationConfig(enabled=enabled),
    )


def create_memgraph_graph_store(**kwargs: object) -> MemgraphGraphStoreIntegration:
    """Compatibility shim — constructs MemgraphGraphStoreIntegration from legacy runtime."""
    runtime = _legacy_create_memgraph_graph_store(**kwargs)
    if isinstance(runtime, MemgraphGraphStoreIntegration):
        return runtime
    return MemgraphGraphStoreIntegration.from_client(runtime)
