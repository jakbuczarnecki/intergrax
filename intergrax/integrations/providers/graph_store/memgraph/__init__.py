# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "MEMGRAPH_GRAPH_STORE_PROVIDER_ID",
    "MemgraphGraphStoreIntegration",
    "MemgraphGraphStoreIntegrationConfig",
    "MemgraphGraphStoreClient",
    "create_memgraph_graph_store",
    "create_memgraph_graph_store_integration",
    "register_memgraph_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_memgraph_graph_store",
        "create_memgraph_graph_store_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "MEMGRAPH_GRAPH_STORE_PROVIDER_ID",
        "MemgraphGraphStoreIntegration",
        "MemgraphGraphStoreIntegrationConfig",
        "MemgraphGraphStoreClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "MEMGRAPH_GRAPH_STORE_PROVIDER_ID",
        "MemgraphGraphStoreIntegration",
        "MemgraphGraphStoreIntegrationConfig",
        "MemgraphGraphStoreClient",
    }
)

def __getattr__(name: str):
    if name == "register_memgraph_integration":
        from intergrax.integrations.providers.graph_store.memgraph.register import register_memgraph_integration

        return register_memgraph_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.graph_store.memgraph import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.graph_store.memgraph import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.graph_store.memgraph import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
