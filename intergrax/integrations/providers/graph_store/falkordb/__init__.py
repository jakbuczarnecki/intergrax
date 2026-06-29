# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "FALKORDB_GRAPH_STORE_PROVIDER_ID",
    "FalkordbGraphStoreIntegration",
    "FalkordbGraphStoreIntegrationConfig",
    "FalkordbGraphStoreClient",
    "create_falkordb_graph_store",
    "create_falkordb_graph_store_integration",
    "register_falkordb_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_falkordb_graph_store",
        "create_falkordb_graph_store_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "FALKORDB_GRAPH_STORE_PROVIDER_ID",
        "FalkordbGraphStoreIntegration",
        "FalkordbGraphStoreIntegrationConfig",
        "FalkordbGraphStoreClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "FALKORDB_GRAPH_STORE_PROVIDER_ID",
        "FalkordbGraphStoreIntegration",
        "FalkordbGraphStoreIntegrationConfig",
        "FalkordbGraphStoreClient",
    }
)

def __getattr__(name: str):
    if name == "register_falkordb_integration":
        from intergrax.integrations.providers.graph_store.falkordb.register import register_falkordb_integration

        return register_falkordb_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.graph_store.falkordb import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.graph_store.falkordb import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.graph_store.falkordb import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
