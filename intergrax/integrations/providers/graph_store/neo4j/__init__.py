# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "NEO4J_GRAPH_STORE_PROVIDER_ID",
    "Neo4jGraphStoreIntegration",
    "Neo4jGraphStoreIntegrationConfig",
    "Neo4jGraphStoreClient",
    "create_neo4j_graph_store",
    "create_neo4j_graph_store_integration",
    "register_neo4j_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_neo4j_graph_store",
        "create_neo4j_graph_store_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "NEO4J_GRAPH_STORE_PROVIDER_ID",
        "Neo4jGraphStoreIntegration",
        "Neo4jGraphStoreIntegrationConfig",
        "Neo4jGraphStoreClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "NEO4J_GRAPH_STORE_PROVIDER_ID",
        "Neo4jGraphStoreIntegration",
        "Neo4jGraphStoreIntegrationConfig",
        "Neo4jGraphStoreClient",
    }
)

def __getattr__(name: str):
    if name == "register_neo4j_integration":
        from intergrax.integrations.providers.graph_store.neo4j.register import register_neo4j_integration

        return register_neo4j_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.graph_store.neo4j import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.graph_store.neo4j import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.graph_store.neo4j import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
