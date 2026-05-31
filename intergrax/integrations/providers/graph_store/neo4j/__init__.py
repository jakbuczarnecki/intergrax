# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_neo4j_graph_store", "register_neo4j_integration"]

def __getattr__(name: str):
    if name == "register_neo4j_integration":
        from intergrax.integrations.providers.graph_store.neo4j.register import register_neo4j_integration
        return register_neo4j_integration
    if name == "create_neo4j_graph_store":
        from intergrax.integrations.providers.graph_store.neo4j.bundle import create_neo4j_graph_store
        return create_neo4j_graph_store
    raise AttributeError(name)
