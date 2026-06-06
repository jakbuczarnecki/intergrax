# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_memgraph_graph_store", "register_memgraph_integration"]

def __getattr__(name: str):
    if name == "register_memgraph_integration":
        from intergrax.integrations.providers.graph_store.memgraph.register import register_memgraph_integration
        return register_memgraph_integration
    if name == "create_memgraph_graph_store":
        from intergrax.integrations.providers.graph_store.memgraph.bundle import create_memgraph_graph_store
        return create_memgraph_graph_store
    raise AttributeError(name)
