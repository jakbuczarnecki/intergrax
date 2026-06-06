# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_falkordb_graph_store", "register_falkordb_integration"]

def __getattr__(name: str):
    if name == "register_falkordb_integration":
        from intergrax.integrations.providers.graph_store.falkordb.register import register_falkordb_integration
        return register_falkordb_integration
    if name == "create_falkordb_graph_store":
        from intergrax.integrations.providers.graph_store.falkordb.bundle import create_falkordb_graph_store
        return create_falkordb_graph_store
    raise AttributeError(name)
