# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_duckdb_relational_store", "register_duckdb_integration"]

def __getattr__(name: str):
    if name == "register_duckdb_integration":
        from intergrax.integrations.providers.relational_store.duckdb.register import register_duckdb_integration
        return register_duckdb_integration
    if name == "create_duckdb_relational_store":
        from intergrax.integrations.providers.relational_store.duckdb.bundle import create_duckdb_relational_store
        return create_duckdb_relational_store
    raise AttributeError(name)
