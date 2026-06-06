# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_timescaledb_relational_store", "register_timescaledb_integration"]

def __getattr__(name: str):
    if name == "register_timescaledb_integration":
        from intergrax.integrations.providers.relational_store.timescaledb.register import register_timescaledb_integration
        return register_timescaledb_integration
    if name == "create_timescaledb_relational_store":
        from intergrax.integrations.providers.relational_store.timescaledb.bundle import create_timescaledb_relational_store
        return create_timescaledb_relational_store
    raise AttributeError(name)
