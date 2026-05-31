# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_mssql_relational_store", "register_mssql_integration"]

def __getattr__(name: str):
    if name == "register_mssql_integration":
        from intergrax.integrations.providers.relational_store.mssql.register import register_mssql_integration
        return register_mssql_integration
    if name == "create_mssql_relational_store":
        from intergrax.integrations.providers.relational_store.mssql.bundle import create_mssql_relational_store
        return create_mssql_relational_store
    raise AttributeError(name)
