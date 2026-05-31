# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_azure_sql_relational_store", "register_azure_sql_integration"]

def __getattr__(name: str):
    if name == "register_azure_sql_integration":
        from intergrax.integrations.providers.relational_store.azure_sql.register import register_azure_sql_integration
        return register_azure_sql_integration
    if name == "create_azure_sql_relational_store":
        from intergrax.integrations.providers.relational_store.azure_sql.bundle import create_azure_sql_relational_store
        return create_azure_sql_relational_store
    raise AttributeError(name)
