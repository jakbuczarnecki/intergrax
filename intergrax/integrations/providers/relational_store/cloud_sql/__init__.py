# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_cloud_sql_relational_store", "register_cloud_sql_integration"]

def __getattr__(name: str):
    if name == "register_cloud_sql_integration":
        from intergrax.integrations.providers.relational_store.cloud_sql.register import register_cloud_sql_integration
        return register_cloud_sql_integration
    if name == "create_cloud_sql_relational_store":
        from intergrax.integrations.providers.relational_store.cloud_sql.bundle import create_cloud_sql_relational_store
        return create_cloud_sql_relational_store
    raise AttributeError(name)
