# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_snowflake_relational_store", "register_snowflake_integration"]

def __getattr__(name: str):
    if name == "register_snowflake_integration":
        from intergrax.integrations.providers.relational_store.snowflake.register import register_snowflake_integration
        return register_snowflake_integration
    if name == "create_snowflake_relational_store":
        from intergrax.integrations.providers.relational_store.snowflake.bundle import create_snowflake_relational_store
        return create_snowflake_relational_store
    raise AttributeError(name)
