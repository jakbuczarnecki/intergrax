# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "SNOWFLAKE_RELATIONAL_STORE_PROVIDER_ID",
    "SnowflakeRelationalStoreIntegration",
    "SnowflakeRelationalStoreIntegrationConfig",
    "SnowflakeRelationalStoreClient",
    "create_snowflake_relational_store",
    "create_snowflake_relational_store_integration",
    "register_snowflake_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_snowflake_relational_store",
        "create_snowflake_relational_store_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "SNOWFLAKE_RELATIONAL_STORE_PROVIDER_ID",
        "SnowflakeRelationalStoreIntegration",
        "SnowflakeRelationalStoreIntegrationConfig",
        "SnowflakeRelationalStoreClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "SNOWFLAKE_RELATIONAL_STORE_PROVIDER_ID",
        "SnowflakeRelationalStoreIntegration",
        "SnowflakeRelationalStoreIntegrationConfig",
        "SnowflakeRelationalStoreClient",
    }
)

def __getattr__(name: str):
    if name == "register_snowflake_integration":
        from intergrax.integrations.providers.relational_store.snowflake.register import register_snowflake_integration

        return register_snowflake_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.relational_store.snowflake import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.relational_store.snowflake import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.relational_store.snowflake import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
