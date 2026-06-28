# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "AZURE_SQL_RELATIONAL_STORE_PROVIDER_ID",
    "AzureSqlRelationalStoreIntegration",
    "AzureSqlRelationalStoreIntegrationConfig",
    "AzureSqlRelationalStoreClient",
    "create_azure_sql_relational_store",
    "create_azure_sql_relational_store_integration",
    "register_azure_sql_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_azure_sql_relational_store",
        "create_azure_sql_relational_store_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "AZURE_SQL_RELATIONAL_STORE_PROVIDER_ID",
        "AzureSqlRelationalStoreIntegration",
        "AzureSqlRelationalStoreIntegrationConfig",
        "AzureSqlRelationalStoreClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "AZURE_SQL_RELATIONAL_STORE_PROVIDER_ID",
        "AzureSqlRelationalStoreIntegration",
        "AzureSqlRelationalStoreIntegrationConfig",
        "AzureSqlRelationalStoreClient",
    }
)

def __getattr__(name: str):
    if name == "register_azure_sql_integration":
        from intergrax.integrations.providers.relational_store.azure_sql.register import register_azure_sql_integration

        return register_azure_sql_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.relational_store.azure_sql import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.relational_store.azure_sql import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.relational_store.azure_sql import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
