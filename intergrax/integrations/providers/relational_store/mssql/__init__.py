# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "MSSQL_RELATIONAL_STORE_PROVIDER_ID",
    "MssqlRelationalStoreIntegration",
    "MssqlRelationalStoreIntegrationConfig",
    "MssqlRelationalStoreClient",
    "create_mssql_relational_store",
    "create_mssql_relational_store_integration",
    "register_mssql_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_mssql_relational_store",
        "create_mssql_relational_store_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "MSSQL_RELATIONAL_STORE_PROVIDER_ID",
        "MssqlRelationalStoreIntegration",
        "MssqlRelationalStoreIntegrationConfig",
        "MssqlRelationalStoreClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "MSSQL_RELATIONAL_STORE_PROVIDER_ID",
        "MssqlRelationalStoreIntegration",
        "MssqlRelationalStoreIntegrationConfig",
        "MssqlRelationalStoreClient",
    }
)

def __getattr__(name: str):
    if name == "register_mssql_integration":
        from intergrax.integrations.providers.relational_store.mssql.register import register_mssql_integration

        return register_mssql_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.relational_store.mssql import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.relational_store.mssql import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.relational_store.mssql import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
