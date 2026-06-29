# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "DUCKDB_RELATIONAL_STORE_PROVIDER_ID",
    "DuckdbRelationalStoreIntegration",
    "DuckdbRelationalStoreIntegrationConfig",
    "DuckdbRelationalStoreClient",
    "create_duckdb_relational_store",
    "create_duckdb_relational_store_integration",
    "register_duckdb_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_duckdb_relational_store",
        "create_duckdb_relational_store_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "DUCKDB_RELATIONAL_STORE_PROVIDER_ID",
        "DuckdbRelationalStoreIntegration",
        "DuckdbRelationalStoreIntegrationConfig",
        "DuckdbRelationalStoreClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "DUCKDB_RELATIONAL_STORE_PROVIDER_ID",
        "DuckdbRelationalStoreIntegration",
        "DuckdbRelationalStoreIntegrationConfig",
        "DuckdbRelationalStoreClient",
    }
)

def __getattr__(name: str):
    if name == "register_duckdb_integration":
        from intergrax.integrations.providers.relational_store.duckdb.register import register_duckdb_integration

        return register_duckdb_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.relational_store.duckdb import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.relational_store.duckdb import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.relational_store.duckdb import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
