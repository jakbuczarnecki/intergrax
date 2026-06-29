# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "BIGQUERY_RELATIONAL_STORE_PROVIDER_ID",
    "BigqueryRelationalStoreIntegration",
    "BigqueryRelationalStoreIntegrationConfig",
    "BigqueryRelationalStoreClient",
    "create_bigquery_relational_store",
    "create_bigquery_relational_store_integration",
    "register_bigquery_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_bigquery_relational_store",
        "create_bigquery_relational_store_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "BIGQUERY_RELATIONAL_STORE_PROVIDER_ID",
        "BigqueryRelationalStoreIntegration",
        "BigqueryRelationalStoreIntegrationConfig",
        "BigqueryRelationalStoreClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "BIGQUERY_RELATIONAL_STORE_PROVIDER_ID",
        "BigqueryRelationalStoreIntegration",
        "BigqueryRelationalStoreIntegrationConfig",
        "BigqueryRelationalStoreClient",
    }
)

def __getattr__(name: str):
    if name == "register_bigquery_integration":
        from intergrax.integrations.providers.relational_store.bigquery.register import register_bigquery_integration

        return register_bigquery_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.relational_store.bigquery import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.relational_store.bigquery import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.relational_store.bigquery import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
