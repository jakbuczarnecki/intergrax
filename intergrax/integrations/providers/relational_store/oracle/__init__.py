# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "ORACLE_RELATIONAL_STORE_PROVIDER_ID",
    "OracleRelationalStoreIntegration",
    "OracleRelationalStoreIntegrationConfig",
    "OracleRelationalStoreClient",
    "create_oracle_relational_store",
    "create_oracle_relational_store_integration",
    "register_oracle_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_oracle_relational_store",
        "create_oracle_relational_store_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "ORACLE_RELATIONAL_STORE_PROVIDER_ID",
        "OracleRelationalStoreIntegration",
        "OracleRelationalStoreIntegrationConfig",
        "OracleRelationalStoreClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "ORACLE_RELATIONAL_STORE_PROVIDER_ID",
        "OracleRelationalStoreIntegration",
        "OracleRelationalStoreIntegrationConfig",
        "OracleRelationalStoreClient",
    }
)

def __getattr__(name: str):
    if name == "register_oracle_integration":
        from intergrax.integrations.providers.relational_store.oracle.register import register_oracle_integration

        return register_oracle_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.relational_store.oracle import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.relational_store.oracle import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.relational_store.oracle import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
