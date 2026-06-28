# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "LANCEDB_VECTOR_STORE_PROVIDER_ID",
    "LancedbVectorStoreIntegration",
    "LancedbVectorStoreIntegrationConfig",
    "LancedbVectorStoreClient",
    "create_lancedb_vector_store",
    "create_lancedb_vector_store_integration",
    "register_lancedb_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_lancedb_vector_store",
        "create_lancedb_vector_store_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "LANCEDB_VECTOR_STORE_PROVIDER_ID",
        "LancedbVectorStoreIntegration",
        "LancedbVectorStoreIntegrationConfig",
        "LancedbVectorStoreClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "LANCEDB_VECTOR_STORE_PROVIDER_ID",
        "LancedbVectorStoreIntegration",
        "LancedbVectorStoreIntegrationConfig",
        "LancedbVectorStoreClient",
    }
)

def __getattr__(name: str):
    if name == "register_lancedb_integration":
        from intergrax.integrations.providers.vector_store.lancedb.register import register_lancedb_integration

        return register_lancedb_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.vector_store.lancedb import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.vector_store.lancedb import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.vector_store.lancedb import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
