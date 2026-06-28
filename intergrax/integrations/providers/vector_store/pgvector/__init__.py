# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "PGVECTOR_VECTOR_STORE_PROVIDER_ID",
    "PgvectorVectorStoreIntegration",
    "PgvectorVectorStoreIntegrationConfig",
    "PgvectorVectorStoreClient",
    "create_pgvector_vector_store",
    "create_pgvector_vector_store_integration",
    "register_pgvector_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_pgvector_vector_store",
        "create_pgvector_vector_store_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "PGVECTOR_VECTOR_STORE_PROVIDER_ID",
        "PgvectorVectorStoreIntegration",
        "PgvectorVectorStoreIntegrationConfig",
        "PgvectorVectorStoreClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "PGVECTOR_VECTOR_STORE_PROVIDER_ID",
        "PgvectorVectorStoreIntegration",
        "PgvectorVectorStoreIntegrationConfig",
        "PgvectorVectorStoreClient",
    }
)

def __getattr__(name: str):
    if name == "register_pgvector_integration":
        from intergrax.integrations.providers.vector_store.pgvector.register import register_pgvector_integration

        return register_pgvector_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.vector_store.pgvector import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.vector_store.pgvector import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.vector_store.pgvector import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
