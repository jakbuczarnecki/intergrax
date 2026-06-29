# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "WEAVIATE_VECTOR_STORE_PROVIDER_ID",
    "WeaviateVectorStoreIntegration",
    "WeaviateVectorStoreIntegrationConfig",
    "WeaviateVectorStoreClient",
    "create_weaviate_vector_store",
    "create_weaviate_vector_store_integration",
    "register_weaviate_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_weaviate_vector_store",
        "create_weaviate_vector_store_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "WEAVIATE_VECTOR_STORE_PROVIDER_ID",
        "WeaviateVectorStoreIntegration",
        "WeaviateVectorStoreIntegrationConfig",
        "WeaviateVectorStoreClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "WEAVIATE_VECTOR_STORE_PROVIDER_ID",
        "WeaviateVectorStoreIntegration",
        "WeaviateVectorStoreIntegrationConfig",
        "WeaviateVectorStoreClient",
    }
)

def __getattr__(name: str):
    if name == "register_weaviate_integration":
        from intergrax.integrations.providers.vector_store.weaviate.register import register_weaviate_integration

        return register_weaviate_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.vector_store.weaviate import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.vector_store.weaviate import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.vector_store.weaviate import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
