# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "MILVUS_VECTOR_STORE_PROVIDER_ID",
    "MilvusVectorStoreIntegration",
    "MilvusVectorStoreIntegrationConfig",
    "MilvusVectorStoreClient",
    "create_milvus_vector_store",
    "create_milvus_vector_store_integration",
    "register_milvus_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_milvus_vector_store",
        "create_milvus_vector_store_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "MILVUS_VECTOR_STORE_PROVIDER_ID",
        "MilvusVectorStoreIntegration",
        "MilvusVectorStoreIntegrationConfig",
        "MilvusVectorStoreClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "MILVUS_VECTOR_STORE_PROVIDER_ID",
        "MilvusVectorStoreIntegration",
        "MilvusVectorStoreIntegrationConfig",
        "MilvusVectorStoreClient",
    }
)

def __getattr__(name: str):
    if name == "register_milvus_integration":
        from intergrax.integrations.providers.vector_store.milvus.register import register_milvus_integration

        return register_milvus_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.vector_store.milvus import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.vector_store.milvus import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.vector_store.milvus import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
