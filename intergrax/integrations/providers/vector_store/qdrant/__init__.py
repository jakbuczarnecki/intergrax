# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Qdrant vector store integration (Phase M.6 P2)."""

from intergrax.utils.lazy_export import export_from_bundle
from intergrax.integrations.providers.vector_store.qdrant.config import (
    ENV_QDRANT_API_KEY,
    ENV_QDRANT_COLLECTION,
    ENV_QDRANT_TENANT_ID,
    ENV_QDRANT_URL,
    QdrantIntegrationConfig,
)

__all__ = [
    "ENV_QDRANT_API_KEY",
    "ENV_QDRANT_COLLECTION",
    "ENV_QDRANT_TENANT_ID",
    "ENV_QDRANT_URL",
    "QdrantIntegrationBundle",
    "QdrantIntegrationConfig",
    "QdrantVectorStoreIntegration",
    "create_qdrant_integration",
    "create_qdrant_vector_store",
    "register_qdrant_integration",
    "resolve_qdrant_config",
    "create_qdrant_vector_store_integration",
]

_LAZY_EXPORTS = frozenset(
    {
        "QdrantIntegrationBundle",
        "create_qdrant_integration",
        "create_qdrant_vector_store",
        "create_qdrant_vector_store_integration",
        "register_qdrant_integration",
        "resolve_qdrant_config",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "QDRANT_VECTOR_STORE_PROVIDER_ID",
        "QdrantVectorStoreIntegration",
        "QdrantVectorStoreIntegrationConfig",
        "QdrantVectorStoreClient",
    }
)

def __getattr__(name: str):
    if name == "register_qdrant_integration":
        from intergrax.integrations.providers.vector_store.qdrant.register import register_qdrant_integration

        return register_qdrant_integration
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.vector_store.qdrant import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)
    if name in _LAZY_EXPORTS:
        from intergrax.integrations.providers.vector_store.qdrant import bundle as _bundle

        return export_from_bundle(_bundle, name, _LAZY_EXPORTS)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
