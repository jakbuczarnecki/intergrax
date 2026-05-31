# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Qdrant vector store integration (Phase M.6 P2)."""

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
]

_LAZY_EXPORTS = frozenset(
    {
        "QdrantIntegrationBundle",
        "QdrantVectorStoreIntegration",
        "create_qdrant_integration",
        "create_qdrant_vector_store",
        "register_qdrant_integration",
        "resolve_qdrant_config",
    }
)


def __getattr__(name: str):
    if name == "register_qdrant_integration":
        from intergrax.integrations.providers.vector_store.qdrant.register import register_qdrant_integration

        return register_qdrant_integration
    if name in _LAZY_EXPORTS:
        from intergrax.integrations.providers.vector_store.qdrant import bundle as _bundle

        return getattr(_bundle, name)
    if name == "QdrantVectorStoreIntegration":
        from intergrax.integrations.providers.vector_store.qdrant.adapter import QdrantVectorStoreIntegration

        return QdrantVectorStoreIntegration
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
