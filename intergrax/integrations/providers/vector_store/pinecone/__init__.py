# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Pinecone vector store integration — catalog bridge to ``intergrax/rag/`` (Phase M.6 P2)."""

from intergrax.integrations.providers.vector_store.pinecone.config import (
    ENV_PINECONE_API_KEY,
    ENV_PINECONE_COLLECTION,
    ENV_PINECONE_INDEX,
    ENV_PINECONE_TENANT_ID,
    PineconeIntegrationConfig,
)

__all__ = [
    "ENV_PINECONE_API_KEY",
    "ENV_PINECONE_COLLECTION",
    "ENV_PINECONE_INDEX",
    "ENV_PINECONE_TENANT_ID",
    "PineconeIntegrationBundle",
    "PineconeIntegrationConfig",
    "PineconeVectorStoreIntegration",
    "create_pinecone_integration",
    "create_pinecone_vector_store",
    "register_pinecone_integration",
    "resolve_pinecone_config",
]

_LAZY_EXPORTS = frozenset(
    {
        "PineconeIntegrationBundle",
        "PineconeVectorStoreIntegration",
        "create_pinecone_integration",
        "create_pinecone_vector_store",
        "register_pinecone_integration",
        "resolve_pinecone_config",
    }
)


def __getattr__(name: str):
    if name == "register_pinecone_integration":
        from intergrax.integrations.providers.vector_store.pinecone.register import register_pinecone_integration

        return register_pinecone_integration
    if name in _LAZY_EXPORTS:
        from intergrax.integrations.providers.vector_store.pinecone import bundle as _bundle

        return getattr(_bundle, name)
    if name == "PineconeVectorStoreIntegration":
        from intergrax.integrations.providers.vector_store.pinecone.adapter import PineconeVectorStoreIntegration

        return PineconeVectorStoreIntegration
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
