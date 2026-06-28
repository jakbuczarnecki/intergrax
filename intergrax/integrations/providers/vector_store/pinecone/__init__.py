# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Pinecone vector store integration — catalog bridge to ``intergrax/rag/`` (Phase M.6 P2)."""

from intergrax.utils.lazy_export import export_from_bundle
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
    "create_pinecone_vector_store_integration",
]

_LAZY_EXPORTS = frozenset(
    {
        "PineconeIntegrationBundle",
        "create_pinecone_integration",
        "create_pinecone_vector_store",
        "create_pinecone_vector_store_integration",
        "register_pinecone_integration",
        "resolve_pinecone_config",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "PINECONE_VECTOR_STORE_PROVIDER_ID",
        "PineconeVectorStoreIntegration",
        "PineconeVectorStoreIntegrationConfig",
        "PineconeVectorStoreClient",
    }
)

def __getattr__(name: str):
    if name == "register_pinecone_integration":
        from intergrax.integrations.providers.vector_store.pinecone.register import register_pinecone_integration

        return register_pinecone_integration
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.vector_store.pinecone import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)
    if name in _LAZY_EXPORTS:
        from intergrax.integrations.providers.vector_store.pinecone import bundle as _bundle

        return export_from_bundle(_bundle, name, _LAZY_EXPORTS)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
