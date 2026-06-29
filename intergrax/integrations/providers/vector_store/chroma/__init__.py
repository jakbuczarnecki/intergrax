# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Chroma vector store integration (Phase M.6 P2)."""

from intergrax.utils.lazy_export import export_from_bundle
from intergrax.integrations.providers.vector_store.chroma.config import (
    ENV_CHROMA_COLLECTION,
    ENV_CHROMA_MODE,
    ENV_CHROMA_PERSIST_DIRECTORY,
    ENV_CHROMA_TENANT_ID,
    ChromaIntegrationConfig,
)

__all__ = [
    "ENV_CHROMA_COLLECTION",
    "ENV_CHROMA_MODE",
    "ENV_CHROMA_PERSIST_DIRECTORY",
    "ENV_CHROMA_TENANT_ID",
    "ChromaIntegrationBundle",
    "ChromaIntegrationConfig",
    "ChromaVectorStoreIntegration",
    "create_chroma_integration",
    "create_chroma_vector_store",
    "register_chroma_integration",
    "resolve_chroma_config",
    "create_chroma_vector_store_integration",
]

_LAZY_EXPORTS = frozenset(
    {
        "ChromaIntegrationBundle",
        "create_chroma_integration",
        "create_chroma_vector_store",
        "create_chroma_vector_store_integration",
        "register_chroma_integration",
        "resolve_chroma_config",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "CHROMA_VECTOR_STORE_PROVIDER_ID",
        "ChromaVectorStoreIntegration",
        "ChromaVectorStoreIntegrationConfig",
        "ChromaVectorStoreClient",
    }
)

def __getattr__(name: str):
    if name == "register_chroma_integration":
        from intergrax.integrations.providers.vector_store.chroma.register import register_chroma_integration

        return register_chroma_integration
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.vector_store.chroma import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)
    if name in _LAZY_EXPORTS:
        from intergrax.integrations.providers.vector_store.chroma import bundle as _bundle

        return export_from_bundle(_bundle, name, _LAZY_EXPORTS)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
