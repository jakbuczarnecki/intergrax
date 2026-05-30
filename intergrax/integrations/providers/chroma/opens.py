# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Low-level Chroma openers — internal to the chroma integration package.

Only this module may import ``chromadb`` before constructing the RAG store.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.vector_store import VectorStore
from intergrax.integrations.providers.chroma.adapter import ChromaVectorStoreIntegration
from intergrax.integrations.providers.chroma.config import ChromaIntegrationConfig


def _import_chromadb() -> Any:
    try:
        import chromadb
    except ImportError as exc:
        raise IntegrationConfigurationError(
            "Chroma integration requires chromadb. "
            "Install with: uv sync  (chromadb is a main project dependency)"
        ) from exc
    return chromadb


def _build_rag_config(config: ChromaIntegrationConfig) -> Any:
    from intergrax.rag.vectorstore.providers.chroma_vector_store import ChromaConfig

    return ChromaConfig(
        collection_name=config.collection_name,
        tenant_id=config.tenant_id,
        persist_directory=config.persist_directory,
        batch_size=config.batch_size,
        metric=config.metric,
        mode=config.mode,
        http_host=config.http_host,
        http_port=config.http_port,
    )


def _open_rag_store(
    config: ChromaIntegrationConfig,
    *,
    store_factory: Optional[Callable[[], VectorStore]] = None,
) -> VectorStore:
    if store_factory is not None:
        return store_factory()
    _import_chromadb()
    from intergrax.rag.vectorstore.providers.chroma_vector_store import ChromaVectorStore

    return ChromaVectorStore(_build_rag_config(config))


def open_chroma_vector_store(
    config: ChromaIntegrationConfig,
    *,
    implementation: Optional[VectorStore] = None,
    store: Optional[VectorStore] = None,
    store_factory: Optional[Callable[[], VectorStore]] = None,
) -> VectorStore:
    if implementation is not None:
        return implementation
    inner = store if store is not None else _open_rag_store(config, store_factory=store_factory)
    return ChromaVectorStoreIntegration(config, inner)
