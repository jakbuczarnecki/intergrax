# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Low-level Qdrant openers — internal to the qdrant integration package.

Only this module may import ``qdrant_client`` before constructing the RAG store.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.vector_store import VectorStore
from intergrax.integrations.providers.vector_store.qdrant.adapter import QdrantVectorStoreIntegration
from intergrax.integrations.providers.vector_store.qdrant.config import QdrantIntegrationConfig


def _import_qdrant_client() -> Any:
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise IntegrationConfigurationError(
            "Qdrant integration requires qdrant-client. "
            "Install with: uv sync  (qdrant-client is a main project dependency)"
        ) from exc
    return QdrantClient


def _build_rag_config(config: QdrantIntegrationConfig) -> Any:
    from intergrax.integrations.providers.vector_store.qdrant.rag_store import QdrantConfig

    url = config.resolved_url()
    sparse_raw = os.getenv("INTERGRAX_RAG_QDRANT_SPARSE", "").strip().lower()
    enable_sparse = sparse_raw in ("1", "true", "yes", "on")
    return QdrantConfig(
        collection_name=config.collection_name,
        tenant_id=config.tenant_id,
        metric=config.metric,
        batch_size=config.batch_size,
        qdrant_url=url,
        qdrant_api_key=config.api_key or None,
        enable_sparse_vectors=enable_sparse,
    )


def _open_rag_store(
    config: QdrantIntegrationConfig,
    *,
    store_factory: Optional[Callable[[], VectorStore]] = None,
) -> VectorStore:
    if store_factory is not None:
        return store_factory()
    _import_qdrant_client()
    from intergrax.integrations.providers.vector_store.qdrant.rag_store import QdrantVectorStore

    return QdrantVectorStore(_build_rag_config(config))


def open_qdrant_vector_store(
    config: QdrantIntegrationConfig,
    *,
    implementation: Optional[VectorStore] = None,
    store: Optional[VectorStore] = None,
    store_factory: Optional[Callable[[], VectorStore]] = None,
) -> VectorStore:
    if implementation is not None:
        return implementation
    inner = store if store is not None else _open_rag_store(config, store_factory=store_factory)
    return QdrantVectorStoreIntegration(config, inner)
