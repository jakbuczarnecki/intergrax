# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Low-level Qdrant openers — internal to the qdrant integration package.

Only this module may import ``qdrant_client`` before constructing the RAG store.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.vector_index_administration import VectorIndexAdministration
from intergrax.integrations.contracts.vector_store import VectorStore
from intergrax.integrations.providers.vector_store.qdrant.index_administration import (
    QdrantControlPlaneClient,
    QdrantVectorIndexAdministration,
)
from intergrax.integrations.providers.vector_store.qdrant.integration import QdrantVectorStoreIntegration
from intergrax.integrations.providers.vector_store.qdrant.config import QdrantIntegrationConfig


def _import_qdrant_client() -> Any:
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise IntegrationConfigurationError(
            "Qdrant integration requires qdrant-client. "
            "Install with: Intergrax-ai[vector-qdrant]."
        ) from exc
    return QdrantClient


def _build_rag_config(config: QdrantIntegrationConfig) -> Any:
    from intergrax.integrations.providers.vector_store.qdrant.rag_store import QdrantConfig

    url = config.resolved_url()
    return QdrantConfig(
        collection_name=config.collection_name,
        tenant_id=config.tenant_id,
        metric=config.metric,
        batch_size=config.batch_size,
        qdrant_url=url,
        qdrant_api_key=config.api_key or None,
        enable_sparse_vectors=config.enable_sparse_vectors,
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


def _build_qdrant_client(config: QdrantIntegrationConfig) -> QdrantControlPlaneClient:
    QdrantClient = _import_qdrant_client()
    if config.resolved_url():
        return QdrantClient(url=config.resolved_url(), api_key=config.api_key or None)
    return QdrantClient(host=config.host, port=config.port, api_key=config.api_key or None)


def open_qdrant_vector_index_administration(
    config: QdrantIntegrationConfig,
) -> VectorIndexAdministration:
    """Public opener for Qdrant vector index administration (control plane)."""
    client = _build_qdrant_client(config)
    return QdrantVectorIndexAdministration(_client=client, _config=config)


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
    return QdrantVectorStoreIntegration.from_store(config, inner)
