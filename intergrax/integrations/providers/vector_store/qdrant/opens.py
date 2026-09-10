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
from intergrax.integrations.contracts.vector_index_metadata import VectorIndexMetadataReader
from intergrax.integrations.contracts.vector_store import VectorStore
from intergrax.integrations.providers.vector_store.qdrant.metadata_reader import (
    QdrantVectorIndexMetadataReader,
)
from intergrax.integrations.providers.vector_store.qdrant.index_administration import (
    QdrantControlPlaneClient,
    QdrantVectorIndexAdministration,
)

from intergrax.integrations.providers.vector_store.qdrant.client_factory import (
    build_qdrant_control_plane_client,
)
from intergrax.integrations.providers.vector_store.qdrant.integration import QdrantVectorStoreIntegration
from intergrax.integrations.providers.vector_store.qdrant.config import QdrantIntegrationConfig


def _import_qdrant_client() -> Any:
    from intergrax.integrations.providers.vector_store.qdrant.client_factory import _import_qdrant_client as _load

    return _load()


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


def open_qdrant_control_plane_client(
    config: QdrantIntegrationConfig,
) -> QdrantControlPlaneClient:
    """Public opener for the Qdrant control-plane client."""
    return build_qdrant_control_plane_client(config)


def open_qdrant_vector_index_administration(
    config: QdrantIntegrationConfig,
) -> VectorIndexAdministration:
    """Public opener for Qdrant vector index administration (control plane)."""
    client = build_qdrant_control_plane_client(config)
    return QdrantVectorIndexAdministration(_client=client, _config=config)


def open_qdrant_vector_index_metadata_reader(
    config: QdrantIntegrationConfig,
) -> VectorIndexMetadataReader:
    """Public opener for Qdrant vector index metadata reads."""
    client = build_qdrant_control_plane_client(config)
    return QdrantVectorIndexMetadataReader(_client=client, _config=config)


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


__all__ = [
    "open_qdrant_control_plane_client",
    "open_qdrant_vector_index_administration",
    "open_qdrant_vector_index_metadata_reader",
    "open_qdrant_vector_store",
]
