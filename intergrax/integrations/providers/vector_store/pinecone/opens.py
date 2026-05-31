# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Low-level Pinecone openers — internal to the pinecone integration package.

Only this module may import the Pinecone SDK before constructing the RAG store.
All composition roots use ``bundle.create_pinecone_*`` or
``profile.resolve(VECTOR_STORE)``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.vector_store import VectorStore
from intergrax.integrations.providers.vector_store.pinecone.adapter import PineconeVectorStoreIntegration
from intergrax.integrations.providers.vector_store.pinecone.config import PineconeIntegrationConfig


def _import_pinecone() -> Any:
    try:
        from pinecone import Pinecone
    except ImportError as exc:
        raise IntegrationConfigurationError(
            "Pinecone integration requires the pinecone package. "
            "Install with: uv sync  (pinecone is a main project dependency)"
        ) from exc
    return Pinecone


def _build_rag_config(config: PineconeIntegrationConfig) -> Any:
    from intergrax.integrations.providers.vector_store.pinecone.rag_store import PineconeConfig

    if not config.api_key:
        raise IntegrationConfigurationError(
            "Pinecone api_key is required (INTERGRAX_PINECONE_API_KEY)"
        )
    return PineconeConfig(
        collection_name=config.collection_name,
        tenant_id=config.tenant_id,
        metric=config.metric,
        batch_size=config.batch_size,
        pinecone_api_key=config.api_key,
        pinecone_index_name=config.resolved_index_name(),
        pinecone_cloud=config.cloud,
        pinecone_region=config.region,
    )


def _open_rag_store(
    config: PineconeIntegrationConfig,
    *,
    store_factory: Optional[Callable[[], VectorStore]] = None,
) -> VectorStore:
    if store_factory is not None:
        return store_factory()
    _import_pinecone()
    from intergrax.integrations.providers.vector_store.pinecone.rag_store import PineconeVectorStore

    rag_config = _build_rag_config(config)
    return PineconeVectorStore(rag_config)


def open_pinecone_vector_store(
    config: PineconeIntegrationConfig,
    *,
    implementation: Optional[VectorStore] = None,
    store: Optional[VectorStore] = None,
    store_factory: Optional[Callable[[], VectorStore]] = None,
) -> VectorStore:
    if implementation is not None:
        return implementation
    if store is not None:
        inner = store
    else:
        inner = _open_rag_store(config, store_factory=store_factory)
    return PineconeVectorStoreIntegration(config, inner)
