# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Complete Pinecone integration bundle — catalog bridge to ``intergrax/rag/``.

The Pinecone SDK is imported only in ``opens.py``. Tier-3 code MUST use
``create_pinecone_vector_store()``, ``create_pinecone_integration()``, or
``profile.resolve(IntegrationCategory.VECTOR_STORE)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from intergrax.integrations.contracts.vector_store import VectorStore
from intergrax.integrations.providers.vector_store.pinecone.adapter import PineconeVectorStoreIntegration
from intergrax.integrations.providers.vector_store.pinecone.config import PineconeIntegrationConfig
from intergrax.integrations.providers.vector_store.pinecone.opens import open_pinecone_vector_store


@dataclass(frozen=True)
class PineconeIntegrationBundle:
    config: PineconeIntegrationConfig
    vector_store: PineconeVectorStoreIntegration


def resolve_pinecone_config(**overrides: object) -> PineconeIntegrationConfig:
    return PineconeIntegrationConfig.from_env(**overrides)


def create_pinecone_integration(
    *,
    vector_store: Optional[VectorStore] = None,
    store: Optional[object] = None,
    store_factory: Optional[Callable[[], object]] = None,
    **config_overrides: object,
) -> PineconeIntegrationBundle:
    config = resolve_pinecone_config(**config_overrides)
    store_impl = open_pinecone_vector_store(
        config,
        implementation=vector_store,
        store=store,
        store_factory=store_factory,
    )
    assert isinstance(store_impl, PineconeVectorStoreIntegration)
    return PineconeIntegrationBundle(config=config, vector_store=store_impl)


def create_pinecone_vector_store(
    *,
    vector_store: Optional[VectorStore] = None,
    store: Optional[object] = None,
    store_factory: Optional[Callable[[], object]] = None,
    **config_overrides: object,
) -> PineconeVectorStoreIntegration:
    """Catalog factory for ``IntegrationSlug.PINECONE`` / ``VECTOR_STORE``."""
    return create_pinecone_integration(
        vector_store=vector_store,
        store=store,
        store_factory=store_factory,
        **config_overrides,
    ).vector_store
