# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Complete Qdrant integration bundle — catalog bridge to ``intergrax/rag/``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from intergrax.integrations.contracts.vector_store import VectorStore
from intergrax.integrations.providers.vector_store.qdrant.adapter import QdrantVectorStoreIntegration
from intergrax.integrations.providers.vector_store.qdrant.config import QdrantIntegrationConfig
from intergrax.integrations.providers.vector_store.qdrant.opens import open_qdrant_vector_store


@dataclass(frozen=True)
class QdrantIntegrationBundle:
    config: QdrantIntegrationConfig
    vector_store: QdrantVectorStoreIntegration


def resolve_qdrant_config(**overrides: object) -> QdrantIntegrationConfig:
    return QdrantIntegrationConfig.from_env(**overrides)


def create_qdrant_integration(
    *,
    vector_store: Optional[VectorStore] = None,
    store: Optional[object] = None,
    store_factory: Optional[Callable[[], object]] = None,
    **config_overrides: object,
) -> QdrantIntegrationBundle:
    config = resolve_qdrant_config(**config_overrides)
    store_impl = open_qdrant_vector_store(
        config,
        implementation=vector_store,
        store=store,
        store_factory=store_factory,
    )
    assert isinstance(store_impl, QdrantVectorStoreIntegration)
    return QdrantIntegrationBundle(config=config, vector_store=store_impl)


def create_qdrant_vector_store(
    *,
    vector_store: Optional[VectorStore] = None,
    store: Optional[object] = None,
    store_factory: Optional[Callable[[], object]] = None,
    **config_overrides: object,
) -> QdrantVectorStoreIntegration:
    """Catalog factory for ``IntegrationSlug.QDRANT`` / ``VECTOR_STORE``."""
    return create_qdrant_integration(
        vector_store=vector_store,
        store=store,
        store_factory=store_factory,
        **config_overrides,
    ).vector_store
