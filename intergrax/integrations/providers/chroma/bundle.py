# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Complete Chroma integration bundle — catalog bridge to ``intergrax/rag/``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from intergrax.integrations.contracts.vector_store import VectorStore
from intergrax.integrations.providers.chroma.adapter import ChromaVectorStoreIntegration
from intergrax.integrations.providers.chroma.config import ChromaIntegrationConfig
from intergrax.integrations.providers.chroma.opens import open_chroma_vector_store


@dataclass(frozen=True)
class ChromaIntegrationBundle:
    config: ChromaIntegrationConfig
    vector_store: ChromaVectorStoreIntegration


def resolve_chroma_config(**overrides: object) -> ChromaIntegrationConfig:
    return ChromaIntegrationConfig.from_env(**overrides)


def create_chroma_integration(
    *,
    vector_store: Optional[VectorStore] = None,
    store: Optional[object] = None,
    store_factory: Optional[Callable[[], object]] = None,
    **config_overrides: object,
) -> ChromaIntegrationBundle:
    config = resolve_chroma_config(**config_overrides)
    store_impl = open_chroma_vector_store(
        config,
        implementation=vector_store,
        store=store,
        store_factory=store_factory,
    )
    assert isinstance(store_impl, ChromaVectorStoreIntegration)
    return ChromaIntegrationBundle(config=config, vector_store=store_impl)


def create_chroma_vector_store(
    *,
    vector_store: Optional[VectorStore] = None,
    store: Optional[object] = None,
    store_factory: Optional[Callable[[], object]] = None,
    **config_overrides: object,
) -> ChromaVectorStoreIntegration:
    """Catalog factory for ``IntegrationSlug.CHROMA`` / ``VECTOR_STORE``."""
    return create_chroma_integration(
        vector_store=vector_store,
        store=store,
        store_factory=store_factory,
        **config_overrides,
    ).vector_store
