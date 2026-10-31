# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Complete Chroma integration bundle — catalog bridge to ``intergrax/rag/``."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.vector_store import VectorStore
from intergrax.integrations.providers.vector_store.chroma.config import ChromaIntegrationConfig
from intergrax.integrations.providers.vector_store.chroma.integration import (
    CHROMA_VECTOR_STORE_PROVIDER_ID,
    ChromaVectorStoreClient,
    ChromaVectorStoreIntegration,
    ChromaVectorStoreIntegrationConfig,
)
from intergrax.integrations.providers.vector_store.chroma.opens import open_chroma_vector_store


@dataclass(frozen=True)
class ChromaIntegrationBundle:
    config: ChromaIntegrationConfig
    vector_store: ChromaVectorStoreIntegration


def resolve_chroma_config(**overrides: object) -> ChromaIntegrationConfig:
    return ChromaIntegrationConfig.from_env(**overrides)


def create_chroma_integration(
    *,
    vector_store: VectorStore | None = None,
    store: object | None = None,
    store_factory: Callable[[], object] | None = None,
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


def create_chroma_vector_store_integration(
    *,
    client: ChromaVectorStoreClient | None = None,
    enabled: bool = False,
) -> ChromaVectorStoreIntegration:
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Chroma vector store integration requires an injected client when enabled=True",
        )
    if client is not None:
        return ChromaVectorStoreIntegration.from_client(client, enabled=enabled)
    return ChromaVectorStoreIntegration.for_provider(
        provider_id=CHROMA_VECTOR_STORE_PROVIDER_ID,
        display_name="Chroma",
        config=ChromaVectorStoreIntegrationConfig(enabled=enabled),
    )


def create_chroma_vector_store(
    *,
    vector_store: VectorStore | None = None,
    store: object | None = None,
    store_factory: Callable[[], object] | None = None,
    **config_overrides: object,
) -> ChromaVectorStoreIntegration:
    """Catalog factory for ``"chroma"`` / ``VECTOR_STORE``."""
    if vector_store is None and store is None and store_factory is None and not config_overrides:
        return create_chroma_vector_store_integration(enabled=False)
    return create_chroma_integration(
        vector_store=vector_store,
        store=store,
        store_factory=store_factory,
        **config_overrides,
    ).vector_store
