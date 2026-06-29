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

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.vector_store import VectorStore
from intergrax.integrations.providers.vector_store.pinecone.config import PineconeIntegrationConfig
from intergrax.integrations.providers.vector_store.pinecone.integration import (
    PINECONE_VECTOR_STORE_PROVIDER_ID,
    PineconeVectorStoreIntegration,
    PineconeVectorStoreIntegrationConfig,
    PineconeVectorStoreClient,
)
from intergrax.integrations.providers.vector_store.pinecone.opens import open_pinecone_vector_store


@dataclass(frozen=True)
class PineconeIntegrationBundle:
    config: PineconeIntegrationConfig
    vector_store: PineconeVectorStoreIntegration


def resolve_pinecone_config(**overrides: object) -> PineconeIntegrationConfig:
    return PineconeIntegrationConfig.from_env(**overrides)


def create_pinecone_vector_store_integration(
    *,
    client: PineconeVectorStoreClient | None = None,
    enabled: bool = False,
) -> PineconeVectorStoreIntegration:
    """
    Build a contract-based Pinecone vector store integration.

    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Pinecone vector store integration requires an injected client when enabled=True",
        )
    if client is not None:
        return PineconeVectorStoreIntegration.from_client(client, enabled=enabled)
    return PineconeVectorStoreIntegration.for_provider(
        provider_id=PINECONE_VECTOR_STORE_PROVIDER_ID,
        display_name="Pinecone",
        config=PineconeVectorStoreIntegrationConfig(enabled=enabled),
    )


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
    """Compatibility shim — constructs ``PineconeVectorStoreIntegration`` via legacy catalog path."""
    return create_pinecone_integration(
        vector_store=vector_store,
        store=store,
        store_factory=store_factory,
        **config_overrides,
    ).vector_store
