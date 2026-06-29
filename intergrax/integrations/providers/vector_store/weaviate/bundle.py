# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Complete Weaviate integration bundle — catalog bridge to ``intergrax/rag/``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from intergrax.integrations._shared.p2.configs import HttpIntegrationConfig
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.vector_store import VectorStore
from intergrax.integrations.providers.vector_store.weaviate.integration import (
    WEAVIATE_VECTOR_STORE_PROVIDER_ID,
    WeaviateVectorStoreIntegration,
    WeaviateVectorStoreIntegrationConfig,
    WeaviateVectorStoreClient,
)
from intergrax.integrations.providers.vector_store.weaviate.opens import open_weaviate_vector_store


@dataclass(frozen=True)
class WeaviateIntegrationBundle:
    config: HttpIntegrationConfig
    vector_store: WeaviateVectorStoreIntegration


def resolve_weaviate_config(**overrides: object) -> HttpIntegrationConfig:
    return HttpIntegrationConfig.from_env("INTERGRAX_WEAVIATE", **overrides)


def create_weaviate_vector_store_integration(
    *,
    client: WeaviateVectorStoreClient | None = None,
    enabled: bool = False,
) -> WeaviateVectorStoreIntegration:
    """
    Build a contract-based Weaviate vector store integration.

    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Weaviate vector store integration requires an injected client when enabled=True",
        )
    if client is not None:
        return WeaviateVectorStoreIntegration.from_client(client, enabled=enabled)
    return WeaviateVectorStoreIntegration.for_provider(
        provider_id=WEAVIATE_VECTOR_STORE_PROVIDER_ID,
        display_name="Weaviate",
        config=WeaviateVectorStoreIntegrationConfig(enabled=enabled),
    )


def create_weaviate_integration(
    *,
    vector_store: Optional[VectorStore] = None,
    store: Optional[object] = None,
    store_factory: Optional[Callable[[], object]] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> WeaviateIntegrationBundle:
    config = resolve_weaviate_config(**config_overrides)
    store_impl = open_weaviate_vector_store(
        config,
        implementation=vector_store,
        store=store,
        store_factory=store_factory,
        client=client,
        client_factory=client_factory,
    )
    assert isinstance(store_impl, WeaviateVectorStoreIntegration)
    return WeaviateIntegrationBundle(config=config, vector_store=store_impl)


def create_weaviate_vector_store(
    *,
    vector_store: Optional[VectorStore] = None,
    store: Optional[object] = None,
    store_factory: Optional[Callable[[], object]] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> WeaviateVectorStoreIntegration:
    """Compatibility shim — constructs ``WeaviateVectorStoreIntegration`` via legacy catalog path."""
    return create_weaviate_integration(
        vector_store=vector_store,
        store=store,
        store_factory=store_factory,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    ).vector_store
