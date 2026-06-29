# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Vespa integration bundle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.vector_store import VectorStore
from intergrax.integrations.providers.vector_store.vespa.client import VespaRestClient
from intergrax.integrations.providers.vector_store.vespa.config import VespaIntegrationConfig
from intergrax.integrations.providers.vector_store.vespa.integration import (
    VESPA_VECTOR_STORE_PROVIDER_ID,
    VespaVectorStoreIntegration,
    VespaVectorStoreIntegrationConfig,
    VespaVectorStoreClient,
)
from intergrax.integrations.providers.vector_store.vespa.opens import open_vespa_rest_client, open_vespa_vector_store


@dataclass(frozen=True)
class VespaIntegrationBundle:
    config: VespaIntegrationConfig
    vector_store: VespaVectorStoreIntegration
    rest_client: VespaRestClient | None = None


def create_vespa_vector_store_integration(
    *,
    client: VespaVectorStoreClient | None = None,
    enabled: bool = False,
) -> VespaVectorStoreIntegration:
    """
    Build a contract-based Vespa vector store integration.

    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Vespa vector store integration requires an injected client when enabled=True",
        )
    if client is not None:
        return VespaVectorStoreIntegration.from_client(client, enabled=enabled)
    return VespaVectorStoreIntegration.for_provider(
        provider_id=VESPA_VECTOR_STORE_PROVIDER_ID,
        display_name="Vespa",
        config=VespaVectorStoreIntegrationConfig(enabled=enabled),
    )


def create_vespa_integration(
    *,
    vector_store: Optional[VectorStore] = None,
    store: Optional[object] = None,
    store_factory: Optional[Callable[[], object]] = None,
    client: Optional[VespaRestClient] = None,
    http_client: Optional[Any] = None,
    http_client_factory: Optional[Callable[[VespaIntegrationConfig], Any]] = None,
    **config_overrides: object,
) -> VespaIntegrationBundle:
    config = VespaIntegrationConfig.from_env(**config_overrides)
    rest_client = None if store_factory is not None or store is not None or vector_store is not None else (
        client or open_vespa_rest_client(
            config,
            http_client=http_client,
            http_client_factory=http_client_factory,
        )
    )
    store_impl = open_vespa_vector_store(
        config,
        implementation=vector_store,
        store=store,
        store_factory=store_factory,
        client=rest_client,
        http_client=http_client,
        http_client_factory=http_client_factory,
    )
    assert isinstance(store_impl, VespaVectorStoreIntegration)
    return VespaIntegrationBundle(config=config, vector_store=store_impl, rest_client=rest_client)


def create_vespa_vector_store(
    *,
    vector_store: Optional[VectorStore] = None,
    store: Optional[object] = None,
    store_factory: Optional[Callable[[], object]] = None,
    client: Optional[VespaRestClient] = None,
    http_client: Optional[Any] = None,
    http_client_factory: Optional[Callable[[VespaIntegrationConfig], Any]] = None,
    **config_overrides: object,
) -> VespaVectorStoreIntegration:
    """Catalog factory for ``"vespa"``."""
    return create_vespa_integration(
        vector_store=vector_store,
        store=store,
        store_factory=store_factory,
        client=client,
        http_client=http_client,
        http_client_factory=http_client_factory,
        **config_overrides,
    ).vector_store
