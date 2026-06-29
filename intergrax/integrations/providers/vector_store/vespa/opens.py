# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Vespa openers."""

from __future__ import annotations

from typing import Any, Callable, Optional

from intergrax.integrations.contracts.vector_store import VectorStore
from intergrax.integrations.providers.vector_store.vespa.adapter import _VespaVectorStore
from intergrax.integrations.providers.vector_store.vespa.client import VespaRestClient
from intergrax.integrations.providers.vector_store.vespa.config import VespaIntegrationConfig
from intergrax.integrations.providers.vector_store.vespa.integration import VespaVectorStoreIntegration


def _create_http_client(config: VespaIntegrationConfig) -> Any:
    import httpx

    timeout = float(config.timeout_seconds or 30.0)
    return httpx.Client(base_url=config.require_url(), timeout=timeout)


def open_vespa_rest_client(
    config: VespaIntegrationConfig,
    *,
    http_client: Optional[Any] = None,
    http_client_factory: Optional[Callable[[VespaIntegrationConfig], Any]] = None,
) -> VespaRestClient:
    if http_client is None:
        factory = http_client_factory or _create_http_client
        http_client = factory(config)
    return VespaRestClient(config, http_client=http_client)


def open_vespa_vector_store(
    config: VespaIntegrationConfig,
    *,
    implementation: Optional[VectorStore] = None,
    store: Optional[VectorStore] = None,
    store_factory: Optional[Callable[[], VectorStore]] = None,
    client: Optional[VespaRestClient] = None,
    http_client: Optional[Any] = None,
    http_client_factory: Optional[Callable[[VespaIntegrationConfig], Any]] = None,
) -> VectorStore:
    if implementation is not None:
        return implementation
    if store is not None:
        inner = store
    elif store_factory is not None:
        inner = store_factory()
    else:
        rest_client = client or open_vespa_rest_client(
            config,
            http_client=http_client,
            http_client_factory=http_client_factory,
        )
        inner = _VespaVectorStore(config, rest_client)
    return VespaVectorStoreIntegration.from_store(config, inner)
