# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Low-level Typesense openers — internal to the typesense integration package."""

from __future__ import annotations

from typing import Any, Callable, Optional

from intergrax.integrations._shared.p2.configs import HttpIntegrationConfig
from intergrax.integrations._shared.p3.configs import VectorIntegrationConfig
from intergrax.integrations.contracts.vector_store import VectorStore
from intergrax.integrations.providers.vector_store.typesense.integration import TypesenseVectorStoreIntegration


def _open_rag_store(
    config: VectorIntegrationConfig,
    *,
    config_overrides: dict[str, object],
    store_factory: Optional[Callable[[], VectorStore]] = None,
) -> VectorStore:
    if store_factory is not None:
        return store_factory()
    from intergrax.integrations._shared.p2.factories import _open_httpx_client
    from intergrax.integrations._shared.p7.factories import _TypesenseHttpVectorStore

    http_config = HttpIntegrationConfig.from_env("INTERGRAX_TYPESENSE", **config_overrides)
    http = _open_httpx_client(
        http_config,
        default_url=config.url or http_config.base_url or "http://127.0.0.1:8108",
    )
    return _TypesenseHttpVectorStore(http, collection=config.collection)


def open_typesense_vector_store(
    config: VectorIntegrationConfig,
    *,
    config_overrides: dict[str, object] | None = None,
    implementation: Optional[VectorStore] = None,
    store: Optional[VectorStore] = None,
    store_factory: Optional[Callable[[], VectorStore]] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
) -> VectorStore:
    del client, client_factory
    if implementation is not None:
        return implementation
    overrides = config_overrides or {}
    if store is not None:
        inner = store
    else:
        inner = _open_rag_store(config, config_overrides=overrides, store_factory=store_factory)
    return TypesenseVectorStoreIntegration.from_store(config, inner)
