# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Complete Inmemory integration bundle — catalog bridge to ``intergrax/rag/``."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from intergrax.integrations._shared.p3.configs import VectorIntegrationConfig
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.vector_store import VectorStore
from intergrax.integrations.providers.vector_store.inmemory.integration import (
    INMEMORY_VECTOR_STORE_PROVIDER_ID,
    InmemoryVectorStoreClient,
    InmemoryVectorStoreIntegration,
    InmemoryVectorStoreIntegrationConfig,
)
from intergrax.integrations.providers.vector_store.inmemory.opens import open_inmemory_vector_store


@dataclass(frozen=True)
class InmemoryIntegrationBundle:
    config: VectorIntegrationConfig
    vector_store: InmemoryVectorStoreIntegration


def resolve_inmemory_config(**overrides: object) -> VectorIntegrationConfig:
    return VectorIntegrationConfig.from_env("INTERGRAX_INMEMORY", **overrides)


def create_inmemory_vector_store_integration(
    *,
    client: InmemoryVectorStoreClient | None = None,
    enabled: bool = False,
) -> InmemoryVectorStoreIntegration:
    """
    Build a contract-based Inmemory vector store integration.

    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Inmemory vector store integration requires an injected client when enabled=True",
        )
    if client is not None:
        return InmemoryVectorStoreIntegration.from_client(client, enabled=enabled)
    return InmemoryVectorStoreIntegration.for_provider(
        provider_id=INMEMORY_VECTOR_STORE_PROVIDER_ID,
        display_name="Inmemory",
        config=InmemoryVectorStoreIntegrationConfig(enabled=enabled),
    )


def create_inmemory_integration(
    *,
    vector_store: VectorStore | None = None,
    store: object | None = None,
    store_factory: Callable[[], object] | None = None,
    **config_overrides: object,
) -> InmemoryIntegrationBundle:
    config = resolve_inmemory_config(**config_overrides)
    store_impl = open_inmemory_vector_store(
        config,
        implementation=vector_store,
        store=store,
        store_factory=store_factory,
    )
    assert isinstance(store_impl, InmemoryVectorStoreIntegration)
    return InmemoryIntegrationBundle(config=config, vector_store=store_impl)


def create_inmemory_vector_store(
    *,
    vector_store: VectorStore | None = None,
    store: object | None = None,
    store_factory: Callable[[], object] | None = None,
    **config_overrides: object,
) -> InmemoryVectorStoreIntegration:
    """Compatibility shim — constructs ``InmemoryVectorStoreIntegration`` via legacy catalog path."""
    if vector_store is None and store is None and store_factory is None and not config_overrides:
        return create_inmemory_vector_store_integration(enabled=False)
    return create_inmemory_integration(
        vector_store=vector_store,
        store=store,
        store_factory=store_factory,
        **config_overrides,
    ).vector_store
