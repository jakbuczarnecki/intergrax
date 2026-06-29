# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Complete Inmemory integration bundle — catalog bridge to ``intergrax/rag/``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from intergrax.integrations._shared.p3.configs import VectorIntegrationConfig
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.vector_store import VectorStore
from intergrax.integrations.providers.vector_store.inmemory.integration import (
    INMEMORY_VECTOR_STORE_PROVIDER_ID,
    InmemoryVectorStoreIntegration,
    InmemoryVectorStoreIntegrationConfig,
    InmemoryVectorStoreClient,
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
    vector_store: Optional[VectorStore] = None,
    store: Optional[object] = None,
    store_factory: Optional[Callable[[], object]] = None,
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
    vector_store: Optional[VectorStore] = None,
    store: Optional[object] = None,
    store_factory: Optional[Callable[[], object]] = None,
    **config_overrides: object,
) -> InmemoryVectorStoreIntegration:
    """Compatibility shim — constructs ``InmemoryVectorStoreIntegration`` via legacy catalog path."""
    return create_inmemory_integration(
        vector_store=vector_store,
        store=store,
        store_factory=store_factory,
        **config_overrides,
    ).vector_store
