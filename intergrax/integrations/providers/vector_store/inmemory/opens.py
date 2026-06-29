# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Low-level Inmemory openers — internal to the inmemory integration package."""

from __future__ import annotations

from typing import Callable, Optional

from intergrax.integrations._shared.p3.configs import VectorIntegrationConfig
from intergrax.integrations.contracts.vector_store import VectorStore
from intergrax.integrations.providers.vector_store.inmemory.integration import InmemoryVectorStoreIntegration


def _open_rag_store(
    config: VectorIntegrationConfig,
    *,
    store_factory: Optional[Callable[[], VectorStore]] = None,
) -> VectorStore:
    if store_factory is not None:
        return store_factory()
    from intergrax.integrations.providers.vector_store.inmemory.rag_store import InMemoryVectorStore

    return InMemoryVectorStore(tenant_id=config.tenant_id)


def open_inmemory_vector_store(
    config: VectorIntegrationConfig,
    *,
    implementation: Optional[VectorStore] = None,
    store: Optional[VectorStore] = None,
    store_factory: Optional[Callable[[], VectorStore]] = None,
) -> VectorStore:
    if implementation is not None:
        return implementation
    inner = store if store is not None else _open_rag_store(config, store_factory=store_factory)
    return InmemoryVectorStoreIntegration.from_store(config, inner)
