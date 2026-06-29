# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Complete pgvector integration bundle — catalog bridge to ``intergrax/rag/``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from intergrax.integrations._shared.p3.configs import VectorIntegrationConfig
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.vector_store import VectorStore
from intergrax.integrations.providers.vector_store.pgvector.integration import (
    PGVECTOR_VECTOR_STORE_PROVIDER_ID,
    PgvectorVectorStoreIntegration,
    PgvectorVectorStoreIntegrationConfig,
    PgvectorVectorStoreClient,
)
from intergrax.integrations.providers.vector_store.pgvector.opens import open_pgvector_vector_store


@dataclass(frozen=True)
class PgvectorIntegrationBundle:
    config: VectorIntegrationConfig
    vector_store: PgvectorVectorStoreIntegration


def resolve_pgvector_config(**overrides: object) -> VectorIntegrationConfig:
    return VectorIntegrationConfig.from_env("INTERGRAX_PGVECTOR", **overrides)


def create_pgvector_vector_store_integration(
    *,
    client: PgvectorVectorStoreClient | None = None,
    enabled: bool = False,
) -> PgvectorVectorStoreIntegration:
    """
    Build a contract-based pgvector vector store integration.

    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "pgvector vector store integration requires an injected client when enabled=True",
        )
    if client is not None:
        return PgvectorVectorStoreIntegration.from_client(client, enabled=enabled)
    return PgvectorVectorStoreIntegration.for_provider(
        provider_id=PGVECTOR_VECTOR_STORE_PROVIDER_ID,
        display_name="pgvector",
        config=PgvectorVectorStoreIntegrationConfig(enabled=enabled),
    )


def create_pgvector_integration(
    *,
    vector_store: Optional[VectorStore] = None,
    store: Optional[object] = None,
    store_factory: Optional[Callable[[], object]] = None,
    **config_overrides: object,
) -> PgvectorIntegrationBundle:
    config = resolve_pgvector_config(**config_overrides)
    store_impl = open_pgvector_vector_store(
        config,
        config_overrides=dict(config_overrides),
        implementation=vector_store,
        store=store,
        store_factory=store_factory,
    )
    assert isinstance(store_impl, PgvectorVectorStoreIntegration)
    return PgvectorIntegrationBundle(config=config, vector_store=store_impl)


def create_pgvector_vector_store(
    *,
    vector_store: Optional[VectorStore] = None,
    store: Optional[object] = None,
    store_factory: Optional[Callable[[], object]] = None,
    **config_overrides: object,
) -> PgvectorVectorStoreIntegration:
    """Compatibility shim — constructs ``PgvectorVectorStoreIntegration`` via legacy catalog path."""
    return create_pgvector_integration(
        vector_store=vector_store,
        store=store,
        store_factory=store_factory,
        **config_overrides,
    ).vector_store
