# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Complete LanceDB integration bundle — catalog bridge to ``intergrax/rag/``."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from intergrax.integrations._shared.p3.configs import VectorIntegrationConfig
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.vector_store import VectorStore
from intergrax.integrations.providers.vector_store.lancedb.integration import (
    LANCEDB_VECTOR_STORE_PROVIDER_ID,
    LancedbVectorStoreClient,
    LancedbVectorStoreIntegration,
    LancedbVectorStoreIntegrationConfig,
)
from intergrax.integrations.providers.vector_store.lancedb.opens import open_lancedb_vector_store


@dataclass(frozen=True)
class LancedbIntegrationBundle:
    config: VectorIntegrationConfig
    vector_store: LancedbVectorStoreIntegration


def resolve_lancedb_config(**overrides: object) -> VectorIntegrationConfig:
    return VectorIntegrationConfig.from_env("INTERGRAX_LANCEDB", **overrides)


def create_lancedb_vector_store_integration(
    *,
    client: LancedbVectorStoreClient | None = None,
    enabled: bool = False,
) -> LancedbVectorStoreIntegration:
    """
    Build a contract-based LanceDB vector store integration.

    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Lancedb vector store integration requires an injected client when enabled=True",
        )
    if client is not None:
        return LancedbVectorStoreIntegration.from_client(client, enabled=enabled)
    return LancedbVectorStoreIntegration.for_provider(
        provider_id=LANCEDB_VECTOR_STORE_PROVIDER_ID,
        display_name="Lancedb",
        config=LancedbVectorStoreIntegrationConfig(enabled=enabled),
    )


def create_lancedb_integration(
    *,
    vector_store: VectorStore | None = None,
    store: object | None = None,
    store_factory: Callable[[], object] | None = None,
    **config_overrides: object,
) -> LancedbIntegrationBundle:
    config = resolve_lancedb_config(**config_overrides)
    store_impl = open_lancedb_vector_store(
        config,
        implementation=vector_store,
        store=store,
        store_factory=store_factory,
    )
    assert isinstance(store_impl, LancedbVectorStoreIntegration)
    return LancedbIntegrationBundle(config=config, vector_store=store_impl)


def create_lancedb_vector_store(
    *,
    vector_store: VectorStore | None = None,
    store: object | None = None,
    store_factory: Callable[[], object] | None = None,
    **config_overrides: object,
) -> LancedbVectorStoreIntegration:
    """Compatibility shim — constructs ``LancedbVectorStoreIntegration`` via legacy catalog path."""
    if vector_store is None and store is None and store_factory is None and not config_overrides:
        return create_lancedb_vector_store_integration(enabled=False)
    return create_lancedb_integration(
        vector_store=vector_store,
        store=store,
        store_factory=store_factory,
        **config_overrides,
    ).vector_store
