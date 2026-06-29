# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Complete Milvus integration bundle — catalog bridge to ``intergrax/rag/``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from intergrax.integrations._shared.p2.configs import HttpIntegrationConfig
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.vector_store import VectorStore
from intergrax.integrations.providers.vector_store.milvus.integration import (
    MILVUS_VECTOR_STORE_PROVIDER_ID,
    MilvusVectorStoreIntegration,
    MilvusVectorStoreIntegrationConfig,
    MilvusVectorStoreClient,
)
from intergrax.integrations.providers.vector_store.milvus.opens import open_milvus_vector_store


@dataclass(frozen=True)
class MilvusIntegrationBundle:
    config: HttpIntegrationConfig
    vector_store: MilvusVectorStoreIntegration


def resolve_milvus_config(**overrides: object) -> HttpIntegrationConfig:
    return HttpIntegrationConfig.from_env("INTERGRAX_MILVUS", **overrides)


def create_milvus_vector_store_integration(
    *,
    client: MilvusVectorStoreClient | None = None,
    enabled: bool = False,
) -> MilvusVectorStoreIntegration:
    """
    Build a contract-based Milvus vector store integration.

    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Milvus vector store integration requires an injected client when enabled=True",
        )
    if client is not None:
        return MilvusVectorStoreIntegration.from_client(client, enabled=enabled)
    return MilvusVectorStoreIntegration.for_provider(
        provider_id=MILVUS_VECTOR_STORE_PROVIDER_ID,
        display_name="Milvus",
        config=MilvusVectorStoreIntegrationConfig(enabled=enabled),
    )


def create_milvus_integration(
    *,
    vector_store: Optional[VectorStore] = None,
    store: Optional[object] = None,
    store_factory: Optional[Callable[[], object]] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> MilvusIntegrationBundle:
    config = resolve_milvus_config(**config_overrides)
    store_impl = open_milvus_vector_store(
        config,
        implementation=vector_store,
        store=store,
        store_factory=store_factory,
        client=client,
        client_factory=client_factory,
    )
    assert isinstance(store_impl, MilvusVectorStoreIntegration)
    return MilvusIntegrationBundle(config=config, vector_store=store_impl)


def create_milvus_vector_store(
    *,
    vector_store: Optional[VectorStore] = None,
    store: Optional[object] = None,
    store_factory: Optional[Callable[[], object]] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> MilvusVectorStoreIntegration:
    """Compatibility shim — constructs ``MilvusVectorStoreIntegration`` via legacy catalog path."""
    return create_milvus_integration(
        vector_store=vector_store,
        store=store,
        store_factory=store_factory,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    ).vector_store
