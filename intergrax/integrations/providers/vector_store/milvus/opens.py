# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Low-level Milvus openers — vendor SDK imports stay in this module and ``rag_store.py``."""

from __future__ import annotations

from typing import Any, Callable, Optional

from intergrax.integrations._shared.p2.configs import HttpIntegrationConfig
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.vector_store import VectorStore
from intergrax.integrations.providers.vector_store.milvus.integration import MilvusVectorStoreIntegration


def _import_milvus_client() -> Any:
    try:
        from pymilvus import MilvusClient
    except ImportError as exc:
        raise IntegrationConfigurationError("Milvus requires pymilvus") from exc
    return MilvusClient


def open_milvus_client(config: HttpIntegrationConfig) -> Any:
    MilvusClient = _import_milvus_client()
    return MilvusClient(uri=config.require_url(), token=config.api_key or None)


def open_milvus_rag_store(
    config: HttpIntegrationConfig,
    *,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
) -> Any:
    from intergrax.integrations.providers.vector_store.milvus.rag_store import MilvusConfig, MilvusVectorStore

    resolved_client = client if client is not None else (client_factory() if client_factory else open_milvus_client(config))
    rag_cfg = MilvusConfig(collection_name=config.collection, tenant_id=config.tenant_id)
    return MilvusVectorStore(rag_cfg, client=resolved_client)


def open_milvus_vector_store(
    config: HttpIntegrationConfig,
    *,
    implementation: Optional[VectorStore] = None,
    store: Optional[VectorStore] = None,
    store_factory: Optional[Callable[[], VectorStore]] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
) -> VectorStore:
    if implementation is not None:
        return implementation
    if store is not None:
        inner = store
    elif store_factory is not None:
        inner = store_factory()
    else:
        inner = open_milvus_rag_store(config, client=client, client_factory=client_factory)
    return MilvusVectorStoreIntegration.from_store(config, inner)
