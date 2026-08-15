# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Low-level Chroma openers — internal to the chroma integration package.

Only this module may import ``chromadb`` before constructing the RAG store.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from intergrax.integrations.contracts.base import (
    IntegrationConfigurationError,
    IntegrationDependencyError,
)
from intergrax.integrations.contracts.vector_store import VectorStore
from intergrax.integrations.providers.vector_store.chroma.config import (
    ChromaIntegrationConfig,
)
from intergrax.integrations.providers.vector_store.chroma.integration import (
    ChromaVectorStoreIntegration,
)


def _import_chromadb() -> Any:
    try:
        import chromadb
    except ImportError as exc:
        raise IntegrationConfigurationError(
            "Chroma integration requires chromadb. "
            "Install with: Intergrax-ai[vector-chroma]."
        ) from exc
    return chromadb


def _build_rag_config(config: ChromaIntegrationConfig) -> Any:
    from intergrax.integrations.providers.vector_store.chroma.rag_store import (
        ChromaConfig,
    )

    return ChromaConfig(
        collection_name=config.collection_name,
        tenant_id=config.tenant_id,
        persist_directory=config.persist_directory,
        batch_size=config.batch_size,
        metric=config.metric,
        mode=config.mode,
        http_host=config.http_host,
        http_port=config.http_port,
    )


def _is_transport_failure(exc: Exception) -> bool:
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    error_type = type(exc)
    if error_type.__module__.startswith(
        ("httpcore", "httpx")
    ) and error_type.__name__ in {
        "ConnectError",
        "ConnectTimeout",
        "NetworkError",
        "ReadTimeout",
    }:
        return True
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "connection refused",
            "connection reset",
            "failed to connect",
            "could not connect",
            "all connection attempts failed",
            "service unavailable",
        )
    )


def _raise_http_setup_failure(exc: Exception) -> None:
    if _is_transport_failure(exc):
        raise IntegrationDependencyError(
            "Chroma HTTP server is unavailable",
        ) from exc
    raise IntegrationConfigurationError(
        "Chroma HTTP client/server compatibility failure",
    ) from exc


def _build_client(config: ChromaIntegrationConfig, chromadb: Any) -> Any:
    if config.mode == "http":
        try:
            client = chromadb.HttpClient(
                host=config.http_host,
                port=config.http_port,
            )
            client.heartbeat()
            return client
        except Exception as exc:  # noqa: BLE001 - classify setup transport/protocol failures
            _raise_http_setup_failure(exc)

    try:
        if config.persist_directory:
            return chromadb.PersistentClient(path=config.persist_directory)
        return chromadb.Client()
    except Exception as exc:
        raise IntegrationConfigurationError(
            "Chroma explicit embedded client could not be created",
        ) from exc


def _open_rag_store(
    config: ChromaIntegrationConfig,
    *,
    store_factory: Callable[[], VectorStore] | None = None,
) -> VectorStore:
    if store_factory is not None:
        return store_factory()
    chromadb = _import_chromadb()
    from intergrax.integrations.providers.vector_store.chroma.rag_store import (
        ChromaVectorStore,
    )

    client = _build_client(config, chromadb)
    try:
        return ChromaVectorStore(_build_rag_config(config), client=client)
    except (IntegrationConfigurationError, IntegrationDependencyError):
        raise
    except Exception as exc:
        if config.mode == "http" and _is_transport_failure(exc):
            raise IntegrationDependencyError(
                "Chroma HTTP server is unavailable during collection setup",
            ) from exc
        raise IntegrationConfigurationError(
            "Chroma client/server compatibility failure during collection setup",
        ) from exc


def open_chroma_vector_store(
    config: ChromaIntegrationConfig,
    *,
    implementation: VectorStore | None = None,
    store: VectorStore | None = None,
    store_factory: Callable[[], VectorStore] | None = None,
) -> VectorStore:
    if implementation is not None:
        return implementation
    inner = (
        store
        if store is not None
        else _open_rag_store(config, store_factory=store_factory)
    )
    return ChromaVectorStoreIntegration.from_store(config, inner)
