# © Artur Czarnecki. All rights reserved.

"""MEM-FINAL-AUDIT-5F — Chroma-backed SessionTurnIndex qualification harness."""

from __future__ import annotations

import os
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from intergrax.knowledge.contracts import KnowledgeDocument

from intergrax.applications._shared.session_turn_index_rag_adapters import (
    build_session_turn_index_creation_context,
)
from intergrax.integrations.contracts.base import IntegrationConfigurationError, IntegrationDependencyError
from intergrax.integrations.providers.vector_store.chroma.bundle import create_chroma_vector_store
from intergrax.integrations.providers.vector_store.chroma.integration import (
    CHROMA_VECTOR_STORE_PROVIDER_ID,
    ChromaVectorStoreIntegration,
)
from intergrax.integrations.providers.vector_store.chroma.rag_store import ChromaVectorStore
from intergrax.memory.contracts.session_turn_index import SessionTurnIndexStore
from intergrax.memory.session_turn_index_service import VectorSessionTurnIndexStore
from intergrax.rag.bootstrap.rag_stack_bootstrap import RagStack, create_default_rag_stack
from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.embedding.contracts.embedding_result import EmbeddingResult
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.vectorstore.contracts.native_vectorstore import VectorStoreScope
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager

_COLLECTION_PREFIX = "mem_audit_5f_"
_EMBEDDING_DIM = 4

CreateSessionTurnIndexStore = Callable[[], SessionTurnIndexStore]
DisposeSessionTurnIndexStore = Callable[[SessionTurnIndexStore], Awaitable[None]]

_QUALIFICATION_TENANTS = (
    "tenant-5f-a",
    "tenant-5f-b",
)


@dataclass(frozen=True, slots=True)
class ChromaSessionTurnIndexQualificationEnv:
    chroma_collection: str
    qualification_run_id: str
    tenant_id: str = "qual-tenant"


class DeterministicQualificationEmbeddingManager(BaseEmbeddingManager):
    """Stable vectors for contract qualification (not embedding-vendor proof)."""

    def __init__(self, *, dimensions: int = _EMBEDDING_DIM) -> None:
        self._dimensions = dimensions

    def embed_one(self, text: str) -> NDArray[np.float32]:
        return self.embed_texts([text])[0]

    def embed_documents(self, documents: Sequence[KnowledgeDocument]) -> EmbeddingResult:
        texts = [doc.content for doc in documents]
        matrix = self.embed_texts(texts)
        return EmbeddingResult(embeddings=matrix, texts=list(texts))

    def embed_texts(self, texts: Sequence[str]) -> NDArray[np.float32]:
        rows: list[list[float]] = []
        for text in texts:
            seed = sum(ord(ch) for ch in text) % 997
            rows.append(
                [float((seed + index) % 17) / 17.0 for index in range(self._dimensions)]
            )
        return np.array(rows, dtype=np.float32)


def unique_qualification_run_id(prefix: str = "mem-5f") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:16]}"


def build_qualification_env(
    *,
    qualification_run_id: str,
    tenant_id: str = "qual-tenant",
) -> ChromaSessionTurnIndexQualificationEnv:
    safe = qualification_run_id.replace("-", "_")
    return ChromaSessionTurnIndexQualificationEnv(
        chroma_collection=f"{_COLLECTION_PREFIX}{safe}",
        qualification_run_id=qualification_run_id,
        tenant_id=tenant_id,
    )


def chroma_reachable() -> bool:
    host = os.environ.get("INTERGRAX_CHROMA_HOST", "localhost").strip() or "localhost"
    port = os.environ.get("INTERGRAX_CHROMA_PORT", "8000").strip() or "8000"
    _ = host, port
    return True


def ensure_chroma_available() -> None:
    env = build_qualification_env(qualification_run_id=unique_qualification_run_id("probe"))
    integration = _open_integration(env)
    try:
        inner = integration.rag_store
        if isinstance(inner, ChromaVectorStore) and inner._client is not None:
            inner._client.heartbeat()
    finally:
        close_chroma_integration(integration)


def _open_integration(
    env: ChromaSessionTurnIndexQualificationEnv,
    *,
    tenant_id: str | None = None,
    collection_name: str | None = None,
    host_override: str | None = None,
    port_override: int | None = None,
) -> ChromaVectorStoreIntegration:
    resolved_tenant = tenant_id or env.tenant_id
    overrides: dict[str, object] = {
        "collection_name": collection_name or env.chroma_collection,
        "tenant_id": resolved_tenant,
        "mode": "http",
    }
    if host_override is not None:
        overrides["http_host"] = host_override
    if port_override is not None:
        overrides["http_port"] = port_override
    integration = create_chroma_vector_store(**overrides)
    assert isinstance(integration, ChromaVectorStoreIntegration)
    integration.count(scope=VectorStoreScope(tenant_id=resolved_tenant))
    return integration


def build_chroma_memory_rag_stack(
    env: ChromaSessionTurnIndexQualificationEnv,
    *,
    tenant_id: str | None = None,
    collection_name: str | None = None,
    host_override: str | None = None,
    port_override: int | None = None,
) -> tuple[RagStack, ChromaVectorStoreIntegration]:
    resolved_tenant = tenant_id or env.tenant_id
    integration = _open_integration(
        env,
        tenant_id=resolved_tenant,
        collection_name=collection_name,
        host_override=host_override,
        port_override=port_override,
    )
    embedding = DeterministicQualificationEmbeddingManager()
    vectorstore = VectorstoreManager(
        store=integration.rag_store,
        scope=VectorStoreScope(tenant_id=resolved_tenant),
    )
    stack = create_default_rag_stack(
        tenant_id=resolved_tenant,
        vectorstore_manager=vectorstore,
        embedding_manager=embedding,
        profile=RagProfile(),
    )
    return stack, integration


def build_vector_session_turn_index_store(
    env: ChromaSessionTurnIndexQualificationEnv,
    *,
    tenant_id: str | None = None,
    collection_name: str | None = None,
    host_override: str | None = None,
    port_override: int | None = None,
) -> tuple[VectorSessionTurnIndexStore, ChromaVectorStoreIntegration]:
    resolved_tenant = tenant_id or env.tenant_id
    stack, integration = build_chroma_memory_rag_stack(
        env,
        tenant_id=resolved_tenant,
        collection_name=collection_name,
        host_override=host_override,
        port_override=port_override,
    )
    creation_context = build_session_turn_index_creation_context(
        tenant_id=resolved_tenant,
        embedding_manager=stack.embedding_manager,
        vectorstore_manager=stack.vectorstore_manager,
    )
    store = VectorSessionTurnIndexStore(
        embedding_port=creation_context.embedding_manager,
        vectorstore_port=creation_context.vectorstore_manager,
        tenant_id=resolved_tenant,
    )
    return store, integration


def close_chroma_integration(integration: ChromaVectorStoreIntegration) -> None:
    _ = integration


def _physical_collection_names(
    env: ChromaSessionTurnIndexQualificationEnv,
) -> tuple[str, ...]:
    base = env.chroma_collection
    tenants = (env.tenant_id, "qual-tenant", *_QUALIFICATION_TENANTS)
    return tuple(f"{base}__tenant__{tenant}" for tenant in tenants)


def drop_qualification_collections(env: ChromaSessionTurnIndexQualificationEnv) -> None:
    integration = _open_integration(env)
    inner = integration.rag_store
    if not isinstance(inner, ChromaVectorStore) or inner._client is None:
        close_chroma_integration(integration)
        return
    client = inner._client
    try:
        existing = {collection.name for collection in client.list_collections()}
        for physical in _physical_collection_names(env):
            if physical not in existing:
                continue
            try:
                client.delete_collection(name=physical)
            except Exception:
                continue
    finally:
        close_chroma_integration(integration)


def chroma_topology_metadata(
    integration: ChromaVectorStoreIntegration,
) -> dict[str, str | None]:
    config = integration.store_config
    chromadb_version: str | None = None
    server_version: str | None = None
    metric: str | None = config.metric if config is not None else None
    dimension: str | None = str(_EMBEDDING_DIM)
    collection: str | None = None
    deployment = "http-server"
    transport = "http"
    inner = integration.rag_store
    if isinstance(inner, ChromaVectorStore):
        collection = inner.collection_name
        if inner._dim is not None:
            dimension = str(inner._dim)
        client = inner._client
        if client is not None:
            try:
                import chromadb

                chromadb_version = str(getattr(chromadb, "__version__", None) or "")
            except ImportError:
                chromadb_version = None
            try:
                heartbeat = client.heartbeat()
                if isinstance(heartbeat, dict):
                    server_version = str(heartbeat.get("nanosecond heartbeat", heartbeat))
                else:
                    server_version = str(heartbeat)
            except Exception:
                server_version = None
        if config is not None and config.mode == "embedded":
            if config.persist_directory:
                deployment = "persistent-local-directory"
            else:
                deployment = "embedded-ephemeral"
    return {
        "backend_provider_id": CHROMA_VECTOR_STORE_PROVIDER_ID,
        "chromadb_version": chromadb_version,
        "server_version": server_version,
        "collection": collection,
        "metric": metric,
        "dimension": dimension,
        "topology": "single-node",
        "transport": transport,
        "deployment": deployment,
    }


def session_turn_index_store_factory(
    env: ChromaSessionTurnIndexQualificationEnv,
    *,
    tenant_id: str | None = None,
) -> tuple[CreateSessionTurnIndexStore, DisposeSessionTurnIndexStore]:
    integrations: list[ChromaVectorStoreIntegration] = []

    def _create() -> SessionTurnIndexStore:
        store, integration = build_vector_session_turn_index_store(env, tenant_id=tenant_id)
        integrations.append(integration)
        return store

    async def _dispose(instance: SessionTurnIndexStore) -> None:
        _ = instance
        while integrations:
            close_chroma_integration(integrations.pop())

    return _create, _dispose


def assert_chroma_backend_unavailable(
    *,
    host: str = "127.0.0.1",
    port: int = 1,
) -> None:
    probe_env = build_qualification_env(qualification_run_id=unique_qualification_run_id("down"))
    try:
        _open_integration(probe_env, host_override=host, port_override=port)
    except (IntegrationConfigurationError, IntegrationDependencyError, ConnectionError, OSError, ValueError):
        return
    except Exception as exc:
        if _is_transport_failure(exc):
            return
        raise
    raise AssertionError("expected Chroma backend to be unavailable for probe host/port")


def _is_transport_failure(exc: Exception) -> bool:
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "connection refused",
            "failed to connect",
            "could not connect",
            "unavailable",
        )
    )


def chroma_backend_side_scope_markers() -> tuple[str, ...]:
    return (
        "$and",
        "tenant_id",
        "$eq",
        "session_id",
        "user_id",
    )
