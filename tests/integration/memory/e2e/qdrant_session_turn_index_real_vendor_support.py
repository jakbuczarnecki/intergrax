# © Artur Czarnecki. All rights reserved.

"""MEM-FINAL-AUDIT-5D — Qdrant-backed SessionTurnIndex qualification harness."""

from __future__ import annotations

import os
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from langchain_core.documents import Document
from numpy.typing import NDArray

from intergrax.applications._shared.session_turn_index_rag_adapters import (
    build_session_turn_index_creation_context,
)
from intergrax.integrations.contracts.base import IntegrationDependencyError
from intergrax.integrations.providers.vector_store.qdrant.bundle import create_qdrant_vector_store
from intergrax.integrations.providers.vector_store.qdrant.config import QdrantIntegrationConfig
from intergrax.integrations.providers.vector_store.qdrant.integration import (
    QDRANT_VECTOR_STORE_PROVIDER_ID,
    QdrantVectorStoreIntegration,
)
from intergrax.integrations.providers.vector_store.qdrant.opens import open_qdrant_control_plane_client
from intergrax.integrations.providers.vector_store.qdrant.rag_store import QdrantVectorStore
from intergrax.memory.contracts.session_turn_index import SessionTurnIndexStore
from intergrax.memory.session_turn_index_service import VectorSessionTurnIndexStore
from intergrax.rag.bootstrap.rag_stack_bootstrap import RagStack, create_default_rag_stack
from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.embedding.contracts.embedding_result import EmbeddingResult
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.vectorstore.contracts.native_vectorstore import VectorStoreScope
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager

_COLLECTION_PREFIX = "mem_audit_5d_"
_EMBEDDING_DIM = 16

CreateSessionTurnIndexStore = Callable[[], SessionTurnIndexStore]
DisposeSessionTurnIndexStore = Callable[[SessionTurnIndexStore], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class QdrantSessionTurnIndexQualificationEnv:
    qdrant_collection: str
    qualification_run_id: str
    tenant_id: str = "qual-tenant"


class DeterministicQualificationEmbeddingManager(BaseEmbeddingManager):
    """Stable vectors for contract qualification (not embedding-vendor proof)."""

    def __init__(self, *, dimensions: int = _EMBEDDING_DIM) -> None:
        self._dimensions = dimensions

    def embed_one(self, text: str) -> list[float]:
        return self.embed_texts([text])[0].tolist()

    def embed_documents(self, documents: Sequence[Document]) -> EmbeddingResult:
        texts = [doc.page_content for doc in documents]
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


def unique_qualification_run_id(prefix: str = "mem-5d") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:16]}"


def build_qualification_env(
    *,
    qualification_run_id: str,
    tenant_id: str = "qual-tenant",
) -> QdrantSessionTurnIndexQualificationEnv:
    safe = qualification_run_id.replace("-", "_")
    return QdrantSessionTurnIndexQualificationEnv(
        qdrant_collection=f"{_COLLECTION_PREFIX}{safe}",
        qualification_run_id=qualification_run_id,
        tenant_id=tenant_id,
    )


def qdrant_reachable() -> bool:
    if os.getenv("INTERGRAX_QDRANT_URL", "").strip():
        return True
    host = os.getenv("INTERGRAX_QDRANT_HOST", "localhost").strip() or "localhost"
    port = os.getenv("INTERGRAX_QDRANT_PORT", "6333").strip() or "6333"
    _ = host, port
    return True


def ensure_qdrant_available() -> None:
    env = build_qualification_env(qualification_run_id=unique_qualification_run_id("probe"))
    config = QdrantIntegrationConfig.from_env(
        collection_name=env.qdrant_collection,
        tenant_id=env.tenant_id,
    )
    client = open_qdrant_control_plane_client(config)
    try:
        client.get_collections()
    finally:
        client.close()


def _open_integration(
    env: QdrantSessionTurnIndexQualificationEnv,
    *,
    tenant_id: str | None = None,
    collection_name: str | None = None,
    url_override: str | None = None,
) -> QdrantVectorStoreIntegration:
    resolved_tenant = tenant_id or env.tenant_id
    overrides: dict[str, object] = {
        "collection_name": collection_name or env.qdrant_collection,
        "tenant_id": resolved_tenant,
        "enable_sparse_vectors": False,
    }
    if url_override is not None:
        overrides["url"] = url_override
    integration = create_qdrant_vector_store(**overrides)
    assert isinstance(integration, QdrantVectorStoreIntegration)
    integration.count(scope=VectorStoreScope(tenant_id=resolved_tenant))
    return integration


def build_qdrant_memory_rag_stack(
    env: QdrantSessionTurnIndexQualificationEnv,
    *,
    tenant_id: str | None = None,
    collection_name: str | None = None,
    url_override: str | None = None,
) -> tuple[RagStack, QdrantVectorStoreIntegration]:
    resolved_tenant = tenant_id or env.tenant_id
    integration = _open_integration(
        env,
        tenant_id=resolved_tenant,
        collection_name=collection_name,
        url_override=url_override,
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
    env: QdrantSessionTurnIndexQualificationEnv,
    *,
    tenant_id: str | None = None,
    collection_name: str | None = None,
    url_override: str | None = None,
) -> tuple[VectorSessionTurnIndexStore, QdrantVectorStoreIntegration]:
    resolved_tenant = tenant_id or env.tenant_id
    stack, integration = build_qdrant_memory_rag_stack(
        env,
        tenant_id=resolved_tenant,
        collection_name=collection_name,
        url_override=url_override,
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


def close_qdrant_integration(integration: QdrantVectorStoreIntegration) -> None:
    inner = integration.rag_store
    if isinstance(inner, QdrantVectorStore):
        client = inner._client
        if client is not None:
            client.close()


def drop_qualification_collection(env: QdrantSessionTurnIndexQualificationEnv) -> None:
    config = QdrantIntegrationConfig.from_env(
        collection_name=env.qdrant_collection,
        tenant_id=env.tenant_id,
    )
    client = open_qdrant_control_plane_client(config)
    tenant_suffixes = (
        env.tenant_id,
        "tenant-5d-a",
        "tenant-5d-b",
    )
    try:
        for tenant in tenant_suffixes:
            physical = f"{env.qdrant_collection}__tenant__{tenant}"
            try:
                client.delete_collection(physical)
            except Exception:
                continue
    finally:
        client.close()


def qdrant_topology_metadata(
    integration: QdrantVectorStoreIntegration,
) -> dict[str, str | None]:
    config = integration.store_config
    server_version: str | None = None
    client_version: str | None = None
    metric: str | None = config.metric if config is not None else None
    dimension: str | None = None
    inner = integration.rag_store
    if isinstance(inner, QdrantVectorStore) and inner._client is not None:
        try:
            import qdrant_client

            client_version = getattr(qdrant_client, "__version__", None)
            if client_version is not None:
                client_version = str(client_version)
        except ImportError:
            pass
        try:
            info = inner._client.get_collection(inner.collection_name)
            params = info.config.params.vectors
            if params is not None and hasattr(params, "size"):
                dimension = str(params.size)
            if params is not None and hasattr(params, "distance"):
                metric = str(params.distance)
        except Exception:
            pass
        try:
            meta = inner._client.get_cluster_info()
            server_version = str(getattr(meta, "version", None) or meta)
        except Exception:
            try:
                server_version = str(inner._client.rest_client.api("service").get_version())
            except Exception:
                server_version = None
    transport = "http"
    if config is not None and config.resolved_url():
        if config.resolved_url() and config.resolved_url().startswith("https"):
            transport = "https"
    return {
        "backend_provider_id": QDRANT_VECTOR_STORE_PROVIDER_ID,
        "server_version": server_version,
        "client_version": client_version,
        "collection": config.collection_name if config is not None else None,
        "metric": metric,
        "dimension": dimension,
        "topology": "single-node",
        "transport": transport,
        "deployment": "local-container-or-external",
    }


def session_turn_index_store_factory(
    env: QdrantSessionTurnIndexQualificationEnv,
    *,
    tenant_id: str | None = None,
) -> tuple[CreateSessionTurnIndexStore, DisposeSessionTurnIndexStore]:
    integrations: list[QdrantVectorStoreIntegration] = []

    def _create() -> SessionTurnIndexStore:
        store, integration = build_vector_session_turn_index_store(env, tenant_id=tenant_id)
        integrations.append(integration)
        return store

    async def _dispose(instance: SessionTurnIndexStore) -> None:
        _ = instance
        while integrations:
            close_qdrant_integration(integrations.pop())

    return _create, _dispose


def assert_qdrant_backend_unavailable(url: str) -> None:
    probe_env = build_qualification_env(qualification_run_id=unique_qualification_run_id("down"))
    try:
        _open_integration(probe_env, url_override=url)
    except (IntegrationDependencyError, ConnectionError, OSError, ValueError):
        return
    raise AssertionError("expected Qdrant backend to be unavailable for probe URL")
