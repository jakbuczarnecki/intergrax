# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Qdrant vector store integration (INTEGRATIONS-2D contract · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import HealthStatus, IntegrationConfigurationError
from intergrax.integrations.contracts.health_probe import IntegrationHealthProbe
from intergrax.integrations.contracts.vector_store import (
    MetadataFilter,
    VectorStore,
    VectorStoreHit,
    VectorStoreRecord,
    VectorStoreScope,
)
from intergrax.integrations.providers.vector_store.qdrant.config import QdrantIntegrationConfig
from intergrax.rag.vectorstore.contracts.hybrid_search import provider_supports_native_hybrid_search
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig
from intergrax.runtime.integrations.categories.storage import VectorStoreIntegrationContract

QDRANT_VECTOR_STORE_PROVIDER_ID = "qdrant"


class QdrantVectorStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Qdrant vector store integration."""

    pass


@runtime_checkable
class QdrantVectorStoreClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class QdrantVectorStoreIntegration(VectorStoreIntegrationContract):
    """
    Single public Qdrant vector store entrypoint.

    Legacy catalog factories and contract factories both construct this class.
    Vector store operations delegate to an inner RAG store when present.
    """

    config: QdrantVectorStoreIntegrationConfig = QdrantVectorStoreIntegrationConfig()
    _client: QdrantVectorStoreClient | None = PrivateAttr(default=None)
    _store_config: QdrantIntegrationConfig | None = PrivateAttr(default=None)
    _inner: VectorStore | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: QdrantVectorStoreClient,
        *,
        enabled: bool = False,
    ) -> QdrantVectorStoreIntegration:
        integration = cls.for_provider(
            provider_id=QDRANT_VECTOR_STORE_PROVIDER_ID,
            display_name="Qdrant",
            config=QdrantVectorStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @classmethod
    def from_store(
        cls,
        store_config: QdrantIntegrationConfig,
        inner: VectorStore,
    ) -> QdrantVectorStoreIntegration:
        integration = cls.for_provider(
            provider_id=QDRANT_VECTOR_STORE_PROVIDER_ID,
            display_name="Qdrant",
            config=QdrantVectorStoreIntegrationConfig(enabled=True),
        )
        integration._store_config = store_config
        integration._inner = inner
        return integration

    @property
    def client(self) -> QdrantVectorStoreClient | None:
        return self._client

    @property
    def store_config(self) -> QdrantIntegrationConfig | None:
        """Catalog bridge settings when constructed via legacy factory."""
        return self._store_config

    @property
    def rag_store(self) -> VectorStore:
        return self._require_inner()

    def add_records(
        self,
        records: Sequence[VectorStoreRecord],
        *,
        scope: VectorStoreScope,
    ) -> Sequence[str] | None:
        return self._require_inner().add_records(records, scope=scope)

    def query(
        self,
        query_embedding: Sequence[float],
        *,
        scope: VectorStoreScope,
        top_k: int,
        metadata_filter: MetadataFilter | None = None,
        include_embeddings: bool = False,
    ) -> list[VectorStoreHit]:
        return self._require_inner().query(
            query_embedding,
            scope=scope,
            top_k=top_k,
            metadata_filter=metadata_filter,
            include_embeddings=include_embeddings,
        )

    def delete(self, ids: Sequence[str], *, scope: VectorStoreScope) -> None:
        self._require_inner().delete(ids, scope=scope)

    def count(self, *, scope: VectorStoreScope) -> int:
        return self._require_inner().count(scope=scope)

    def list_source_record_ids(
        self,
        *,
        source_id: str,
        scope: VectorStoreScope,
    ) -> Sequence[str]:
        inner = self._require_inner()
        lookup = getattr(inner, "list_source_record_ids", None)
        if not callable(lookup):
            raise RuntimeError("vectorstore_source_record_lookup_not_supported")
        return lookup(source_id=source_id, scope=scope)

    def supports_native_hybrid_search(self) -> bool:
        inner = self._inner
        if inner is None:
            return False
        return provider_supports_native_hybrid_search(inner)

    def query_hybrid(
        self,
        query_embedding: Sequence[float],
        query_text: str,
        *,
        scope: VectorStoreScope,
        top_k: int,
        metadata_filter: MetadataFilter | None = None,
        include_embeddings: bool = False,
        alpha: float = 0.5,
    ) -> list[VectorStoreHit]:
        if not self.supports_native_hybrid_search():
            raise IntegrationConfigurationError(
                "Qdrant vector store integration does not expose native hybrid search"
            )
        inner = self._require_inner()
        return inner.query_hybrid(
            query_embedding,
            query_text,
            scope=scope,
            top_k=top_k,
            metadata_filter=metadata_filter,
            include_embeddings=include_embeddings,
            alpha=alpha,
        )

    def health(self) -> HealthStatus | bool:
        inner = self._inner
        if inner is None:
            return True
        if isinstance(inner, IntegrationHealthProbe):
            return inner.health()
        return True

    def _require_inner(self) -> VectorStore:
        if self._inner is None:
            raise IntegrationConfigurationError(
                "Qdrant vector store integration requires an inner store for vector operations",
            )
        return self._inner


VectorStore.register(QdrantVectorStoreIntegration)
