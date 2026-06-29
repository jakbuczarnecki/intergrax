# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Pinecone vector store integration (INTEGRATIONS-2D contract · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.health_probe import IntegrationHealthProbe
from intergrax.integrations.contracts.vector_store import MetadataFilter, VectorStore, VectorStoreHit
from intergrax.integrations.providers.vector_store.pinecone.config import PineconeIntegrationConfig
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig
from intergrax.runtime.integrations.categories.storage import VectorStoreIntegrationContract

PINECONE_VECTOR_STORE_PROVIDER_ID = "pinecone"


class PineconeVectorStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Pinecone vector store integration."""

    pass


@runtime_checkable
class PineconeVectorStoreClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class PineconeVectorStoreIntegration(VectorStoreIntegrationContract):
    """
    Single public Pinecone vector store entrypoint.

    Legacy catalog factories and contract factories both construct this class.
    Vector store operations delegate to an inner RAG store when present.
    """

    config: PineconeVectorStoreIntegrationConfig = PineconeVectorStoreIntegrationConfig()
    _client: PineconeVectorStoreClient | None = PrivateAttr(default=None)
    _store_config: PineconeIntegrationConfig | None = PrivateAttr(default=None)
    _inner: VectorStore | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: PineconeVectorStoreClient,
        *,
        enabled: bool = False,
    ) -> PineconeVectorStoreIntegration:
        integration = cls.for_provider(
            provider_id=PINECONE_VECTOR_STORE_PROVIDER_ID,
            display_name="Pinecone",
            config=PineconeVectorStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @classmethod
    def from_store(
        cls,
        store_config: PineconeIntegrationConfig,
        inner: VectorStore,
    ) -> PineconeVectorStoreIntegration:
        integration = cls.for_provider(
            provider_id=PINECONE_VECTOR_STORE_PROVIDER_ID,
            display_name="Pinecone",
            config=PineconeVectorStoreIntegrationConfig(enabled=True),
        )
        integration._store_config = store_config
        integration._inner = inner
        return integration

    @property
    def client(self) -> PineconeVectorStoreClient | None:
        return self._client

    @property
    def store_config(self) -> PineconeIntegrationConfig | None:
        """Catalog bridge settings when constructed via legacy factory."""
        return self._store_config

    @property
    def rag_store(self) -> VectorStore:
        return self._require_inner()

    def add_documents(
        self,
        documents: Sequence[Any],
        embeddings: Sequence[Sequence[float]],
        *,
        ids: Sequence[str] | None = None,
    ) -> None:
        self._require_inner().add_documents(documents, embeddings, ids=ids)

    def query(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int,
        metadata_filter: MetadataFilter | None = None,
        include_embeddings: bool = False,
    ) -> list[VectorStoreHit]:
        return self._require_inner().query(
            query_embedding,
            top_k=top_k,
            metadata_filter=metadata_filter,
            include_embeddings=include_embeddings,
        )

    def delete(self, ids: Sequence[str]) -> None:
        self._require_inner().delete(ids)

    def count(self) -> int:
        return self._require_inner().count()

    def health(self) -> bool:
        inner = self._inner
        if inner is None:
            return True
        if isinstance(inner, IntegrationHealthProbe):
            return bool(inner.health())
        return True

    def _require_inner(self) -> VectorStore:
        if self._inner is None:
            raise IntegrationConfigurationError(
                "Pinecone vector store integration requires an inner store for vector operations",
            )
        return self._inner


VectorStore.register(PineconeVectorStoreIntegration)
