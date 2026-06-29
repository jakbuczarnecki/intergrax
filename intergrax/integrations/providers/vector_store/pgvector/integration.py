# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Pgvector vector store integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError, HealthStatus
from intergrax.integrations.contracts.health_probe import IntegrationHealthProbe
from intergrax.integrations.contracts.vector_store import MetadataFilter, VectorStore, VectorStoreHit
from intergrax.runtime.integrations.categories.storage import VectorStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

PGVECTOR_VECTOR_STORE_PROVIDER_ID = "pgvector"


class PgvectorVectorStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Pgvector vector store integration."""

    pass


@runtime_checkable
class PgvectorVectorStoreClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class PgvectorVectorStoreIntegration(VectorStoreIntegrationContract):
    """
    Single public Pgvector vector store entrypoint.

    Legacy catalog factories construct this class. Catalog factory (create_pgvector_vector_store) delegates to this class.
    """

    config: PgvectorVectorStoreIntegrationConfig = PgvectorVectorStoreIntegrationConfig()
    _client: PgvectorVectorStoreClient | None = PrivateAttr(default=None)
    _store_config: object | None = PrivateAttr(default=None)
    _inner: VectorStore | None = PrivateAttr(default=None)

    @classmethod
    def from_store(
        cls,
        store_config: object,
        inner: VectorStore,
        *,
        enabled: bool = True,
    ) -> PgvectorVectorStoreIntegration:
        integration = cls.for_provider(
            provider_id=PGVECTOR_VECTOR_STORE_PROVIDER_ID,
            display_name="Pgvector",
            config=PgvectorVectorStoreIntegrationConfig(enabled=enabled),
        )
        integration._store_config = store_config
        integration._inner = inner
        return integration


    @property
    def store_config(self) -> object | None:
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


    def _require_inner(self) -> VectorStore:
        if self._inner is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires an inner store for vector operations",
            )
        return self._inner


    @classmethod
    def from_client(
        cls,
        client: PgvectorVectorStoreClient,
        *,
        enabled: bool = False,
    ) -> PgvectorVectorStoreIntegration:
        integration = cls.for_provider(
            provider_id=PGVECTOR_VECTOR_STORE_PROVIDER_ID,
            display_name="Pgvector",
            config=PgvectorVectorStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> PgvectorVectorStoreClient | None:
        return self._client

    def health(self) -> HealthStatus | bool:
        inner = self._inner
        if inner is None:
            return True
        if isinstance(inner, IntegrationHealthProbe):
            return inner.health()
        probe = getattr(inner, "health", None)
        if callable(probe):
            return probe()
        return True

VectorStore.register(PgvectorVectorStoreIntegration)
