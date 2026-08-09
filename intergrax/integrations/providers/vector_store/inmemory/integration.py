# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Inmemory vector store integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.vector_store import (
    MetadataFilter,
    VectorStore,
    VectorStoreHit,
    VectorStoreRecord,
    VectorStoreScope,
)
from intergrax.runtime.integrations.categories.storage import VectorStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

INMEMORY_VECTOR_STORE_PROVIDER_ID = "inmemory"


class InmemoryVectorStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Inmemory vector store integration."""

    pass


@runtime_checkable
class InmemoryVectorStoreClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class InmemoryVectorStoreIntegration(VectorStoreIntegrationContract):
    """
    Single public Inmemory vector store entrypoint.

    Legacy catalog factories construct this class. Catalog factory (create_inmemory_vector_store) delegates to this class.
    """

    config: InmemoryVectorStoreIntegrationConfig = InmemoryVectorStoreIntegrationConfig()
    _client: InmemoryVectorStoreClient | None = PrivateAttr(default=None)
    _store_config: object | None = PrivateAttr(default=None)
    _inner: VectorStore | None = PrivateAttr(default=None)

    @classmethod
    def from_store(
        cls,
        store_config: object,
        inner: VectorStore,
        *,
        enabled: bool = True,
    ) -> InmemoryVectorStoreIntegration:
        integration = cls.for_provider(
            provider_id=INMEMORY_VECTOR_STORE_PROVIDER_ID,
            display_name="Inmemory",
            config=InmemoryVectorStoreIntegrationConfig(enabled=enabled),
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
        return self._require_inner().list_source_record_ids(
            source_id=source_id,
            scope=scope,
        )


    def _require_inner(self) -> VectorStore:
        if self._inner is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires an inner store for vector operations",
            )
        return self._inner


    @classmethod
    def from_client(
        cls,
        client: InmemoryVectorStoreClient,
        *,
        enabled: bool = False,
    ) -> InmemoryVectorStoreIntegration:
        integration = cls.for_provider(
            provider_id=INMEMORY_VECTOR_STORE_PROVIDER_ID,
            display_name="Inmemory",
            config=InmemoryVectorStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> InmemoryVectorStoreClient | None:
        return self._client

VectorStore.register(InmemoryVectorStoreIntegration)
