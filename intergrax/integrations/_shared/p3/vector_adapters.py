# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Weaviate / Milvus facades implementing ``VectorStore`` for catalog registration."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from intergrax.integrations.contracts.vector_store import (
    MetadataFilter,
    VectorStore,
    VectorStoreHit,
    VectorStoreRecord,
    VectorStoreScope,
)
from intergrax.integrations.providers.vector_store.inmemory.rag_store import InMemoryVectorStore


class _VectorStoreFacade(VectorStore):
    """Delegates to an in-process store when vendor client lacks direct VectorStore mapping."""

    def __init__(self, *, collection: str, tenant_id: str, inner: Optional[VectorStore] = None) -> None:
        self._collection = collection
        self._inner = inner or InMemoryVectorStore(tenant_id=tenant_id)

    def add_records(
        self,
        records: Sequence[VectorStoreRecord],
        *,
        scope: VectorStoreScope,
    ) -> Sequence[str] | None:
        return self._inner.add_records(records, scope=scope)

    def query(
        self,
        query_embedding: Sequence[float],
        *,
        scope: VectorStoreScope,
        top_k: int,
        metadata_filter: Optional[MetadataFilter] = None,
        include_embeddings: bool = False,
    ) -> list[VectorStoreHit]:
        return self._inner.query(
            query_embedding,
            scope=scope,
            top_k=top_k,
            metadata_filter=metadata_filter,
            include_embeddings=include_embeddings,
        )

    def delete(self, ids: Sequence[str], *, scope: VectorStoreScope) -> None:
        self._inner.delete(ids, scope=scope)

    def count(self, *, scope: VectorStoreScope) -> int:
        return self._inner.count(scope=scope)


class WeaviateVectorFacade(_VectorStoreFacade):
    def __init__(self, client: Any, *, collection: str, tenant_id: str) -> None:
        super().__init__(collection=collection, tenant_id=tenant_id)
        self._client = client


class MilvusVectorFacade(_VectorStoreFacade):
    def __init__(self, client: Any, *, collection: str, tenant_id: str) -> None:
        super().__init__(collection=collection, tenant_id=tenant_id)
        self._client = client
