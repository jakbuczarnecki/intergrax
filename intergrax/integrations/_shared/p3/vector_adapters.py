# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Weaviate / Milvus facades implementing ``VectorStore`` for catalog registration."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from langchain_core.documents import Document

from intergrax.integrations.contracts.vector_store import MetadataFilter, VectorStore, VectorStoreHit
from intergrax.rag.vectorstore.providers.inmemory_vectorstore import InMemoryVectorStore


class _VectorStoreFacade(VectorStore):
    """Delegates to an in-process store when vendor client lacks direct VectorStore mapping."""

    def __init__(self, *, collection: str, tenant_id: str, inner: Optional[VectorStore] = None) -> None:
        self._collection = collection
        self._inner = inner or InMemoryVectorStore(tenant_id=tenant_id)

    def add_documents(
        self,
        documents: Sequence[Document],
        embeddings: Sequence[Sequence[float]],
        *,
        ids: Optional[Sequence[str]] = None,
    ) -> None:
        self._inner.add_documents(documents, embeddings, ids=ids)

    def query(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int,
        metadata_filter: Optional[MetadataFilter] = None,
        include_embeddings: bool = False,
    ) -> list[VectorStoreHit]:
        return self._inner.query(
            query_embedding,
            top_k=top_k,
            metadata_filter=metadata_filter,
            include_embeddings=include_embeddings,
        )

    def delete(self, ids: Sequence[str]) -> None:
        self._inner.delete(ids)

    def count(self) -> int:
        return self._inner.count()


class WeaviateVectorFacade(_VectorStoreFacade):
    def __init__(self, client: Any, *, collection: str, tenant_id: str) -> None:
        super().__init__(collection=collection, tenant_id=tenant_id)
        self._client = client


class MilvusVectorFacade(_VectorStoreFacade):
    def __init__(self, client: Any, *, collection: str, tenant_id: str) -> None:
        super().__init__(collection=collection, tenant_id=tenant_id)
        self._client = client
