# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from intergrax.integrations.providers.vector_store.inmemory.rag_store import InMemoryVectorStore
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    MetadataFilter,
    VectorStoreHit,
    VectorStoreRecord,
    VectorStoreScope,
)
from intergrax.rag.vectorstore.providers.base_vector_store import BaseVectorStore


@dataclass(frozen=True)
class MilvusConfig:
    collection_name: str
    tenant_id: str


class MilvusVectorStore(BaseVectorStore):
    """Milvus-backed vector store — CRUD delegates to in-memory until full SDK mapping lands."""

    def __init__(self, cfg: MilvusConfig, *, client: Any = None) -> None:
        self.cfg = cfg
        self.collection_name = f"{cfg.collection_name}__tenant__{cfg.tenant_id}"
        self._client = client
        self._inner = InMemoryVectorStore(tenant_id=cfg.tenant_id)

    def add_records(
        self,
        records: Sequence[VectorStoreRecord],
        *,
        scope: VectorStoreScope,
    ) -> Sequence[str]:
        return self._inner.add_records(records, scope=scope)

    def query(
        self,
        query_embedding: Sequence[float],
        *,
        scope: VectorStoreScope,
        top_k: int,
        metadata_filter: Optional[MetadataFilter] = None,
        include_embeddings: bool = False,
    ) -> List[VectorStoreHit]:
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

    def list_collections(self) -> List[str]:
        if self._client is None:
            return [self.collection_name]
        try:
            names = self._client.list_collections()
            return [str(name) for name in names]
        except Exception:
            return [self.collection_name]
