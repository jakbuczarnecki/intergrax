# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from langchain_core.documents import Document

from intergrax.integrations.providers.vector_store.inmemory.rag_store import InMemoryVectorStore
from intergrax.rag.vectorstore.contracts.vector_store import MetadataFilter, VectorStoreHit
from intergrax.rag.vectorstore.providers.base_vector_store import BaseVectorStore


@dataclass(frozen=True)
class WeaviateConfig:
    collection_name: str
    tenant_id: str


class WeaviateVectorStore(BaseVectorStore):
    """Weaviate-backed store; hybrid BM25+dense via inner lexical-capable store until full SDK mapping."""

    def __init__(self, cfg: WeaviateConfig, *, client: Any = None) -> None:
        self.cfg = cfg
        self.collection_name = f"{cfg.collection_name}__tenant__{cfg.tenant_id}"
        self._client = client
        self._inner = InMemoryVectorStore(tenant_id=cfg.tenant_id)

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
    ) -> List[VectorStoreHit]:
        return self._inner.query(
            query_embedding,
            top_k=top_k,
            metadata_filter=metadata_filter,
            include_embeddings=include_embeddings,
        )

    def query_hybrid(
        self,
        query_embedding: Sequence[float],
        query_text: str,
        *,
        top_k: int,
        metadata_filter: Optional[MetadataFilter] = None,
        include_embeddings: bool = False,
        alpha: float = 0.5,
    ) -> List[VectorStoreHit]:
        return self._inner.query_hybrid(
            query_embedding,
            query_text,
            top_k=top_k,
            metadata_filter=metadata_filter,
            include_embeddings=include_embeddings,
            alpha=alpha,
        )

    def delete(self, ids: Sequence[str]) -> None:
        self._inner.delete(ids)

    def count(self) -> int:
        return self._inner.count()

    def list_collections(self) -> List[str]:
        if self._client is None:
            return [self.collection_name]
        try:
            collections = self._client.collections.list_all()
            return [str(name) for name in collections.keys()]
        except Exception:
            return [self.collection_name]
