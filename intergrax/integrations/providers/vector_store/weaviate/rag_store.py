# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.documents import Document

from intergrax.integrations.providers.vector_store.inmemory.rag_store import InMemoryVectorStore
from intergrax.rag.vectorstore.contracts.vector_store import MetadataFilter, VectorStoreHit
from intergrax.rag.vectorstore.providers.base_vector_store import BaseVectorStore


@dataclass(frozen=True)
class WeaviateConfig:
    collection_name: str
    tenant_id: str
    use_native_hybrid: bool = True


class WeaviateVectorStore(BaseVectorStore):
    """
    Weaviate-backed vector store.

    When a live ``client`` is provided and ``use_native_hybrid`` is true, uses Weaviate
    ``query.hybrid`` (BM25 + vector). Otherwise delegates to an in-memory lexical-capable store.
    """

    def __init__(self, cfg: WeaviateConfig, *, client: Any = None) -> None:
        self.cfg = cfg
        self.collection_name = f"{cfg.collection_name}__tenant__{cfg.tenant_id}"
        self._client = client
        self._inner = InMemoryVectorStore(tenant_id=cfg.tenant_id)
        self._native = bool(cfg.use_native_hybrid and client is not None)
        self._collection: Any = None
        if self._native:
            self._collection = self._ensure_collection()

    def _ensure_collection(self) -> Any:
        assert self._client is not None
        try:
            from weaviate.classes.config import Configure, DataType, Property

            if not self._client.collections.exists(self.collection_name):
                self._client.collections.create(
                    name=self.collection_name,
                    properties=[Property(name="text", data_type=DataType.TEXT)],
                    vectorizer_config=Configure.Vectorizer.none(),
                )
            return self._client.collections.get(self.collection_name)
        except Exception:
            self._native = False
            return None

    def add_documents(
        self,
        documents: Sequence[Document],
        embeddings: Sequence[Sequence[float]],
        *,
        ids: Optional[Sequence[str]] = None,
    ) -> None:
        if self._native and self._collection is not None:
            n = len(documents)
            id_list = list(ids) if ids else [str(uuid.uuid4()) for _ in range(n)]
            for i, doc in enumerate(documents):
                props = {
                    "text": doc.page_content or "",
                    "tenant_id": self.cfg.tenant_id,
                    **(doc.metadata or {}),
                }
                try:
                    self._collection.data.insert(
                        properties=props,
                        vector=list(map(float, embeddings[i])),
                        uuid=id_list[i],
                    )
                except Exception:
                    self._native = False
                    break
        if not self._native:
            self._inner.add_documents(documents, embeddings, ids=ids)

    def query(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int,
        metadata_filter: Optional[MetadataFilter] = None,
        include_embeddings: bool = False,
    ) -> List[VectorStoreHit]:
        if self._native and self._collection is not None:
            hits = self._native_hybrid(
                query_embedding,
                "",
                top_k=top_k,
                metadata_filter=metadata_filter,
                alpha=1.0,
            )
            if hits is not None:
                return hits
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
        if self._native and self._collection is not None and query_text:
            hits = self._native_hybrid(
                query_embedding,
                query_text,
                top_k=top_k,
                metadata_filter=metadata_filter,
                alpha=alpha,
            )
            if hits is not None:
                return hits
        return self._inner.query_hybrid(
            query_embedding,
            query_text,
            top_k=top_k,
            metadata_filter=metadata_filter,
            include_embeddings=include_embeddings,
            alpha=alpha,
        )

    def _native_hybrid(
        self,
        query_embedding: Sequence[float],
        query_text: str,
        *,
        top_k: int,
        metadata_filter: Optional[MetadataFilter],
        alpha: float,
    ) -> Optional[List[VectorStoreHit]]:
        try:
            assert self._collection is not None
            where_filter = None
            if metadata_filter is not None:
                tenant = metadata_filter.conditions.get("tenant_id", self.cfg.tenant_id)
                where_filter = {
                    "path": ["tenant_id"],
                    "operator": "Equal",
                    "valueText": str(tenant),
                }
            response = self._collection.query.hybrid(
                query=query_text or " ",
                vector=list(map(float, query_embedding)),
                alpha=float(alpha),
                limit=top_k,
                filters=where_filter,
            )
            hits: List[VectorStoreHit] = []
            for rank, obj in enumerate(response.objects):
                props = obj.properties or {}
                hits.append(
                    VectorStoreHit(
                        id=str(obj.uuid),
                        content=str(props.get("text", "")),
                        metadata=dict(props),
                        similarity_score=float(getattr(obj.metadata, "score", 1.0 - rank * 0.01)),
                        rank=rank,
                    )
                )
            return hits
        except Exception:
            return None

    def delete(self, ids: Sequence[str]) -> None:
        if self._native and self._collection is not None:
            for doc_id in ids:
                try:
                    self._collection.data.delete_by_id(doc_id)
                except Exception:
                    pass
        self._inner.delete(ids)

    def count(self) -> int:
        if self._native and self._collection is not None:
            try:
                agg = self._collection.aggregate.over_all(total_count=True)
                return int(agg.total_count or 0)
            except Exception:
                pass
        return self._inner.count()

    def list_collections(self) -> List[str]:
        if self._client is None:
            return [self.collection_name]
        try:
            collections = self._client.collections.list_all()
            return [str(name) for name in collections.keys()]
        except Exception:
            return [self.collection_name]
