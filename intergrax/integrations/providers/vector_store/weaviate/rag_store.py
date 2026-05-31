# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.documents import Document

from intergrax.integrations.providers.vector_store.inmemory.rag_store import InMemoryVectorStore
from intergrax.integrations.providers.vector_store.weaviate.schema import (
    SCHEMA_VERSION,
    WeaviateSchemaConfig,
    ensure_weaviate_collection,
    metadata_filter_to_weaviate,
)
from intergrax.rag.vectorstore.contracts.vector_store import MetadataFilter, VectorStoreHit
from intergrax.rag.vectorstore.providers.base_vector_store import BaseVectorStore


@dataclass(frozen=True)
class WeaviateConfig:
    collection_name: str
    tenant_id: str
    use_native_hybrid: bool = True
    multi_tenant: bool = True
    schema_version: int = SCHEMA_VERSION


class WeaviateVectorStore(BaseVectorStore):
    """
    Weaviate-backed vector store with schema migration and native multi-tenancy.

    When a live ``client`` is provided and ``use_native_hybrid`` is true, uses Weaviate
    ``query.hybrid`` (BM25 + vector). Otherwise delegates to an in-memory lexical-capable store.
    """

    def __init__(self, cfg: WeaviateConfig, *, client: Any = None) -> None:
        self.cfg = cfg
        self.collection_name = cfg.collection_name
        self._client = client
        self._inner = InMemoryVectorStore(tenant_id=cfg.tenant_id)
        self._native = bool(cfg.use_native_hybrid and client is not None)
        self._collection: Any = None
        self._tenant_collection: Any = None
        if self._native:
            self._collection = self._ensure_collection()

    def _ensure_collection(self) -> Any:
        assert self._client is not None
        try:
            schema_cfg = WeaviateSchemaConfig(
                collection_name=self.collection_name,
                schema_version=self.cfg.schema_version,
                multi_tenant=self.cfg.multi_tenant,
                tenant_id=self.cfg.tenant_id,
            )
            collection = ensure_weaviate_collection(self._client, schema_cfg)
            if self.cfg.multi_tenant:
                self._tenant_collection = collection.with_tenant(self.cfg.tenant_id)
            else:
                self._tenant_collection = collection
            return collection
        except Exception:
            self._native = False
            return None

    def _active_collection(self) -> Any:
        if self._tenant_collection is not None:
            return self._tenant_collection
        return self._collection

    def add_documents(
        self,
        documents: Sequence[Document],
        embeddings: Sequence[Sequence[float]],
        *,
        ids: Optional[Sequence[str]] = None,
    ) -> None:
        if self._native and self._active_collection() is not None:
            n = len(documents)
            id_list = list(ids) if ids else [str(uuid.uuid4()) for _ in range(n)]
            for i, doc in enumerate(documents):
                props = {
                    "text": doc.page_content or "",
                    "tenant_id": self.cfg.tenant_id,
                    "doc_id": str((doc.metadata or {}).get("doc_id", id_list[i])),
                    "intergrax_schema_version": self.cfg.schema_version,
                    **(doc.metadata or {}),
                }
                try:
                    self._active_collection().data.insert(
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
        if self._native and self._active_collection() is not None:
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
        if self._native and self._active_collection() is not None and query_text:
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
            collection = self._active_collection()
            assert collection is not None
            conditions = dict(metadata_filter.conditions) if metadata_filter else {}
            where_filter = metadata_filter_to_weaviate(
                conditions,
                default_tenant=self.cfg.tenant_id,
            )
            query_kwargs: Dict[str, Any] = {
                "query": query_text or " ",
                "vector": list(map(float, query_embedding)),
                "alpha": float(alpha),
                "limit": top_k,
            }
            if where_filter is not None:
                query_kwargs["filters"] = where_filter
            response = collection.query.hybrid(**query_kwargs)
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
        if self._native and self._active_collection() is not None:
            for doc_id in ids:
                try:
                    self._active_collection().data.delete_by_id(doc_id)
                except Exception:
                    pass
        self._inner.delete(ids)

    def count(self) -> int:
        if self._native and self._active_collection() is not None:
            try:
                agg = self._active_collection().aggregate.over_all(total_count=True)
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
