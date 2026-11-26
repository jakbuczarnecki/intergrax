# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.vector_store.weaviate.schema import (
    SCHEMA_VERSION,
    WeaviateSchemaConfig,
    ensure_weaviate_collection,
    metadata_filter_to_weaviate,
)
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    MetadataFilter,
    VectorStoreHit,
    VectorStoreRecord,
    VectorStoreScope,
)
from intergrax.rag.vectorstore.providers.base_vector_store import BaseVectorStore
from intergrax.rag.vectorstore.providers.native_provider_boundary import (
    native_hit,
    provider_metadata,
    require_membership_support,
    validate_query,
    validate_records,
    validate_scope,
)
from intergrax.utils import attribute_access


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
    ``query.hybrid`` (BM25 + vector). Without a native client, CRUD refuses.
    """

    def __init__(self, cfg: WeaviateConfig, *, client: Any = None) -> None:
        self.cfg = cfg
        self.collection_name = cfg.collection_name
        self._client = client
        self._native = bool(cfg.use_native_hybrid and client is not None)
        self._collection: Any = None
        self._tenant_collection: Any = None
        if self._native:
            self._collection = self._ensure_collection()

    def supports_native_hybrid_search(self) -> bool:
        return self._native

    def _native_required(self) -> IntegrationConfigurationError:
        return IntegrationConfigurationError("Weaviate vector store requires a native client for vector operations")

    def _ensure_collection(self) -> Any:
        assert self._client is not None
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

    def _active_collection(self) -> Any:
        if self._tenant_collection is not None:
            return self._tenant_collection
        return self._collection

    def add_records(
        self,
        records: Sequence[VectorStoreRecord],
        *,
        scope: VectorStoreScope,
    ) -> Sequence[str]:
        collection = self._active_collection()
        if not self._native or collection is None:
            raise self._native_required()
        validated = validate_records(records, scope=scope, tenant_id=self.cfg.tenant_id)
        id_list = [record.vector_id for record in validated]
        for record in validated:
            props = {
                "text": record.document.content,
                **provider_metadata(record.document, scope=scope),
                "intergrax_schema_version": self.cfg.schema_version,
            }
            collection.data.insert(
                properties=props,
                vector=record.embedding.tolist(),
                uuid=record.vector_id,
            )
        return id_list

    def query(
        self,
        query_embedding: Sequence[float],
        *,
        scope: VectorStoreScope,
        top_k: int,
        metadata_filter: MetadataFilter | None = None,
        include_embeddings: bool = False,
    ) -> list[VectorStoreHit]:
        if not self._native or self._active_collection() is None:
            raise self._native_required()
        del include_embeddings
        return self._native_hybrid(
            query_embedding,
            "",
            scope=scope,
            top_k=top_k,
            metadata_filter=metadata_filter,
            alpha=1.0,
        )

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
        if not self._native or self._active_collection() is None:
            raise self._native_required()
        del include_embeddings
        if not query_text:
            raise self._native_required()
        return self._native_hybrid(
            query_embedding,
            query_text,
            scope=scope,
            top_k=top_k,
            metadata_filter=metadata_filter,
            alpha=alpha,
        )

    def _native_hybrid(
        self,
        query_embedding: Sequence[float],
        query_text: str,
        *,
        scope: VectorStoreScope,
        top_k: int,
        metadata_filter: MetadataFilter | None,
        alpha: float,
    ) -> list[VectorStoreHit]:
        collection = self._active_collection()
        if collection is None:
            raise self._native_required()
        vector, limit = validate_query(query_embedding, top_k=top_k)
        validate_scope(scope, tenant_id=self.cfg.tenant_id)
        effective_filter = MetadataFilter.for_scope(scope, metadata_filter)
        require_membership_support(effective_filter, provider="weaviate")
        conditions = dict(effective_filter.conditions)
        where_filter = metadata_filter_to_weaviate(
            conditions,
            default_tenant=self.cfg.tenant_id,
        )
        query_kwargs: dict[str, Any] = {
            "query": query_text or " ",
            "vector": vector.tolist(),
            "alpha": float(alpha),
            "limit": limit,
        }
        if where_filter is not None:
            query_kwargs["filters"] = where_filter
        response = collection.query.hybrid(**query_kwargs)
        hits: list[VectorStoreHit] = []
        for rank, obj in enumerate(response.objects):
            props = obj.properties or {}
            hits.append(
                native_hit(
                    vector_id=str(obj.uuid),
                    content=str(props.get("text", "")),
                    metadata=dict(props),
                    similarity_score=float(attribute_access.optional(obj.metadata, "score", 1.0 - rank * 0.01)),
                    rank=rank,
                    scope=scope,
                )
            )
        return hits

    def delete(self, ids: Sequence[str], *, scope: VectorStoreScope) -> None:
        if not self._native or self._active_collection() is None:
            raise self._native_required()
        validate_scope(scope, tenant_id=self.cfg.tenant_id)
        if scope.namespace is not None or scope.workspace_id is not None:
            raise RuntimeError("weaviate scoped delete is unsupported")
        for doc_id in ids:
            self._active_collection().data.delete_by_id(doc_id)

    def count(self, *, scope: VectorStoreScope) -> int:
        if not self._native or self._active_collection() is None:
            raise self._native_required()
        del scope
        agg = self._active_collection().aggregate.over_all(total_count=True)
        return int(agg.total_count or 0)

    def list_collections(self) -> list[str]:
        if self._client is None:
            return [self.collection_name]
        try:
            collections = self._client.collections.list_all()
            return [str(name) for name in collections.keys()]
        except Exception:
            return [self.collection_name]
