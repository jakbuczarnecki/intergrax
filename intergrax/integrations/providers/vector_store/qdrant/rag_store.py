# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from intergrax.utils import attribute_access

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
import uuid

from intergrax.rag.vectorstore.config.vector_config import Metric
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    MetadataFilter,
    VectorStoreContractError,
    VectorStoreHit,
    VectorStoreRecord,
    VectorStoreScope,
)
from intergrax.rag.vectorstore.hybrid.lexical_hybrid import LexicalHybridSupport
from intergrax.rag.vectorstore.providers.base_vector_store import BaseVectorStore
from intergrax.rag.vectorstore.providers.native_provider_boundary import (
    native_hit,
    provider_metadata,
    validate_query,
    validate_records,
    validate_scope,
)
from intergrax.knowledge.contracts.validation import require_non_empty_str

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter as QFilter,
        PointIdsList,
        FilterSelector,
        HasIdCondition,
        IsNullCondition,
    )
except ImportError:
    QdrantClient = None  # type: ignore
    Distance = None      # type: ignore
    VectorParams = None  # type: ignore
    PointStruct = None   # type: ignore
    QFilter = None       # type: ignore
    PointIdsList = None  # type: ignore
    FilterSelector = None  # type: ignore
    HasIdCondition = None  # type: ignore
    IsNullCondition = None  # type: ignore

try:
    from qdrant_client.http.models import (  # type: ignore[no-redef]
        Fusion,
        FusionQuery,
        Prefetch,
        SparseIndexParams,
        SparseVector as QdrantSparseVector,
        SparseVectorParams,
    )
except ImportError:
    Fusion = None  # type: ignore
    FusionQuery = None  # type: ignore
    Prefetch = None  # type: ignore
    SparseIndexParams = None  # type: ignore
    QdrantSparseVector = None  # type: ignore
    SparseVectorParams = None  # type: ignore

from intergrax.rag.vectorstore.sparse.sparse_encoder import SparseEncoder, resolve_sparse_encoder

_DENSE_VECTOR_NAME = "dense"
_SPARSE_VECTOR_NAME = "sparse"
_LOGICAL_ID_METADATA_KEY = "logical_id"


def _normalize_point_id(raw_id: str) -> str | int:
    """Map a logical chunk id to a Qdrant-compatible point id (UUID or unsigned int)."""
    try:
        return str(uuid.UUID(raw_id))
    except ValueError:
        pass
    if raw_id.isdigit():
        return int(raw_id)
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw_id))


def _logical_vector_id(payload: Dict[str, Any]) -> str:
    """Read the portable logical ID; never expose a Qdrant point ID."""
    try:
        return require_non_empty_str(
            payload.get(_LOGICAL_ID_METADATA_KEY),
            field_name=_LOGICAL_ID_METADATA_KEY,
        )
    except (TypeError, ValueError) as exc:
        raise VectorStoreContractError(
            "qdrant point is missing a valid logical vector ID"
        ) from exc



@dataclass(frozen=True)
class QdrantConfig:
    """
    Configuration model for Qdrant vector store provider.

    The provider is responsible for creating and managing
    the Qdrant client instance based on this configuration.
    """

    collection_name: str
    tenant_id: str
    batch_size: int = 256
    metric: Metric = "cosine"

    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    enable_sparse_vectors: bool = False



class QdrantVectorStore(LexicalHybridSupport, BaseVectorStore):
    """
    Literal extraction of Qdrant initialization logic from VectorstoreManager.
    No behavioral changes.
    """

    def __init__(self, cfg: QdrantConfig, *, sparse_encoder: Optional[SparseEncoder] = None) -> None:
        LexicalHybridSupport.__init__(self)
        self.cfg = cfg
        self.collection_name = f"{cfg.collection_name}__tenant__{cfg.tenant_id}"

        self._client = None
        self._dim: Optional[int] = None
        self._payloads: Dict[str, Dict[str, Any]] = {}
        self._sparse_enabled = bool(cfg.enable_sparse_vectors)
        self._sparse_encoder = sparse_encoder or resolve_sparse_encoder()

        self._init_qdrant()

    @staticmethod
    def _collection_vector_size(collection_info: Any) -> int | None:
        try:
            vectors = collection_info.config.params.vectors
        except Exception:
            return None
        if vectors is None:
            return None
        if isinstance(vectors, dict):
            dense = vectors.get(_DENSE_VECTOR_NAME)
            if dense is not None:
                return int(dense.size)
            if len(vectors) == 1:
                only = next(iter(vectors.values()))
                return int(only.size)
            return None
        return int(vectors.size)

    @staticmethod
    def _collection_point_count(collection_info: Any) -> int | None:
        try:
            count = collection_info.points_count
            if count is None:
                return None
            return int(count)
        except Exception:
            return None

    def _raise_embedding_dimension_mismatch(self) -> None:
        raise VectorStoreContractError("qdrant_embedding_dimension_mismatch")

    def _recreate_empty_incompatible_collection(self, collection_info: Any) -> None:
        point_count = self._collection_point_count(collection_info)
        if point_count is None or point_count > 0:
            self._raise_embedding_dimension_mismatch()
        assert self._client is not None
        self._client.delete_collection(self.collection_name)

    def _init_qdrant(self) -> None:
        if QdrantClient is None:
            raise ImportError("qdrant-client is not installed. `pip install qdrant-client`")

        if self.cfg.qdrant_url:
            self._client = QdrantClient(
                url=self.cfg.qdrant_url,
                api_key=self.cfg.qdrant_api_key,
            )
        else:
            # Local default
            self._client = QdrantClient(
                host="localhost",
                port=6333,
                api_key=self.cfg.qdrant_api_key,
            )

    def _create_qdrant_collection(self) -> None:
        assert self._client is not None
        assert self._dim is not None

        metric_map = {
            "cosine": Distance.COSINE,
            "dot": Distance.DOT,
            "euclidean": Distance.EUCLID,
        }
        dist = metric_map.get(self.cfg.metric, Distance.COSINE)

        if self._sparse_enabled and SparseVectorParams is not None:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    _DENSE_VECTOR_NAME: VectorParams(size=self._dim, distance=dist),
                },
                sparse_vectors_config={
                    _SPARSE_VECTOR_NAME: SparseVectorParams(index=SparseIndexParams()),
                },
            )
        else:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self._dim, distance=dist),
            )

    def _collection_exists(self) -> bool:
        assert self._client is not None
        try:
            self._client.get_collection(self.collection_name)
            return True
        except VectorStoreContractError:
            raise
        except Exception:
            return False

    def _ensure_qdrant_collection(self) -> None:
        assert self._client is not None

        info = None
        try:
            info = self._client.get_collection(self.collection_name)
        except VectorStoreContractError:
            raise
        except Exception:
            info = None

        if info is not None:
            if self._dim is not None:
                existing_dim = self._collection_vector_size(info)
                if existing_dim is not None and existing_dim != self._dim:
                    self._recreate_empty_incompatible_collection(info)
                else:
                    return
            else:
                return

        if self._dim is None:
            return

        self._create_qdrant_collection()

    def _qdrant_filter(self, where: Optional[Dict[str, Any]]) -> Optional[QFilter]:  # type: ignore
        """Equality-only helper for scroll/search paths that use plain dicts."""
        if not where or QFilter is None:
            return None
        must: List[Dict[str, Any]] = []
        for k, v in where.items():
            must.append({"key": k, "match": {"value": v}})
        return QFilter(**{"must": must})

    def _qdrant_filter_from_metadata(
        self,
        metadata_filter: MetadataFilter,
    ) -> Optional[QFilter]:  # type: ignore
        if QFilter is None:
            return None
        must: List[Dict[str, Any]] = []
        for key, value in metadata_filter.conditions.items():
            must.append({"key": key, "match": {"value": value}})
        for condition in metadata_filter.membership:
            must.append(
                {
                    "key": condition.field,
                    "match": {"any": list(condition.allowed_values)},
                }
            )
        if not must:
            return None
        return QFilter(**{"must": must})
    

    def _upsert_qdrant(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[Dict[str, Any]],
    ) -> None:
        self._ensure_qdrant_collection()
        points = []
        for i in range(len(ids)):
            raw_id = str(ids[i])
            point_id = _normalize_point_id(raw_id)
            payload = dict(metadatas[i])
            payload[_LOGICAL_ID_METADATA_KEY] = raw_id
            text = str(payload.get("text", ""))
            if self._sparse_enabled and QdrantSparseVector is not None:
                sparse = self._sparse_encoder.encode(text)
                points.append(
                    PointStruct(
                        id=point_id,
                        vector={
                            _DENSE_VECTOR_NAME: list(map(float, embeddings[i])),
                            _SPARSE_VECTOR_NAME: QdrantSparseVector(
                                indices=sparse.indices,
                                values=sparse.values,
                            ),
                        },
                        payload=payload,
                    )
                )
            else:
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=list(map(float, embeddings[i])),
                        payload=payload,
                    )
                )
        self._client.upsert(collection_name=self.collection_name, points=points)


    def add_records(
        self,
        records: Sequence[VectorStoreRecord],
        *,
        scope: VectorStoreScope,
    ) -> Sequence[str]:
        validated = validate_records(records, scope=scope, tenant_id=self.cfg.tenant_id)
        if not validated:
            return []
        ids_list = [record.vector_id for record in validated]
        X = [record.embedding.tolist() for record in validated]
        self._dim = self._dim or len(X[0])

        for start in range(0, len(validated), self.cfg.batch_size):
            end = min(start + self.cfg.batch_size, len(validated))
            ids_batch = ids_list[start:end]
            embeddings_batch = X[start:end]
            self._ensure_dim_consistency(embeddings_batch)

            records_batch = validated[start:end]
            metas_batch = [
                {
                    **provider_metadata(record.document, scope=scope),
                    "text": record.document.content,
                }
                for record in records_batch
            ]

            self._upsert_qdrant(ids_batch, embeddings_batch, metas_batch)
            for doc_id, meta in zip(ids_batch, metas_batch):
                self._payloads[str(doc_id)] = dict(meta)
                self._index_lexical(str(doc_id), str(meta.get("text", "")))
        return ids_list
    

    def query(
        self,
        query_embedding: Sequence[float],
        *,
        scope: VectorStoreScope,
        top_k: int,
        metadata_filter: Optional[MetadataFilter] = None,
        include_embeddings: bool = False,
    ) -> List[VectorStoreHit]:
        if QFilter is None:
            raise RuntimeError("Qdrant client is not available.")

        vector, limit = validate_query(query_embedding, top_k=top_k)
        validate_scope(scope, tenant_id=self.cfg.tenant_id)
        effective_filter = MetadataFilter.for_scope(scope, metadata_filter)
        qfilter = self._qdrant_filter_from_metadata(effective_filter)
        using = _DENSE_VECTOR_NAME if self._sparse_enabled else None
        try:
            results = self._client.query_points(
                collection_name=self.collection_name,
                query=vector,
                using=using,
                query_filter=qfilter,
                limit=limit,
                with_payload=True,
                with_vectors=include_embeddings,
            )
        except Exception:
            return []

        hits: List[VectorStoreHit] = []

        for rank, r in enumerate(results.points):
            payload = r.payload or {}
            text = payload.get("text", "")

            hits.append(
                native_hit(
                    vector_id=_logical_vector_id(payload),
                    content=text,
                    metadata=payload,
                    similarity_score=float(r.score),
                    rank=rank,
                    scope=scope,
                    embedding=(
                        (
                            list(r.vector.values())
                            if isinstance(r.vector, dict)
                            else r.vector
                        )
                        if include_embeddings
                        else None
                    ),
                )
            )

        return hits

    def query_hybrid(
        self,
        query_embedding: Sequence[float],
        query_text: str,
        *,
        scope: VectorStoreScope,
        top_k: int,
        metadata_filter: Optional[MetadataFilter] = None,
        include_embeddings: bool = False,
        alpha: float = 0.5,
    ) -> List[VectorStoreHit]:
        if (
            self._sparse_enabled
            and Prefetch is not None
            and FusionQuery is not None
            and QdrantSparseVector is not None
        ):
            hits = self._query_qdrant_fusion(
                query_embedding,
                query_text,
                scope=scope,
                top_k=top_k,
                metadata_filter=metadata_filter,
                include_embeddings=include_embeddings,
            )
            if hits:
                return hits
        return super().query_hybrid(
            query_embedding,
            query_text,
            scope=scope,
            top_k=top_k,
            metadata_filter=metadata_filter,
            include_embeddings=include_embeddings,
            alpha=alpha,
        )

    def _query_qdrant_fusion(
        self,
        query_embedding: Sequence[float],
        query_text: str,
        *,
        scope: VectorStoreScope,
        top_k: int,
        metadata_filter: Optional[MetadataFilter],
        include_embeddings: bool,
    ) -> List[VectorStoreHit]:
        assert self._client is not None
        vector, limit = validate_query(query_embedding, top_k=top_k)
        validate_scope(scope, tenant_id=self.cfg.tenant_id)
        sparse = self._sparse_encoder.encode(query_text)

        effective_filter = MetadataFilter.for_scope(scope, metadata_filter)
        qfilter = self._qdrant_filter_from_metadata(effective_filter)

        prefetch_k = max(limit * 3, limit)
        try:
            results = self._client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    Prefetch(
                        query=QdrantSparseVector(indices=sparse.indices, values=sparse.values),
                        using=_SPARSE_VECTOR_NAME,
                        filter=qfilter,
                        limit=prefetch_k,
                    ),
                    Prefetch(
                        query=vector,
                        using=_DENSE_VECTOR_NAME,
                        filter=qfilter,
                        limit=prefetch_k,
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=limit,
                with_payload=True,
                with_vectors=include_embeddings,
            )
        except Exception:
            return []

        hits: List[VectorStoreHit] = []
        for rank, r in enumerate(results.points):
            payload = r.payload or {}
            hits.append(
                native_hit(
                    vector_id=_logical_vector_id(payload),
                    content=str(payload.get("text", "")),
                    metadata={**payload, "qdrant_hybrid": True},
                    similarity_score=float(r.score),
                    rank=rank,
                    scope=scope,
                    embedding=(
                        (
                            list(r.vector.values())
                            if isinstance(r.vector, dict)
                            else r.vector
                        )
                        if include_embeddings and r.vector
                        else None
                    ),
                )
            )
        return hits

    def delete(self, ids: Sequence[str], *, scope: VectorStoreScope) -> None:
        if not ids:
            return
        validate_scope(scope, tenant_id=self.cfg.tenant_id)
        self._ensure_qdrant_collection()
        point_ids = [_normalize_point_id(str(point_id)) for point_id in ids]
        qfilter = self._qdrant_filter(
            dict(MetadataFilter.for_scope(scope, None).conditions)
        )
        try:
            if FilterSelector is not None and HasIdCondition is not None and qfilter is not None:
                qfilter.must.append(HasIdCondition(has_id=point_ids))
                self._client.delete(
                    collection_name=self.collection_name,
                    points_selector=FilterSelector(filter=qfilter),
                )
            elif PointIdsList is not None and scope.namespace is None and scope.workspace_id is None:
                self._client.delete(
                    collection_name=self.collection_name,
                    points_selector=PointIdsList(points=point_ids),
                )
            else:
                raise VectorStoreContractError(
                    "qdrant scoped delete is unsupported"
                )
        except TypeError as exc:
            raise VectorStoreContractError(
                "qdrant scoped delete is unsupported"
            ) from exc

    def list_source_record_ids(
        self,
        *,
        source_id: str,
        scope: VectorStoreScope,
    ) -> Sequence[str]:
        canonical_source_id = require_non_empty_str(source_id, field_name="source_id")
        validate_scope(scope, tenant_id=self.cfg.tenant_id)
        if not self._collection_exists():
            return []
        self._ensure_qdrant_collection()
        if QFilter is None:
            raise RuntimeError("qdrant source record lookup is unavailable")

        effective_where: Dict[str, Any] = {
            "tenant_id": scope.tenant_id,
            "source_id": canonical_source_id,
        }
        if scope.namespace is not None:
            effective_where["namespace"] = scope.namespace
        if scope.workspace_id is not None:
            effective_where["workspace_id"] = scope.workspace_id
        qfilter = self._qdrant_filter(effective_where)
        if qfilter is None:
            raise RuntimeError("qdrant source record lookup filter is unavailable")
        if IsNullCondition is None:
            if scope.namespace is None or scope.workspace_id is None:
                raise RuntimeError("qdrant null-scope filtering is unavailable")
        else:
            if scope.namespace is None:
                qfilter.must.append(
                    IsNullCondition(is_null={"key": "namespace"})
                )
            if scope.workspace_id is None:
                qfilter.must.append(
                    IsNullCondition(is_null={"key": "workspace_id"})
                )

        ids: list[str] = []
        next_offset: Any = None
        while True:
            records, next_offset = self._client.scroll(
                collection_name=self.collection_name,
                scroll_filter=qfilter,
                limit=256,
                offset=next_offset,
                with_payload=[_LOGICAL_ID_METADATA_KEY],
                with_vectors=False,
            )
            if not records:
                break
            ids.extend(_logical_vector_id(point.payload or {}) for point in records)
            if next_offset is None:
                break
        return sorted(ids)


    def count(self, *, scope: VectorStoreScope) -> int:
        validate_scope(scope, tenant_id=self.cfg.tenant_id)
        try:
            self._ensure_qdrant_collection()
            qfilter = self._qdrant_filter(
                dict(MetadataFilter.for_scope(scope, None).conditions)
            )
            c = self._client.count(
                self.collection_name,
                count_filter=qfilter,
                exact=True,
            )
            return int(attribute_access.optional(c, "count", 0))
        except Exception:            
            return 0

    def list_collections(self) -> List[str]:
        return [self.collection_name]

    def list_document_ids(self, *, limit: int = 100, offset: int = 0) -> List[str]:
        self._ensure_qdrant_collection()
        ids: List[str] = []
        next_offset: Any = None
        skipped = 0
        target_offset = max(0, offset)
        while len(ids) < max(1, limit):
            records, next_offset = self._client.scroll(
                collection_name=self.collection_name,
                limit=min(128, max(1, limit) + target_offset),
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )
            if not records:
                break
            for point in records:
                payload = point.payload or {}
                logical = _logical_vector_id(payload)
                if skipped < target_offset:
                    skipped += 1
                    continue
                ids.append(logical)
                if len(ids) >= max(1, limit):
                    break
            if next_offset is None:
                break
        return ids

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        logical = (document_id or "").strip()
        if not logical:
            return None
        cached = self._payloads.get(logical)
        if cached is not None:
            metadata = {key: value for key, value in cached.items() if key != "text"}
            return {
                "id": logical,
                "text": str(cached.get("text") or ""),
                "metadata": metadata,
            }
        self._ensure_qdrant_collection()
        point_id = _normalize_point_id(logical)
        try:
            points = self._client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_payload=True,
                with_vectors=False,
            )
        except Exception:
            return None
        if not points:
            return None
        payload = points[0].payload or {}
        logical = _logical_vector_id(payload)
        metadata = {key: value for key, value in payload.items() if key != "text"}
        return {
            "id": logical,
            "text": str(payload.get("text") or ""),
            "metadata": metadata,
        }

    def search_by_metadata(
        self,
        *,
        conditions: Dict[str, Any],
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        effective_where = dict(conditions)
        existing = effective_where.get("tenant_id")
        if existing is not None and existing != self.cfg.tenant_id:
            raise ValueError(
                f"Query tenant_id mismatch: expected '{self.cfg.tenant_id}', got '{existing}'."
            )
        effective_where["tenant_id"] = self.cfg.tenant_id
        self._ensure_qdrant_collection()
        qfilter = self._qdrant_filter(effective_where)
        results: List[Dict[str, Any]] = []
        next_offset: Any = None
        page_limit = max(1, min(128, int(limit)))
        while len(results) < max(1, limit):
            records, next_offset = self._client.scroll(
                collection_name=self.collection_name,
                scroll_filter=qfilter,
                limit=min(page_limit, max(1, limit) - len(results)),
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )
            if not records:
                break
            for point in records:
                payload = point.payload or {}
                logical = _logical_vector_id(payload)
                metadata = {key: value for key, value in payload.items() if key != "text"}
                results.append(
                    {
                        "id": logical,
                        "text": str(payload.get("text") or ""),
                        "metadata": metadata,
                    }
                )
                if len(results) >= max(1, limit):
                    break
            if next_offset is None:
                break
        return results

    def purge_collection(self, *, dry_run: bool = True, tenant_id: str = "") -> Dict[str, Any]:
        if tenant_id and tenant_id != self.cfg.tenant_id:
            raise ValueError(
                f"Purge tenant_id mismatch: expected '{self.cfg.tenant_id}', got '{tenant_id}'."
            )
        document_count = self.count(
            scope=VectorStoreScope(tenant_id=self.cfg.tenant_id)
        )
        if dry_run:
            return {
                "dry_run": True,
                "would_delete": document_count,
                "tenant_id": self.cfg.tenant_id,
            }
        deleted = 0
        while True:
            more = self.list_document_ids(limit=500, offset=0)
            if not more:
                break
            self.delete(
                more,
                scope=VectorStoreScope(tenant_id=self.cfg.tenant_id),
            )
            deleted += len(more)
            if len(more) < 500:
                break
        self._payloads.clear()
        return {
            "dry_run": False,
            "deleted": deleted or document_count,
            "tenant_id": self.cfg.tenant_id,
        }

