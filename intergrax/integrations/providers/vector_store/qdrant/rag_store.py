# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
import uuid

from langchain_core.documents import Document

from intergrax.rag.embedding.contracts.embedding_metadata_key import EmbeddingMetadataKey
from intergrax.rag.vectorstore.config.vector_config import Metric
from intergrax.rag.vectorstore.contracts.vector_store import MetadataFilter, VectorStoreHit
from intergrax.rag.vectorstore.hybrid.lexical_hybrid import LexicalHybridSupport
from intergrax.rag.vectorstore.providers.base_vector_store import BaseVectorStore

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter as QFilter,
        PointIdsList,
    )
except ImportError:
    QdrantClient = None  # type: ignore
    Distance = None      # type: ignore
    VectorParams = None  # type: ignore
    PointStruct = None   # type: ignore
    QFilter = None       # type: ignore
    PointIdsList = None  # type: ignore

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

    def _ensure_qdrant_collection(self) -> None:
        assert self._client is not None

        try:
            self._client.get_collection(self.collection_name)
            return
        except Exception:
            pass

        if self._dim is None:
            return

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

    def _qdrant_filter(self, where: Optional[Dict[str, Any]]) -> Optional[QFilter]:  # type: ignore
        """Lightweight helper: simple dict -> Filter(must=[FieldCondition(...)])."""
        if not where or QFilter is None:
            return None
        must: List[Dict[str, Any]] = []
        for k, v in where.items():
            # simple equality
            must.append({"key": k, "match": {"value": v}})
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
            text = str(metadatas[i].get("text", ""))
            if self._sparse_enabled and QdrantSparseVector is not None:
                sparse = self._sparse_encoder.encode(text)
                points.append(
                    PointStruct(
                        id=ids[i],
                        vector={
                            _DENSE_VECTOR_NAME: list(map(float, embeddings[i])),
                            _SPARSE_VECTOR_NAME: QdrantSparseVector(
                                indices=sparse.indices,
                                values=sparse.values,
                            ),
                        },
                        payload=metadatas[i],
                    )
                )
            else:
                points.append(
                    PointStruct(
                        id=ids[i],
                        vector=list(map(float, embeddings[i])),
                        payload=metadatas[i],
                    )
                )
        self._client.upsert(collection_name=self.collection_name, points=points)


    def add_documents(
        self,
        documents: Sequence[Document],
        embeddings: Sequence[Sequence[float]],
        *,
        ids: Optional[Sequence[str]] = None,
    ) -> None:
        if len(documents) == 0:
            return

        
        X = self._to_list_of_lists(embeddings)
        if len(X) != len(documents):
            raise ValueError("Number of documents must match number of embeddings")

        n = len(documents)
        ids_list = list(ids) if ids else [str(uuid.uuid4()) for _ in range(n)]
        if len(ids_list) != n:
            raise ValueError("Length of `ids` must match number of documents")

        first_dim = len(X[0]) if X and X[0] else None
        if first_dim is None:
            raise ValueError("Embeddings appear empty/corrupt; cannot infer dimension.")
        self._dim = self._dim or first_dim

        for start in range(0, n, self.cfg.batch_size):
            end = min(start + self.cfg.batch_size, n)
            ids_batch = ids_list[start:end]
            embeddings_batch = X[start:end]
            self._ensure_dim_consistency(embeddings_batch)

            docs_batch = documents[start:end]
            metas_batch = self._doc_payloads(docs_batch, base=None)

            for i in range(len(metas_batch)):
                if EmbeddingMetadataKey.VECTOR in metas_batch[i]:
                    metas_batch[i].pop(EmbeddingMetadataKey.VECTOR, None)

            # Tenant metadata enforcement
            for i in range(len(metas_batch)):
                existing = metas_batch[i].get("tenant_id")
                if existing is not None and existing != self.cfg.tenant_id:
                    raise ValueError(
                        f"Metadata tenant_id mismatch: expected '{self.cfg.tenant_id}', got '{existing}'."
                    )
                metas_batch[i]["tenant_id"] = self.cfg.tenant_id

            # Store raw text inside metadata (Qdrant branch behavior)
            for i, d in enumerate(docs_batch):
                metas_batch[i] = dict(metas_batch[i], text=d.page_content or "")

            self._upsert_qdrant(ids_batch, embeddings_batch, metas_batch)
            for doc_id, meta in zip(ids_batch, metas_batch):
                self._payloads[str(doc_id)] = dict(meta)
                self._index_lexical(str(doc_id), str(meta.get("text", "")))
    

    def query(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int,
        metadata_filter: Optional[MetadataFilter] = None,
        include_embeddings: bool = False,
    ) -> List[VectorStoreHit]:
        if QFilter is None:
            raise RuntimeError("Qdrant client is not available.")

        # Convert embedding to float list (exactly as in manager)
        vector = list(map(float, query_embedding))

        # Enforce tenant filter (exact logic from manager)
        effective_where: Dict[str, Any] = (
            dict(metadata_filter.conditions)
            if metadata_filter is not None
            else {}
        )
        existing = effective_where.get("tenant_id")
        if existing is not None and existing != self.cfg.tenant_id:
            raise ValueError(
                f"Query tenant_id mismatch: expected '{self.cfg.tenant_id}', got '{existing}'."
            )
        effective_where["tenant_id"] = self.cfg.tenant_id

        qfilter = self._qdrant_filter(effective_where)
        using = _DENSE_VECTOR_NAME if self._sparse_enabled else None
        try:
            results = self._client.query_points(
                collection_name=self.collection_name,
                query=vector,
                using=using,
                query_filter=qfilter,
                limit=top_k,
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
                VectorStoreHit(
                    id=str(r.id),
                    content=text,
                    metadata=payload,
                    similarity_score=float(r.score),
                    rank=rank,
                    embedding=list(r.vector) if include_embeddings else None,
                )
            )

        return hits

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
        if (
            self._sparse_enabled
            and Prefetch is not None
            and FusionQuery is not None
            and QdrantSparseVector is not None
        ):
            hits = self._query_qdrant_fusion(
                query_embedding,
                query_text,
                top_k=top_k,
                metadata_filter=metadata_filter,
                include_embeddings=include_embeddings,
            )
            if hits:
                return hits
        return super().query_hybrid(
            query_embedding,
            query_text,
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
        top_k: int,
        metadata_filter: Optional[MetadataFilter],
        include_embeddings: bool,
    ) -> List[VectorStoreHit]:
        assert self._client is not None
        vector = list(map(float, query_embedding))
        sparse = self._sparse_encoder.encode(query_text)

        effective_where: Dict[str, Any] = (
            dict(metadata_filter.conditions) if metadata_filter is not None else {}
        )
        effective_where["tenant_id"] = self.cfg.tenant_id
        qfilter = self._qdrant_filter(effective_where)

        prefetch_k = max(top_k * 3, top_k)
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
                limit=top_k,
                with_payload=True,
                with_vectors=include_embeddings,
            )
        except Exception:
            return []

        hits: List[VectorStoreHit] = []
        for rank, r in enumerate(results.points):
            payload = r.payload or {}
            hits.append(
                VectorStoreHit(
                    id=str(r.id),
                    content=str(payload.get("text", "")),
                    metadata={**payload, "qdrant_hybrid": True},
                    similarity_score=float(r.score),
                    rank=rank,
                    embedding=list(r.vector) if include_embeddings and r.vector else None,
                )
            )
        return hits

    def delete(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        self._ensure_qdrant_collection()
        try:
            if PointIdsList is not None:
                self._client.delete(
                    collection_name=self.collection_name,
                    points_selector=PointIdsList(points=list(ids)),
                )
            else:
                self._client.delete(
                    self.collection_name,
                    points_selector={"points": list(ids)},
                )
        except TypeError:
            self._client.delete(
                self.collection_name,
                points_selector={"points": list(ids)},
            )


    def count(self) -> int:
        try:
            self._ensure_qdrant_collection()
            c = self._client.count(self.collection_name, exact=True)
            return int(getattr(c, "count", 0))
        except Exception:            
            return 0