# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
import uuid

from langchain_core.documents import Document

from intergrax.rag.vectorstore.config.vector_config import Metric
from intergrax.rag.vectorstore.contracts.vector_store import MetadataFilter, VectorStoreHit
from intergrax.rag.vectorstore.providers.base_vector_store import BaseVectorStore

try:
    from pinecone import Pinecone
except ImportError:
    Pinecone = None  # type: ignore


@dataclass(frozen=True)
class PineconeConfig:
    """
    Configuration model for Pinecone vector store provider.
    """

    collection_name: str
    tenant_id: str
    metric: Metric = "cosine"
    batch_size: int = 100

    pinecone_api_key: Optional[str] = None
    pinecone_index_name: Optional[str] = None
    pinecone_cloud: Optional[str] = None
    pinecone_region: Optional[str] = None


class PineconeVectorStore(BaseVectorStore):
    """
    Literal extraction of Pinecone initialization logic from VectorstoreManager.
    No behavioral changes.
    """

    def __init__(self, cfg: PineconeConfig) -> None:
        self.cfg = cfg

        # Pinecone does NOT use __tenant__ suffix in manager implementation
        self.collection_name = cfg.collection_name

        self._client = None
        self._collection = None
        self._dim = None

        self._init_pinecone()

    def _init_pinecone(self) -> None:
        if Pinecone is None:
            raise ImportError("pinecone client is not installed. `pip install pinecone-client`")

        if not self.cfg.pinecone_api_key:
            raise ValueError("Pinecone requires `pinecone_api_key` in PineconeConfig.")

        pc = Pinecone(api_key=self.cfg.pinecone_api_key)
        self._client = pc

        index_name = self.cfg.pinecone_index_name or self.collection_name
        self.collection_name = index_name  # unify naming

        try:
            self._collection = pc.Index(index_name)
        except Exception:
            # Create lazily when we know dimension
            self._collection = None
    
    def _pinecone_metric(self) -> str:
        # Pinecone uses “dotproduct” instead of “dot”
        mapping = {"cosine": "cosine", "euclidean": "euclidean", "dot": "dotproduct"}
        return mapping.get(self.cfg.metric, "cosine")
    
    def _ensure_pinecone_index(self) -> None:
        assert self._client is not None, "Pinecone client is not initialized"
        assert self._dim is not None, "Embedding dim unknown; cannot create Pinecone index."

        pc = self._client
        index_name = self.collection_name
        try:
            self._collection = pc.Index(index_name)
        except Exception:
            pc.create_index(
                name=index_name,
                dimension=self._dim,
                metric=self._pinecone_metric(),
            )
            self._collection = pc.Index(index_name)

    def _upsert_pinecone(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[Dict[str, Any]],
    ) -> None:
        self._ensure_pinecone_index()
        vectors = [
            {"id": ids[i], "values": list(map(float, embeddings[i])), "metadata": metadatas[i]}
            for i in range(len(ids))
        ]
        try:
            self._collection.upsert(vectors=vectors)
        except TypeError:
            self._collection.upsert(items=vectors)
    

    def add_documents(
        self,
        documents: Sequence[Document],
        embeddings: Sequence[Sequence[float]],
        *,
        ids: Optional[Sequence[str]] = None,
    ) -> None:
        if len(documents) != len(embeddings):
            raise ValueError("documents and embeddings must have the same length.")

        if not documents:
            return

        if self._dim is None:
            self._dim = len(embeddings[0])

        ids_list = list(ids) if ids else [str(uuid.uuid4()) for _ in documents]

        metadatas: List[Dict[str, Any]] = []

        for doc in documents:
            metadata = dict(doc.metadata or {})
            existing = metadata.get("tenant_id")
            if existing is not None and existing != self.cfg.tenant_id:
                raise ValueError(
                    f"Upsert tenant_id mismatch: expected '{self.cfg.tenant_id}', got '{existing}'."
                )
            metadata["tenant_id"] = self.cfg.tenant_id
            metadata["text"] = doc.page_content
            metadatas.append(metadata)

        for i in range(0, len(ids_list), self.cfg.batch_size):
            batch_ids = ids_list[i : i + self.cfg.batch_size]
            batch_embeddings = embeddings[i : i + self.cfg.batch_size]
            batch_metadatas = metadatas[i : i + self.cfg.batch_size]

            self._upsert_pinecone(
                batch_ids,
                batch_embeddings,
                batch_metadatas,
            )

    def query(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int,
        metadata_filter: Optional[MetadataFilter] = None,
        include_embeddings: bool = False,
    ) -> List[VectorStoreHit]:
        vector = list(map(float, query_embedding))

        effective_where = dict(metadata_filter or {})
        effective_where.update({"tenant_id": self.cfg.tenant_id})

        self._ensure_pinecone_index()

        qr = self._collection.query(
            vector=vector,
            top_k=top_k,
            include_values=False,
            include_metadata=True,
            filter=effective_where,
        )

        matches = qr.get("matches", []) if isinstance(qr, dict) else qr.matches

        hits: List[VectorStoreHit] = []

        for m in matches:
            if isinstance(m, dict):
                _id = m.get("id")
                score = float(m.get("score", 0.0))
                md = m.get("metadata", {}) or {}
            else:
                _id = m.id
                score = float(m.score)
                md = m.metadata or {}

            metadata = dict(md)
            text = str(metadata.get("text", ""))

            hits.append(
                VectorStoreHit(
                    id=str(_id),
                    score=score,  # Pinecone: higher score = better
                    metadata=metadata,
                    document=text,
                )
            )

        return hits
    

    def delete(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        self._ensure_pinecone_index()
        self._collection.delete(ids=list(ids))


    def count(self) -> int:
        self._ensure_pinecone_index()
        try:
            stats = self._collection.describe_index_stats()
            return int(stats.get("total_vector_count", 0))
        except Exception:
            stats = self._client.describe_index_stats(index_name=self.collection_name)
            return int(stats.get("total_vector_count", 0))