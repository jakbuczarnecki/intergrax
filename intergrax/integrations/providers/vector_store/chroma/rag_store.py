# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Dict, List, Literal, Optional, Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings

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
    validate_query,
    validate_records,
    validate_scope,
)


@dataclass(frozen=True)
class ChromaConfig:
    """
    Configuration model for Chroma vector store provider.

    The provider is responsible for creating and managing
    the Chroma client instance based on this configuration.
    """

    collection_name: str
    tenant_id: str
    persist_directory: Optional[str] = None
    settings: Optional[ChromaSettings] = None
    batch_size: int = 256
    metric: Literal["cosine", "l2"] = "cosine"

    mode: Literal["embedded", "http"] = "embedded"
    http_host: str = "localhost"
    http_port: int = 8000
    

class ChromaVectorStore(BaseVectorStore):
    """
    Literal extraction of Chroma initialization logic from VectorstoreManager.
    No behavioral changes.
    """

    def __init__(self, cfg: ChromaConfig, *, client: Any = None) -> None:
        self.cfg = cfg
        self.collection_name = f"{cfg.collection_name}__tenant__{cfg.tenant_id}"

        self._client = client
        self._collection = None
        self._dim: Optional[int] = None

        self._init_chroma()

    def _init_chroma(self) -> None:
        if self._client is not None:
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
            )
            return
        settings = self.cfg.settings or ChromaSettings()

        if self.cfg.mode == "http":
            self._client = chromadb.HttpClient(
                host=self.cfg.http_host,
                port=self.cfg.http_port,
            )
        else:
            persist_dir = self.cfg.persist_directory
            if persist_dir:
                os.makedirs(persist_dir, exist_ok=True)
                self._client = chromadb.PersistentClient(
                    path=persist_dir,
                    settings=settings,
                )
            else:
                self._client = chromadb.Client(settings=settings)

        space = "cosine" if self.cfg.metric == "cosine" else "l2"

        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": "Document embeddings for intergrax system",
                "hnsw:space": space,
            },
        )

    def _upsert_chroma(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[Dict[str, Any]],
        documents: Sequence[str],
    ) -> None:
        try:
            self._collection.upsert(
                ids=list(ids),
                embeddings=list(embeddings),
                metadatas=list(metadatas),
                documents=list(documents),
            )
        except AttributeError:
            self._collection.add(
                ids=list(ids),
                embeddings=list(embeddings),
                metadatas=list(metadatas),
                documents=list(documents),
            )


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
                provider_metadata(record.document, scope=scope)
                for record in records_batch
            ]

            self._upsert_chroma(
                ids_batch,
                embeddings_batch,
                metas_batch,
                [record.document.content for record in records_batch],
            )
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
        vector, limit = validate_query(query_embedding, top_k=top_k)
        validate_scope(scope, tenant_id=self.cfg.tenant_id)
        Q = [vector.tolist()]

        include = ["metadatas", "documents", "distances"]
        if include_embeddings:
            include.append("embeddings")

        effective_where = dict(
            MetadataFilter.for_scope(scope, metadata_filter).conditions
        )

        res = self._collection.query(
            query_embeddings=Q,
            n_results=limit,
            where=self._normalize_chroma_where(effective_where),
            include=include,
        )

        distances = res.get("distances", [])

        if self.cfg.metric == "cosine":
            scores = [[1.0 - float(d) for d in row] for row in distances]
        else:
            # L2 normalization
            scores = [[1.0 / (1.0 + float(d)) for d in row] for row in distances]        

        ids_out = res.get("ids", [[]])
        metadatas_out = res.get("metadatas", [[]])
        documents_out = res.get("documents", [[]])
        embeddings_out = (
            res.get("embeddings", [[]]) if include_embeddings else [[]]
        )

        hits: List[VectorStoreHit] = []

        row_ids = ids_out[0] if ids_out else []
        row_scores = scores[0] if scores else []
        row_metas = metadatas_out[0] if metadatas_out else []
        row_docs = documents_out[0] if documents_out else []
        row_embs = (
            embeddings_out[0] if include_embeddings and embeddings_out else []
        )

        for rank in range(
            min(len(row_ids), len(row_scores), len(row_metas), len(row_docs))
        ):
            hits.append(
                native_hit(
                    vector_id=str(row_ids[rank]),
                    content=str(row_docs[rank]),
                    metadata=dict(row_metas[rank] or {}),
                    similarity_score=float(row_scores[rank]),
                    rank=rank,
                    scope=scope,
                    embedding=(
                        row_embs[rank]
                        if include_embeddings and row_embs
                        else None
                    ),
                )
            )

        return hits
    

    def _normalize_chroma_where(self, where: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Chroma expects `where` to have exactly one top-level operator when multiple
        conditions are present. We support a friendly dict form and convert it.

        Examples:
        {"user_id": "u1"} ->
            {"user_id": {"$eq": "u1"}}

        {"user_id": "u1", "deleted": False} ->
            {"$and": [{"user_id": {"$eq": "u1"}}, {"deleted": {"$eq": False}}]}

        If `where` already contains an operator key (starts with '$'), it is returned as-is.
        """
        if not where:
            return None

        # Already operator-based (user may pass {"$and": [...]} etc.)
        if any(isinstance(k, str) and k.startswith("$") for k in where.keys()):
            return where

        items = list(where.items())

        # Single condition
        if len(items) == 1:
            k, v = items[0]
            return {str(k): {"$eq": v}}

        # Multiple conditions -> $and
        and_terms = [{str(k): {"$eq": v}} for k, v in items]
        return {"$and": and_terms}
    

    def delete(self, ids: Sequence[str], *, scope: VectorStoreScope) -> None:
        if not ids:
            return
        validate_scope(scope, tenant_id=self.cfg.tenant_id)
        self._collection.delete(
            ids=list(ids),
            where=self._normalize_chroma_where(
                MetadataFilter.for_scope(scope, None).conditions
            ),
        )

    def count(self, *, scope: VectorStoreScope) -> int:
        validate_scope(scope, tenant_id=self.cfg.tenant_id)
        result = self._collection.get(
            where=self._normalize_chroma_where(
                MetadataFilter.for_scope(scope, None).conditions
            ),
            include=[],
        )
        return len(result.get("ids", []))

    def list_collections(self) -> List[str]:
        if self._client is None:
            return [self.collection_name]
        return [collection.name for collection in self._client.list_collections()]