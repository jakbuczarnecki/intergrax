# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Union
import uuid

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document
import numpy as np
from numpy.typing import NDArray

from intergrax.rag.contracts.vector_store import MetadataFilter, VectorStore, VectorStoreHit
from intergrax.rag.vectorstore_manager import VSConfig


class ChromaVectorStore(VectorStore):
    """
    Literal extraction of Chroma initialization logic from VectorstoreManager.
    No behavioral changes.
    """

    def __init__(self, cfg: VSConfig) -> None:
        self.cfg = cfg
        self.collection_name = f"{cfg.collection_name}__tenant__{cfg.tenant_id}"

        self._client = None
        self._collection = None
        self._dim: Optional[int] = None

        self._init_chroma()

    def _init_chroma(self) -> None:
        settings = self.cfg.chroma_settings or ChromaSettings()

        persist_dir = self.cfg.chroma_persist_directory
        if persist_dir:
            os.makedirs(persist_dir, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=persist_dir,
                settings=settings,
            )
        else:
            self._client = chromadb.Client(settings=settings)

        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Document embeddings for intergrax system"},
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


    def _to_list_of_lists(
        self, embeddings: Union[NDArray[np.float32], Sequence[Sequence[float]]]
    ) -> List[List[float]]:
        if isinstance(embeddings, np.ndarray):
            if embeddings.ndim != 2:
                raise ValueError("Embeddings numpy array must be 2D [N, D]")
            return embeddings.astype(np.float32).tolist()
        return [list(map(float, row)) for row in embeddings]


    def _doc_texts(self, docs: Sequence[Document]) -> List[str]:
        return [d.page_content or "" for d in docs]


    def _doc_payloads(
        self, docs: Sequence[Document], *, base: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for d in docs:
            meta = dict(base or {})
            if d.metadata:
                meta.update(d.metadata)
            out.append(meta)
        return out


    def _make_ids(self, n: int) -> List[str]:
        return [uuid.uuid4().hex for _ in range(n)]


    def _ensure_dim_consistency(self, batch: Sequence[Sequence[float]]) -> None:
        if self._dim is None:
            return
        for v in batch:
            if len(v) != self._dim:
                raise ValueError(
                    f"Embedding dimension mismatch. expected={self._dim}, got={len(v)}"
                )

    def add_documents(
        self,
        documents: Sequence[Document],
        embeddings: Sequence[Sequence[float]],
        *,
        ids: Optional[Sequence[str]] = None,
    ) -> None:
        if len(documents) == 0:
            return

        X = self._to_list_of_lists(
            np.asarray(list(embeddings), dtype=np.float32)
        )

        if len(X) != len(documents):
            raise ValueError("Number of documents must match number of embeddings")

        n = len(documents)
        ids_list = list(ids) if ids else self._make_ids(n)

        if len(ids_list) != n:
            raise ValueError("Length of `ids` must match number of documents")

        first_dim = len(X[0]) if X and X[0] else None
        if first_dim is None:
            raise ValueError(
                "Embeddings appear empty/corrupt; cannot infer dimension."
            )

        self._dim = self._dim or first_dim

        batch_size = 256

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)

            ids_batch = ids_list[start:end]
            embeddings_batch = X[start:end]
            self._ensure_dim_consistency(embeddings_batch)

            docs_batch = documents[start:end]
            metas_batch = self._doc_payloads(docs_batch, base=None)

            # tenant enforcement (legacy behavior)
            for i in range(len(metas_batch)):
                existing = metas_batch[i].get("tenant_id")
                if existing is not None and existing != self.cfg.tenant_id:
                    raise ValueError(
                        f"Metadata tenant_id mismatch: expected '{self.cfg.tenant_id}', got '{existing}'."
                    )
                metas_batch[i]["tenant_id"] = self.cfg.tenant_id

            self._upsert_chroma(
                ids_batch,
                embeddings_batch,
                metas_batch,
                self._doc_texts(docs_batch),
            )

    def query(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int,
        metadata_filter: Optional[MetadataFilter] = None,
        include_embeddings: bool = False,
    ) -> List[VectorStoreHit]:
        Q = [list(map(float, query_embedding))]

        include = ["metadatas", "documents", "distances"]
        if include_embeddings:
            include.append("embeddings")

        where = metadata_filter.conditions if metadata_filter else None

        tenant_filter = {"tenant_id": self.cfg.tenant_id}
        effective_where = dict(where or {})
        effective_where.update(tenant_filter)

        res = self._collection.query(
            query_embeddings=Q,
            n_results=top_k,
            where=self._normalize_chroma_where(effective_where),
            include=include,
        )

        distances = res.get("distances", [])
        if self.cfg.metric == "cosine":
            scores = [[1.0 - float(d) for d in row] for row in distances]
        else:
            scores = distances

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
                VectorStoreHit(
                    id=str(row_ids[rank]),
                    content=str(row_docs[rank]),
                    metadata=dict(row_metas[rank] or {}),
                    similarity_score=float(row_scores[rank]),
                    rank=rank,
                    embedding=(
                        list(row_embs[rank])
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
    

    def delete(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        self._collection.delete(ids=list(ids))

    def count(self) -> int:
        return int(self._collection.count())