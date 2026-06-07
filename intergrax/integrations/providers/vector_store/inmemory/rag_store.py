# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
import math
import uuid

from langchain_core.documents import Document

from intergrax.rag.embedding.contracts.embedding_metadata_key import EmbeddingMetadataKey
from intergrax.rag.vectorstore.contracts.vector_store import MetadataFilter, VectorStoreHit
from intergrax.rag.vectorstore.hybrid.lexical_hybrid import LexicalHybridSupport
from intergrax.rag.vectorstore.providers.base_vector_store import BaseVectorStore


class InMemoryVectorStore(LexicalHybridSupport, BaseVectorStore):
    """
    In-memory VectorStore provider.

    Designed for:
    - bootstrap defaults
    - unit tests
    - local development

    API mirrors QdrantVectorStore behavior.
    """

    def __init__(self, tenant_id: str) -> None:
        LexicalHybridSupport.__init__(self)
        self._tenant_id = tenant_id
        self._vectors: Dict[str, List[float]] = {}
        self._payloads: Dict[str, Dict[str, Any]] = {}

    # ---------------------------------------------------------

    def add_documents(
        self,
        documents: Sequence[Document],
        embeddings: Sequence[Sequence[float]],
        *,
        ids: Optional[Sequence[str]] = None,
    ) -> None:

        if len(documents) == 0:
            return

        if len(documents) != len(embeddings):
            raise ValueError("documents and embeddings length mismatch")

        n = len(documents)
        ids_list = list(ids) if ids else [str(uuid.uuid4()) for _ in range(n)]

        for i in range(n):

            vector = list(map(float, embeddings[i]))
            doc = documents[i]

            payload = dict(doc.metadata or {})

            if EmbeddingMetadataKey.VECTOR in payload:
                payload.pop(EmbeddingMetadataKey.VECTOR, None)

            existing = payload.get("tenant_id")
            if existing is not None and existing != self._tenant_id:
                raise ValueError(
                    f"Metadata tenant_id mismatch: expected '{self._tenant_id}', got '{existing}'."
                )

            payload["tenant_id"] = self._tenant_id
            payload["text"] = doc.page_content or ""

            self._vectors[ids_list[i]] = vector
            self._payloads[ids_list[i]] = payload
            self._index_lexical(ids_list[i], payload.get("text", ""))

    # ---------------------------------------------------------

    def query(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int,
        metadata_filter: Optional[MetadataFilter] = None,
        include_embeddings: bool = False,
    ) -> List[VectorStoreHit]:

        vector = list(map(float, query_embedding))

        effective_where: Dict[str, Any] = (
            dict(metadata_filter.conditions)
            if metadata_filter is not None
            else {}
        )

        existing = effective_where.get("tenant_id")
        if existing is not None and existing != self._tenant_id:
            raise ValueError(
                f"Query tenant_id mismatch: expected '{self._tenant_id}', got '{existing}'."
            )

        effective_where["tenant_id"] = self._tenant_id

        candidates: List[tuple[str, float]] = []

        for id_, emb in self._vectors.items():

            payload = self._payloads[id_]

            # metadata filtering
            match = True
            for k, v in effective_where.items():
                if payload.get(k) != v:
                    match = False
                    break

            if not match:
                continue

            score = self._cosine_similarity(vector, emb)
            candidates.append((id_, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = candidates[:top_k]

        hits: List[VectorStoreHit] = []

        for rank, (id_, score) in enumerate(candidates):

            payload = self._payloads[id_]
            text = payload.get("text", "")

            hits.append(
                VectorStoreHit(
                    id=id_,
                    content=text,
                    metadata=payload,
                    similarity_score=float(score),
                    rank=rank,
                    embedding=self._vectors[id_] if include_embeddings else None,
                )
            )

        return hits

    # ---------------------------------------------------------

    def delete(self, ids: Sequence[str]) -> None:

        for id_ in ids:
            self._vectors.pop(id_, None)
            self._payloads.pop(id_, None)
            self._lexical_index.remove(id_)

    # ---------------------------------------------------------

    def count(self) -> int:
        return len(self._vectors)

    def list_collections(self) -> List[str]:
        return [f"inmemory:{self._tenant_id}"]

    def list_document_ids(self, *, limit: int = 100, offset: int = 0) -> List[str]:
        ids = list(self._payloads.keys())
        if offset < 0:
            offset = 0
        return ids[offset : offset + max(1, limit)]

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        payload = self._payloads.get(document_id)
        if payload is None:
            return None
        metadata = {key: value for key, value in payload.items() if key != "text"}
        return {
            "id": document_id,
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
        if existing is not None and existing != self._tenant_id:
            raise ValueError(
                f"Query tenant_id mismatch: expected '{self._tenant_id}', got '{existing}'."
            )
        effective_where["tenant_id"] = self._tenant_id

        results: List[Dict[str, Any]] = []
        for doc_id, payload in self._payloads.items():
            match = all(payload.get(key) == value for key, value in effective_where.items())
            if not match:
                continue
            metadata = {key: value for key, value in payload.items() if key != "text"}
            results.append(
                {
                    "id": doc_id,
                    "text": str(payload.get("text") or ""),
                    "metadata": metadata,
                }
            )
            if len(results) >= max(1, limit):
                break
        return results

    def purge_collection(self, *, dry_run: bool = True, tenant_id: str = "") -> Dict[str, Any]:
        if tenant_id and tenant_id != self._tenant_id:
            raise ValueError(
                f"Purge tenant_id mismatch: expected '{self._tenant_id}', got '{tenant_id}'."
            )
        document_count = len(self._payloads)
        if dry_run:
            return {"dry_run": True, "would_delete": document_count, "tenant_id": self._tenant_id}
        self.delete(list(self._payloads.keys()))
        return {"dry_run": False, "deleted": document_count, "tenant_id": self._tenant_id}

    # ---------------------------------------------------------

    def _cosine_similarity(
        self,
        a: Sequence[float],
        b: Sequence[float],
    ) -> float:

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)