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
from intergrax.rag.vectorstore.providers.base_vector_store import BaseVectorStore


class InMemoryVectorStore(BaseVectorStore):
    """
    In-memory VectorStore provider.

    Designed for:
    - bootstrap defaults
    - unit tests
    - local development

    API mirrors QdrantVectorStore behavior.
    """

    def __init__(self, tenant_id: str) -> None:
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

    # ---------------------------------------------------------

    def count(self) -> int:
        return len(self._vectors)

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