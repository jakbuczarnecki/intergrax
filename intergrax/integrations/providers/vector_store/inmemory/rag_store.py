# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
import math

from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    MetadataFilter,
    VectorStoreHit,
    VectorStoreRecord,
    VectorStoreScope,
)
from intergrax.rag.vectorstore.hybrid.lexical_hybrid import LexicalHybridSupport
from intergrax.rag.vectorstore.providers.base_vector_store import BaseVectorStore
from intergrax.rag.vectorstore.providers.native_provider_boundary import (
    native_hit,
    provider_metadata,
    reconstruct_document,
    validate_query,
    validate_records,
    validate_scope,
)


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
        self._documents: Dict[str, Any] = {}

    # ---------------------------------------------------------

    def add_records(
        self,
        records: Sequence[VectorStoreRecord],
        *,
        scope: VectorStoreScope,
    ) -> Sequence[str]:

        validated = validate_records(records, scope=scope, tenant_id=self._tenant_id)
        ids: list[str] = []

        for record in validated:
            payload = provider_metadata(record.document, scope=scope)
            vector_id = record.vector_id
            self._vectors[vector_id] = record.embedding.tolist()
            self._payloads[vector_id] = dict(payload)
            self._documents[vector_id] = reconstruct_document(
                record.document.content,
                payload,
                scope=scope,
            )
            self._index_lexical(vector_id, record.document.content)
            ids.append(vector_id)
        return ids

    # ---------------------------------------------------------

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
        validate_scope(scope, tenant_id=self._tenant_id)
        effective_where = dict(
            MetadataFilter.for_scope(scope, metadata_filter).conditions
        )

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
        candidates = candidates[:limit]

        hits: List[VectorStoreHit] = []

        for rank, (id_, score) in enumerate(candidates):

            payload = self._payloads[id_]
            hits.append(
                native_hit(
                    vector_id=id_,
                    content=self._documents[id_].content,
                    metadata=payload,
                    similarity_score=score,
                    rank=rank,
                    scope=scope,
                    embedding=self._vectors[id_] if include_embeddings else None,
                )
            )

        return hits

    # ---------------------------------------------------------

    def delete(self, ids: Sequence[str], *, scope: VectorStoreScope) -> None:

        validate_scope(scope, tenant_id=self._tenant_id)
        for id_ in ids:
            payload = self._payloads.get(id_)
            if payload is None or not self._matches_scope(payload, scope):
                continue
            self._vectors.pop(id_, None)
            self._payloads.pop(id_, None)
            self._documents.pop(id_, None)
            self._lexical_index.remove(id_)

    # ---------------------------------------------------------

    def count(self, *, scope: VectorStoreScope) -> int:
        validate_scope(scope, tenant_id=self._tenant_id)
        return sum(
            self._matches_scope(payload, scope)
            for payload in self._payloads.values()
        )

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
        document = self._documents.get(document_id)
        if document is None:
            return None
        return {
            "id": document_id,
            "text": document.content,
            "metadata": dict(document.metadata),
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
            document = self._documents[doc_id]
            results.append(
                {
                    "id": doc_id,
                    "text": document.content,
                    "metadata": dict(document.metadata),
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
        self.delete(
            list(self._payloads.keys()),
            scope=VectorStoreScope(tenant_id=self._tenant_id),
        )
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

    @staticmethod
    def _matches_scope(
        payload: Dict[str, Any],
        scope: VectorStoreScope,
    ) -> bool:
        return (
            payload.get("tenant_id") == scope.tenant_id
            and payload.get("namespace") == scope.namespace
            and payload.get("workspace_id") == scope.workspace_id
        )