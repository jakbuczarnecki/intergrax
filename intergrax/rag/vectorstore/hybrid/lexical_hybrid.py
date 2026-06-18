# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Dense + lexical hybrid query with RRF fusion."""

from __future__ import annotations
from intergrax.utils import attribute_access

from typing import Any, Dict, List, Optional, Sequence

from intergrax.rag.vectorstore.contracts.vector_store import MetadataFilter, VectorStoreHit
from intergrax.rag.vectorstore.sparse.lexical_index import LexicalIndex


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]],
    *,
    k: int = 60,
) -> List[tuple[str, float]]:
    scores: Dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class LexicalHybridSupport:
    """
    Mixin: maintain lexical index alongside dense vectors.

    Subclasses must implement dense ``query`` and populate payloads with ``text``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._lexical_index = LexicalIndex()

    def _index_lexical(self, doc_id: str, text: str) -> None:
        self._lexical_index.upsert(doc_id, text)

    def _lexical_hits(
        self,
        query_text: str,
        *,
        top_k: int,
        metadata_filter: Optional[MetadataFilter],
        payload_by_id: Dict[str, Dict[str, Any]],
    ) -> List[VectorStoreHit]:
        allowed = None
        if metadata_filter is not None:
            allowed = set()
            for doc_id, payload in payload_by_id.items():
                match = True
                for key, value in metadata_filter.conditions.items():
                    if payload.get(key) != value:
                        match = False
                        break
                if match:
                    allowed.add(doc_id)

        lexical = self._lexical_index.search(query_text, top_k=top_k, allowed_ids=allowed)
        hits: List[VectorStoreHit] = []
        for rank, (doc_id, score) in enumerate(lexical):
            payload = payload_by_id.get(doc_id, {})
            hits.append(
                VectorStoreHit(
                    id=doc_id,
                    content=str(payload.get("text", "")),
                    metadata=dict(payload),
                    similarity_score=float(score),
                    rank=rank,
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
        """RRF fusion of dense vector search and in-process BM25."""
        prefetch = max(top_k * 3, top_k)
        dense_hits = self.query(
            query_embedding,
            top_k=prefetch,
            metadata_filter=metadata_filter,
            include_embeddings=include_embeddings,
        )
        payload_by_id: Dict[str, Dict[str, Any]] = {
            h.id: dict(h.metadata or {}, text=h.content) for h in dense_hits
        }
        for doc_id in list(self._lexical_index._doc_terms.keys()):  # noqa: SLF001
            if doc_id not in payload_by_id and hasattr(self, "_payloads"):
                p = attribute_access.optional(self, "_payloads", {}).get(doc_id)
                if p:
                    payload_by_id[doc_id] = dict(p)

        lexical_hits = self._lexical_hits(
            query_text,
            top_k=prefetch,
            metadata_filter=metadata_filter,
            payload_by_id=payload_by_id,
        )

        dense_ranked = [h.id for h in dense_hits]
        lexical_ranked = [h.id for h in lexical_hits]
        fused = reciprocal_rank_fusion([dense_ranked, lexical_ranked])

        by_id: Dict[str, VectorStoreHit] = {h.id: h for h in dense_hits}
        for h in lexical_hits:
            by_id.setdefault(h.id, h)

        out: List[VectorStoreHit] = []
        for rank, (doc_id, rrf_score) in enumerate(fused[:top_k]):
            hit = by_id.get(doc_id)
            if hit is None:
                continue
            dense_score = hit.similarity_score
            combined = alpha * dense_score + (1.0 - alpha) * rrf_score
            out.append(
                VectorStoreHit(
                    id=hit.id,
                    content=hit.content,
                    metadata={**(hit.metadata or {}), "hybrid_rrf": rrf_score},
                    similarity_score=float(combined),
                    rank=rank,
                    embedding=hit.embedding,
                )
            )
        return out
