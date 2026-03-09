# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Dict, List

from intergrax.rag.embedding.embedding_manager import EmbeddingManager
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager
from intergrax.rag.retrievers.contracts.base_retriever import (
    BaseRetriever,
    RetrieverCandidate,
    RetrieverQuery,
)


class MultiQueryRetriever(BaseRetriever):

    def __init__(
        self,
        vector_store: VectorstoreManager,
        embedding_manager: EmbeddingManager,
        *,
        prefetch_factor: int = 5,
        num_queries: int = 3,
    ) -> None:

        self._vs = vector_store
        self._em = embedding_manager
        self._prefetch_factor = int(prefetch_factor)
        self._num_queries = int(num_queries)

    @classmethod
    def name(cls) -> str:
        return "multiquery"

    def _generate_queries(self, query: str) -> List[str]:
        """
        Simple deterministic query expansion.
        Placeholder for future LLM-based expansion.
        """

        variants = {query}

        words = query.split()

        if len(words) > 2:
            variants.add(" ".join(words[:2]))
            variants.add(" ".join(words[-2:]))

        return list(variants)[: self._num_queries]

    def retrieve(
        self,
        query: RetrieverQuery,
    ) -> List[RetrieverCandidate]:

        if not query.query_text:
            return []

        expanded_queries = self._generate_queries(query.query_text)

        top_k = int(query.top_k)
        prefetch_k = max(top_k, top_k * self._prefetch_factor)

        all_hits: Dict[str, RetrieverCandidate] = {}

        for q in expanded_queries:

            if q == query.query_text and query.query_embedding:
                q_vec = query.query_embedding
            else:
                q_vec = self._em.embed_one(q)

            hits = self._vs.query(
                query_embedding=q_vec,
                top_k=prefetch_k,
                metadata_filter=query.metadata_filter,
                include_embeddings=query.include_embeddings,
            )

            for hit in hits:

                existing = all_hits.get(hit.id)

                if existing is None or hit.similarity_score > existing.score:

                    all_hits[hit.id] = RetrieverCandidate(
                        id=hit.id,
                        content=hit.content,
                        metadata=hit.metadata,
                        score=hit.similarity_score,
                        embedding=hit.embedding,
                        rank=hit.rank,
                    )

        candidates = list(all_hits.values())

        candidates.sort(
            key=lambda x: x.score,
            reverse=True,
        )

        return candidates[:top_k]