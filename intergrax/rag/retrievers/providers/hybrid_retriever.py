# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List
import math
import re

from intergrax.rag.embedding.embedding_manager import EmbeddingManager
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager
from intergrax.rag.retrievers.contracts.base_retriever import (
    BaseRetriever,
    RetrieverCandidate,
    RetrieverQuery,
)


class HybridRetriever(BaseRetriever):

    def __init__(
        self,
        vector_store: VectorstoreManager,
        embedding_manager: EmbeddingManager,
        *,
        prefetch_factor: int = 10,
        alpha: float = 0.5,
    ) -> None:

        self._vs = vector_store
        self._em = embedding_manager
        self._prefetch_factor = int(prefetch_factor)
        self._alpha = float(alpha)

    @classmethod
    def name(cls) -> str:
        return "hybrid"
    

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())


    def _lexical_score(self, query_tokens: List[str], text: str) -> float:
        tokens = self._tokenize(text)

        if not tokens:
            return 0.0

        matches = sum(1 for t in query_tokens if t in tokens)

        return matches / len(query_tokens)

    def retrieve(
        self,
        query: RetrieverQuery,
    ) -> List[RetrieverCandidate]:

        if not query.query_text:
            return []

        q_vec = (
            query.query_embedding
            if query.query_embedding is not None
            else self._em.embed_one(query.query_text)
        )

        query_tokens = self._tokenize(query.query_text)

        top_k = int(query.top_k)
        prefetch_k = max(top_k, top_k * self._prefetch_factor)

        hits = self._vs.query(
            query_embedding=q_vec,
            top_k=prefetch_k,
            metadata_filter=query.metadata_filter,
            include_embeddings=query.include_embeddings,
        )

        candidates: List[RetrieverCandidate] = []

        for hit in hits:

            lexical = self._lexical_score(query_tokens, hit.content)

            hybrid_score = (
                self._alpha * hit.similarity_score
                + (1 - self._alpha) * lexical
            )

            candidates.append(
                RetrieverCandidate(
                    id=hit.id,
                    content=hit.content,
                    metadata=hit.metadata,
                    score=hybrid_score,
                    embedding=hit.embedding,
                    rank=hit.rank,
                )
            )

        candidates.sort(
            key=lambda x: x.score,
            reverse=True,
        )

        return candidates[:top_k]