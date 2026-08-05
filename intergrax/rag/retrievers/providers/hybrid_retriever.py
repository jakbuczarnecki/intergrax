# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import re

from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager
from intergrax.rag.retrievers.contracts.base_retriever import (
    BaseRetriever,
    RetrievalHit,
    RetrieverQuery,
)


class HybridRetriever(BaseRetriever):

    def __init__(
        self,
        vector_store: BaseVectorstoreManager,
        embedding_manager: BaseEmbeddingManager,
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
    

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())


    def _lexical_score(self, query_tokens: list[str], text: str) -> float:
        tokens = self._tokenize(text)

        if not tokens:
            return 0.0

        matches = sum(1 for t in query_tokens if t in tokens)

        return matches / len(query_tokens)

    def retrieve(
        self,
        query: RetrieverQuery,
    ) -> tuple[RetrievalHit, ...]:

        if not query.query_text:
            return ()

        q_vec = (
            query.query_embedding
            if query.query_embedding is not None
            else self._em.embed_one(query.query_text)
        )

        query_tokens = self._tokenize(query.query_text)

        top_k = int(query.top_k)
        prefetch_k = max(top_k, top_k * self._prefetch_factor)

        if hasattr(self._vs, "query_hybrid") and query.query_text:
            hits = self._vs.query_hybrid(
                q_vec,
                query.query_text,
                top_k=prefetch_k,
                metadata_filter=query.metadata_filter,
                include_embeddings=query.include_embeddings,
                alpha=self._alpha,
            )
            native_hits = tuple(
                RetrievalHit.from_vector_store_hit(
                    hit,
                    channel="hybrid",
                    retriever_name=self.name(),
                )
                for hit in hits
            )
            return tuple(
                RetrievalHit(
                    document=hit.document,
                    score=hit.score,
                    rank=rank,
                    channel=hit.channel,
                    vector_id=hit.vector_id,
                    embedding=hit.embedding,
                    source_rank=hit.source_rank,
                    retriever_name=self.name(),
                )
                for rank, hit in enumerate(native_hits[:top_k])
            )

        hits = self._vs.query(
            query_embedding=q_vec,
            top_k=prefetch_k,
            metadata_filter=query.metadata_filter,
            include_embeddings=query.include_embeddings,
        )

        candidates: list[RetrievalHit] = []
        for hit in hits:
            native_hit = RetrievalHit.from_vector_store_hit(
                hit,
                channel="hybrid",
                retriever_name=self.name(),
            )
            lexical = self._lexical_score(query_tokens, native_hit.content)
            hybrid_score = self._alpha * hit.similarity_score + (1 - self._alpha) * lexical
            candidates.append(
                RetrievalHit(
                    document=native_hit.document,
                    score=hybrid_score,
                    rank=0,
                    channel="hybrid",
                    vector_id=native_hit.vector_id,
                    embedding=native_hit.embedding,
                    source_rank=native_hit.source_rank,
                    retriever_name=self.name(),
                )
            )

        candidates.sort(key=lambda x: x.score, reverse=True)
        return tuple(
            RetrievalHit(
                document=hit.document,
                score=hit.score,
                rank=rank,
                channel=hit.channel,
                vector_id=hit.vector_id,
                embedding=hit.embedding,
                source_rank=hit.source_rank,
                retriever_name=self.name(),
            )
            for rank, hit in enumerate(candidates[:top_k])
        )