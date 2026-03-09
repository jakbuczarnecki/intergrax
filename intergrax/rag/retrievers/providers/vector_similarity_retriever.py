# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List


from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager
from intergrax.rag.retrievers.contracts.base_retriever import (
    BaseRetriever,
    RetrieverCandidate,
    RetrieverQuery,
)


class VectorSimilarityRetriever(BaseRetriever):

    def __init__(
        self,
        vector_store: BaseVectorstoreManager,
        embedding_manager: BaseEmbeddingManager,
        *,
        prefetch_factor: int = 10,
    ) -> None:

        self._vs = vector_store
        self._em = embedding_manager
        self._prefetch_factor = int(prefetch_factor)

    @classmethod
    def name(cls) -> str:
        return "vector_similarity"

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

        top_k = int(query.top_k)
        prefetch_k = max(top_k, top_k * self._prefetch_factor)

        results = self._vs.query(
            query_embedding=q_vec,
            top_k=prefetch_k,
            metadata_filter=query.metadata_filter,
            include_embeddings=query.include_embeddings,
        )

        candidates: List[RetrieverCandidate] = []

        for hit in results:

            candidates.append(
                RetrieverCandidate(
                    id=hit.id,
                    content=hit.content,
                    metadata=hit.metadata,
                    score=hit.similarity_score,
                    embedding=hit.embedding,
                    rank=hit.rank,
                )
            )

        return candidates[:top_k]