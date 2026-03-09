# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List
import numpy as np

from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager
from intergrax.rag.retrievers.contracts.base_retriever import (
    BaseRetriever,
    RetrieverCandidate,
    RetrieverQuery,
)


class MMRRetriever(BaseRetriever):

    def __init__(
        self,
        vector_store: BaseVectorstoreManager,
        embedding_manager: BaseEmbeddingManager,
        *,
        prefetch_factor: int = 10,
        lambda_mult: float = 0.5,
    ) -> None:

        self._vs = vector_store
        self._em = embedding_manager
        self._prefetch_factor = int(prefetch_factor)
        self._lambda = float(lambda_mult)

    @classmethod
    def name(cls) -> str:
        return "mmr"

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(
            np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
        )

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

        hits = self._vs.query(
            query_embedding=q_vec,
            top_k=prefetch_k,
            metadata_filter=query.metadata_filter,
            include_embeddings=True,
        )

        if not hits:
            return []

        query_vec = np.asarray(q_vec, dtype="float32").reshape(-1)

        valid_hits = [h for h in hits if h.embedding is not None]

        if not valid_hits:
            return []

        embeddings = [
            np.asarray(h.embedding, dtype="float32").reshape(-1)
            for h in valid_hits
        ]

        selected: List[int] = []
        remaining = list(range(len(valid_hits)))

        while remaining and len(selected) < top_k:

            if not selected:

                scores = [
                    self._cosine(query_vec, embeddings[i])
                    for i in remaining
                ]

                best = remaining[int(np.argmax(scores))]

                selected.append(best)
                remaining.remove(best)

                continue

            mmr_scores = []

            for i in remaining:

                sim_query = self._cosine(query_vec, embeddings[i])

                sim_selected = max(
                    self._cosine(embeddings[i], embeddings[j])
                    for j in selected
                )

                score = (
                    self._lambda * sim_query
                    - (1 - self._lambda) * sim_selected
                )

                mmr_scores.append(score)

            best = remaining[int(np.argmax(mmr_scores))]

            selected.append(best)
            remaining.remove(best)

        candidates: List[RetrieverCandidate] = []

        for rank, idx in enumerate(selected):

            hit = valid_hits[idx]

            candidates.append(
                RetrieverCandidate(
                    id=hit.id,
                    content=hit.content,
                    metadata=hit.metadata,
                    score=hit.similarity_score,
                    embedding=hit.embedding,
                    rank=rank,
                )
            )

        return candidates