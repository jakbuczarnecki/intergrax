# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import numpy as np
from typing import List

from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager
from intergrax.rag.retrievers.contracts.base_retriever import (
    BaseRetriever,
    RetrievalHit,
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
    ) -> tuple[RetrievalHit, ...]:

        if not query.query_text:
            return ()

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
            return ()

        query_vec = np.asarray(q_vec, dtype="float32").reshape(-1)

        native_hits = tuple(
            RetrievalHit.from_vector_store_hit(
                hit,
                channel="dense",
                retriever_name=self.name(),
            )
            for hit in hits
        )
        valid_hits = [h for h in native_hits if h.embedding is not None]

        if not valid_hits:
            return ()

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

        candidates: list[RetrievalHit] = []

        for rank, idx in enumerate(selected):

            hit = valid_hits[idx]

            candidates.append(
                RetrievalHit(
                    document=hit.document,
                    score=hit.score,
                    rank=rank,
                    channel=hit.channel,
                    vector_id=hit.vector_id,
                    embedding=hit.embedding if query.include_embeddings else None,
                    source_rank=hit.source_rank,
                    retriever_name=self.name(),
                )
            )

        return tuple(candidates)