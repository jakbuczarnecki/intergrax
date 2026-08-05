# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import replace
from typing import Optional

from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.query.query_expander import DeterministicQueryExpander, QueryExpander
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager
from intergrax.rag.retrievers.contracts.base_retriever import (
    BaseRetriever,
    RetrievalHit,
    RetrieverQuery,
)


class MultiQueryRetriever(BaseRetriever):

    def __init__(
        self,
        vector_store: BaseVectorstoreManager,
        embedding_manager: BaseEmbeddingManager,
        *,
        prefetch_factor: int = 5,
        num_queries: int = 3,
        query_expander: Optional[QueryExpander] = None,
    ) -> None:

        self._vs = vector_store
        self._em = embedding_manager
        self._prefetch_factor = int(prefetch_factor)
        self._num_queries = int(num_queries)
        self._expander = query_expander or DeterministicQueryExpander()

    @classmethod
    def name(cls) -> str:
        return "multiquery"

    def _generate_queries(self, query: str) -> list[str]:
        return self._expander.expand(query, num_queries=self._num_queries)

    def retrieve(
        self,
        query: RetrieverQuery,
    ) -> tuple[RetrievalHit, ...]:

        if not query.query_text:
            return ()

        expanded_queries = self._generate_queries(query.query_text)

        top_k = int(query.top_k)
        prefetch_k = max(top_k, top_k * self._prefetch_factor)

        all_hits: dict[tuple[str, str | None, str | None], RetrievalHit] = {}

        for query_index, q in enumerate(expanded_queries):

            if q == query.query_text and query.query_embedding is not None:
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
                native_hit = RetrievalHit.from_vector_store_hit(
                    hit,
                    channel="dense",
                    query_id=str(query_index),
                    query_text=q,
                    retriever_name=self.name(),
                )
                key = (
                    native_hit.document.scope.tenant_id,
                    native_hit.document.scope.namespace,
                    native_hit.vector_id,
                )
                existing = all_hits.get(key)
                if existing is None or native_hit.score > existing.score:
                    all_hits[key] = native_hit

        candidates = sorted(all_hits.values(), key=lambda x: x.score, reverse=True)
        return tuple(replace(hit, rank=rank) for rank, hit in enumerate(candidates[:top_k]))