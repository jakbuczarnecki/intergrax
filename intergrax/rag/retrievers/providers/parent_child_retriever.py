# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Dict, List

from intergrax.rag.document_splitters.contracts.chunk_metadata_key import ChunkMetadataKey
from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager
from intergrax.rag.retrievers.contracts.base_retriever import (
    BaseRetriever,
    RetrieverCandidate,
    RetrieverQuery,
)


class ParentChildRetriever(BaseRetriever):

    def __init__(
        self,
        vector_store: BaseVectorstoreManager,
        embedding_manager: BaseEmbeddingManager,
        *,
        prefetch_factor: int = 10,
        max_per_parent: int = 2,
    ) -> None:

        self._vs = vector_store
        self._em = embedding_manager
        self._prefetch_factor = int(prefetch_factor)
        self._max_per_parent = int(max_per_parent)

    @classmethod
    def name(cls) -> str:
        return "parent_child"

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
            include_embeddings=query.include_embeddings,
        )

        parent_counts: Dict[str, int] = {}

        candidates: List[RetrieverCandidate] = []

        for hit in hits:

            parent_id = hit.metadata.get(
                ChunkMetadataKey.PARENT_CHUNK_ID
            )

            if parent_id is None:
                parent_id = hit.metadata.get(
                    ChunkMetadataKey.CHUNK_ID
                )

            if parent_id is None:
                parent_id = hit.id

            count = parent_counts.get(parent_id, 0)

            if count >= self._max_per_parent:
                continue

            parent_counts[parent_id] = count + 1

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

            if len(candidates) >= top_k:
                break

        return candidates