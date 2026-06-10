# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Dict, List, Optional

from intergrax.rag.document_splitters.contracts.chunk_metadata_key import ChunkMetadataKey
from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager
from intergrax.rag.vectorstore.contracts.vector_store import MetadataFilter
from intergrax.rag.retrievers.contracts.base_retriever import (
    BaseRetriever,
    RetrieverCandidate,
    RetrieverQuery,
)


class HierarchicalRetriever(BaseRetriever):

    def __init__(
        self,
        chunks_store: BaseVectorstoreManager,
        embedding_manager: BaseEmbeddingManager,
        *,
        toc_store: Optional[BaseVectorstoreManager] = None,
        k_chunks: int = 30,
        k_toc: int = 8,
        max_toc_parents: int = 5,
    ) -> None:

        self._chunks = chunks_store
        self._toc = toc_store
        self._em = embedding_manager

        self._k_chunks = int(k_chunks)
        self._k_toc = int(k_toc)
        self._max_toc_parents = int(max_toc_parents)

    @classmethod
    def name(cls) -> str:
        return "hierarchical"

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

        # Step 1: base retrieval from chunks
        base_hits = self._chunks.query(
            query_embedding=q_vec,
            top_k=max(top_k, self._k_chunks),
            metadata_filter=query.metadata_filter,
            include_embeddings=query.include_embeddings,
        )

        results: Dict[str, RetrieverCandidate] = {}

        for hit in base_hits:

            results[hit.id] = RetrieverCandidate(
                id=hit.id,
                content=hit.content,
                metadata=hit.metadata,
                score=hit.similarity_score,
                embedding=hit.embedding,
                rank=hit.rank,
            )

        # Step 2: optional TOC expansion
        if self._toc is not None:

            toc_hits = self._toc.query(
                query_embedding=q_vec,
                top_k=self._k_toc,
                metadata_filter=query.metadata_filter,
                include_embeddings=False,
            )

            parents: List[str] = []
            seen = set()

            for hit in toc_hits:

                parent = hit.metadata.get(ChunkMetadataKey.PARENT_CHUNK_ID)

                if not parent:
                    continue

                parent = str(parent)

                if parent in seen:
                    continue

                seen.add(parent)
                parents.append(parent)

                if len(parents) >= self._max_toc_parents:
                    break

            for parent in parents:

                parent_filter: dict[str, object] = {
                    ChunkMetadataKey.PARENT_CHUNK_ID: parent,
                }
                if query.metadata_filter is not None:
                    parent_filter.update(query.metadata_filter.conditions)
                local_hits = self._chunks.query(
                    query_embedding=q_vec,
                    top_k=self._k_chunks,
                    metadata_filter=MetadataFilter(conditions=parent_filter),
                    include_embeddings=query.include_embeddings,
                )

                for hit in local_hits:

                    if hit.id not in results:

                        results[hit.id] = RetrieverCandidate(
                            id=hit.id,
                            content=hit.content,
                            metadata=hit.metadata,
                            score=hit.similarity_score,
                            embedding=hit.embedding,
                            rank=hit.rank,
                        )

        candidates = list(results.values())

        candidates.sort(
            key=lambda x: x.score,
            reverse=True,
        )

        return candidates[:top_k]