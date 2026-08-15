# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import replace

from intergrax.rag.document_splitters.contracts.chunk_metadata_key import ChunkMetadataKey
from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager
from intergrax.rag.retrievers.contracts.base_retriever import (
    BaseRetriever,
    RetrievalHit,
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
            **({"scope": query.scope} if query.scope is not None else {}),
            top_k=prefetch_k,
            metadata_filter=query.metadata_filter,
            include_embeddings=query.include_embeddings,
        )

        native_hits = tuple(
            RetrievalHit.from_vector_store_hit(
                hit,
                channel="dense",
                retriever_name=self.name(),
            )
            for hit in hits
        )
        parent_counts: dict[tuple[str, str | None, str | None, str], int] = {}

        candidates: list[RetrievalHit] = []

        for hit in native_hits:

            parent_id = hit.document.metadata.get(
                ChunkMetadataKey.PARENT_CHUNK_ID
            )

            if parent_id is None:
                parent_id = hit.document.metadata.get(
                    ChunkMetadataKey.CHUNK_ID
                )

            if parent_id is None:
                parent_id = hit.vector_id

            parent_id = str(parent_id)
            parent_key = (
                hit.document.scope.tenant_id,
                hit.document.scope.namespace,
                hit.document.scope.workspace_id,
                parent_id,
            )
            count = parent_counts.get(parent_key, 0)

            if count >= self._max_per_parent:
                continue

            parent_counts[parent_key] = count + 1

            candidates.append(
                replace(
                    hit,
                    parent_vector_id=parent_id,
                    child_vector_id=hit.vector_id,
                    rank=len(candidates),
                    retriever_name=self.name(),
                )
            )

            if len(candidates) >= top_k:
                break

        return tuple(candidates)