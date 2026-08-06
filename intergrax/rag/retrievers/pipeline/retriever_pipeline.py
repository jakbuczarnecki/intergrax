# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List, Sequence

from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.retrievers.contracts.base_retriever import (
    RetrievalHit,
    RetrieverQuery,
)
from intergrax.rag.retrievers.engine.retriever_engine import RetrieverEngine
from intergrax.rag.retrievers.engine.retriever_execution import RetrieverExecutionMetadata
from intergrax.rag.vectorstore.contracts.vector_store import MetadataFilter
from intergrax.rag.vectorstore.contracts.native_vectorstore import VectorStoreScope


class RetrieverPipeline:
    """
    Pipeline responsible for document retrieval.

    Responsibilities
    ----------------
    - accept query input
    - construct RetrieverQuery
    - delegate retrieval execution to RetrieverEngine
    """

    def __init__(
        self,
        engine: RetrieverEngine,
        *,
        embedding_manager: BaseEmbeddingManager
    ) -> None:
        self._engine = engine
        self._embedding_manager = embedding_manager

    @property
    def last_execution(self) -> RetrieverExecutionMetadata | None:
        return self._engine.last_execution

    def retrieve(
        self,
        query_text: str,
        *,
        retriever_id: str,
        query_embedding: Sequence[float] | None = None,
        top_k: int = 5,
        metadata_filter: MetadataFilter | None=None,
        scope: VectorStoreScope | None = None,
        include_embeddings: bool = False,
    ) -> List[RetrievalHit]:
        """
        Execute retrieval for a query text.
        """

        retriever = self._engine.get_retriever(retriever_id)

        if query_embedding is None and retriever.requires_query_embedding:
            query_embedding = self._embedding_manager.embed_one(query_text)

        query = RetrieverQuery(
            query_text=query_text,
            query_embedding=query_embedding,
            top_k=top_k,
            metadata_filter=metadata_filter,
            scope=scope,
            include_embeddings=include_embeddings,
        )

        return self._engine.retrieve(
            query,
            retriever_id=retriever_id,
        )

    def retrieve_query(
        self,
        query: RetrieverQuery,
        *,
        retriever_id: str,        
    ) -> List[RetrievalHit]:
        """
        Execute retrieval using an already constructed query.
        """

        return self._engine.retrieve(
            query,
            retriever_id=retriever_id,
        )