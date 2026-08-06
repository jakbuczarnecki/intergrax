# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List, Sequence

from intergrax.rag.retrievers.contracts.base_retriever import (
    RetrievalHit,
    RetrieverQuery,
)
from intergrax.rag.retrievers.contracts.base_retriever_manager import BaseRetrieverManager
from intergrax.rag.retrievers.engine.retriever_execution import RetrieverExecutionMetadata
from intergrax.rag.retrievers.pipeline.retriever_pipeline import RetrieverPipeline
from intergrax.rag.vectorstore.contracts.native_vectorstore import VectorStoreScope


class RetrieverManager(BaseRetrieverManager):
    """
    Public entry point for retrieval operations.

    Responsibilities
    ----------------
    - expose simplified retrieval API
    - delegate execution to RetrieverPipeline
    """

    def __init__(
        self,
        pipeline: RetrieverPipeline,
    ) -> None:

        self._pipeline = pipeline

    @property
    def last_execution(self) -> RetrieverExecutionMetadata | None:
        return self._pipeline.last_execution

    def retrieve(
        self,
        query_text: str,
        *,
        retriever_id: str,
        query_embedding: Sequence[float] | None = None,
        top_k: int = 5,
        metadata_filter=None,
        scope: VectorStoreScope | None = None,
        include_embeddings: bool = False,
    ) -> List[RetrievalHit]:
        """
        Retrieve candidates for query text.
        """

        return self._pipeline.retrieve(
            query_text=query_text,
            query_embedding=query_embedding,
            top_k=top_k,
            metadata_filter=metadata_filter,
            scope=scope,
            include_embeddings=include_embeddings,
            retriever_id=retriever_id,
        )

    def retrieve_query(
        self,
        query: RetrieverQuery,
        retriever_id: str,
    ) -> List[RetrievalHit]:
        """
        Retrieve using preconstructed RetrieverQuery.
        """

        return self._pipeline.retrieve_query(query, retriever_id=retriever_id)