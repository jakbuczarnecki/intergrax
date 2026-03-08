# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List, Sequence

from intergrax.rag.retrievers.contracts.base_retriever import (
    RetrieverCandidate,
    RetrieverQuery,
)
from intergrax.rag.retrievers.retriever_pipeline import RetrieverPipeline


class RetrieverManager:
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

    def retrieve(
        self,
        query_text: str,
        *,
        query_embedding: Sequence[float] | None = None,
        top_k: int = 5,
        metadata_filter=None,
        include_embeddings: bool = False,
    ) -> List[RetrieverCandidate]:
        """
        Retrieve candidates for query text.
        """

        return self._pipeline.retrieve(
            query_text=query_text,
            query_embedding=query_embedding,
            top_k=top_k,
            metadata_filter=metadata_filter,
            include_embeddings=include_embeddings,
        )

    def retrieve_query(
        self,
        query: RetrieverQuery,
    ) -> List[RetrieverCandidate]:
        """
        Retrieve using preconstructed RetrieverQuery.
        """

        return self._pipeline.retrieve_query(query)