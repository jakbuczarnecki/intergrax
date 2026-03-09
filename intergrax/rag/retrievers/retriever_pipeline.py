# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List, Sequence

from intergrax.rag.retrievers.contracts.base_retriever import (
    RetrieverCandidate,
    RetrieverQuery,
)
from intergrax.rag.retrievers.engine.retriever_engine import RetrieverEngine


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
        retriever_id: str,
    ) -> None:
        self._engine = engine
        self._retriever_id = retriever_id

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
        Execute retrieval for a query text.
        """

        query = RetrieverQuery(
            query_text=query_text,
            query_embedding=query_embedding,
            top_k=top_k,
            metadata_filter=metadata_filter,
            include_embeddings=include_embeddings,
        )

        return self._engine.retrieve(
            query,
            retriever_id=self._retriever_id,
        )

    def retrieve_query(
        self,
        query: RetrieverQuery,
    ) -> List[RetrieverCandidate]:
        """
        Execute retrieval using an already constructed query.
        """

        return self._engine.retrieve(
            query,
            retriever_id=self._retriever_id,
        )