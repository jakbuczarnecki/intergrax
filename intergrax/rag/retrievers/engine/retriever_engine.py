# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List

from intergrax.rag.retrievers.contracts.base_retriever import (
    BaseRetriever,
    RetrieverCandidate,
    RetrieverQuery,
)
from intergrax.rag.retrievers.registry.retriever_registry import RetrieverRegistry


class RetrieverEngine:
    """
    Execution engine responsible for performing retrieval
    using a retriever resolved from the RetrieverRegistry.

    Responsibilities
    ----------------
    - retriever resolution
    - retry handling
    """

    def __init__(
        self,
        registry: RetrieverRegistry,
        *,
        max_retries: int = 1,
    ) -> None:

        self._registry = registry
        self._max_retries = int(max_retries)

    def retrieve(
        self,
        query: RetrieverQuery,
        retriever_id: str,
    ) -> List[RetrieverCandidate]:
        """
        Execute retrieval for a query using a registered retriever.
        """

        retriever: BaseRetriever = self._registry.get(retriever_id)

        return self._retrieve_with_retry(retriever, query)

    def _retrieve_with_retry(
        self,
        retriever: BaseRetriever,
        query: RetrieverQuery,
    ) -> List[RetrieverCandidate]:

        last_exc: Exception | None = None

        for _ in range(self._max_retries + 1):
            try:
                return retriever.retrieve(query)
            except Exception as exc:
                last_exc = exc

        raise RuntimeError(
            "Retriever execution failed after retries"
        ) from last_exc