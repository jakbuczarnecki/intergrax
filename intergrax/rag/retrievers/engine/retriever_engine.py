# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List, Optional

from intergrax.rag.retrieval.retrieval_errors import (
    RetrievalError,
    RetrievalErrorKind,
    classify_retrieval_exception,
)
from intergrax.rag.retrievers.contracts.base_retriever import (
    BaseRetriever,
    RetrieverCandidate,
    RetrieverQuery,
)
from intergrax.rag.retrievers.engine.retriever_execution import RetrieverExecutionMetadata
from intergrax.rag.retrievers.registry.retriever_registry import RetrieverRegistry
from intergrax.rag.retrievers.resilience.retriever_fallback import retriever_fallback_chain
from intergrax.rag.retrievers.resilience.vector_store_circuit_breaker import (
    RetrieverVectorCircuitBreaker,
)


class RetrieverEngine:
    """
    Execution engine responsible for performing retrieval
    using a retriever resolved from the RetrieverRegistry.

    Responsibilities
    ----------------
    - retriever resolution
    - retry handling
    - canonical fallback chain after retry exhaustion
    """

    DEFAULT_MAX_RETRIES = 2

    def __init__(
        self,
        registry: RetrieverRegistry,
        *,
        max_retries: int = DEFAULT_MAX_RETRIES,
        fallback_enabled: bool = True,
        vector_circuit_breaker: Optional[RetrieverVectorCircuitBreaker] = None,
    ) -> None:
        self._registry = registry
        self._max_retries = int(max_retries)
        self._fallback_enabled = bool(fallback_enabled)
        self._vector_circuit_breaker = vector_circuit_breaker
        self._last_execution: Optional[RetrieverExecutionMetadata] = None

    @property
    def last_execution(self) -> Optional[RetrieverExecutionMetadata]:
        return self._last_execution

    def retrieve(
        self,
        query: RetrieverQuery,
        retriever_id: str,
    ) -> List[RetrieverCandidate]:
        """Execute retrieval for a query using a registered retriever."""
        if self._fallback_enabled:
            chain = retriever_fallback_chain(retriever_id, self._registry.list_names())
        else:
            chain = [retriever_id]

        attempted: list[str] = []
        last_error: Optional[RetrievalError] = None

        for candidate_id in chain:
            attempted.append(candidate_id)
            try:
                retriever = self.get_retriever(candidate_id)
            except Exception as exc:
                last_error = classify_retrieval_exception(
                    exc,
                    retriever_id=candidate_id,
                    attempted_retriever_ids=attempted,
                )
                continue

            try:
                candidates = self._retrieve_with_retry(
                    retriever,
                    query,
                    retriever_id=candidate_id,
                )
            except RetrievalError as exc:
                last_error = RetrievalError(
                    kind=exc.kind,
                    message=exc.message,
                    retriever_id=exc.retriever_id,
                    attempted_retriever_ids=tuple(attempted),
                    retryable=exc.retryable,
                    cause=exc.cause,
                )
                continue

            self._last_execution = RetrieverExecutionMetadata(
                requested_retriever_id=retriever_id,
                used_retriever_id=candidate_id,
                attempted_retriever_ids=list(attempted),
                fallback_applied=candidate_id != retriever_id,
            )
            return candidates

        if last_error is not None:
            raise RetrievalError(
                kind=last_error.kind,
                message=last_error.message,
                retriever_id=last_error.retriever_id,
                attempted_retriever_ids=tuple(attempted),
                retryable=last_error.retryable,
                cause=last_error.cause,
            )

        raise RetrievalError(
            kind=RetrievalErrorKind.RETRIEVER_EXHAUSTED,
            message="Retriever execution failed after retries and fallback chain",
            retriever_id=retriever_id,
            attempted_retriever_ids=tuple(attempted),
            retryable=False,
        )

    def get_retriever(self, retriever_id: str) -> BaseRetriever:
        return self._registry.get(retriever_id)

    def _retrieve_with_retry(
        self,
        retriever: BaseRetriever,
        query: RetrieverQuery,
        *,
        retriever_id: str,
    ) -> List[RetrieverCandidate]:
        last_exc: Exception | None = None

        for _ in range(self._max_retries + 1):
            try:
                if self._vector_circuit_breaker is not None:
                    return self._vector_circuit_breaker.call(lambda: retriever.retrieve(query))
                return retriever.retrieve(query)
            except Exception as exc:
                last_exc = exc

        if last_exc is None:
            raise RetrievalError(
                kind=RetrievalErrorKind.RETRIEVER_EXHAUSTED,
                message=f"Retriever {retriever_id!r} failed without exception",
                retriever_id=retriever_id,
                retryable=False,
            )

        raise classify_retrieval_exception(
            last_exc,
            retriever_id=retriever_id,
        )
