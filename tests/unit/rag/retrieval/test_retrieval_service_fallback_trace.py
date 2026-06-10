# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from typing import List, Sequence

import pytest

from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.retrieval.retrieval_errors import RetrievalError, RetrievalErrorKind
from intergrax.rag.retrieval.retrieval_request import RetrievalRequest
from intergrax.rag.retrieval.retrieval_service import RetrievalService
from intergrax.rag.retrievers.contracts.base_retriever import (
    RetrieverCandidate,
    RetrieverQuery,
)
from intergrax.rag.retrievers.contracts.base_retriever_manager import BaseRetrieverManager
from intergrax.rag.retrievers.engine.retriever_execution import RetrieverExecutionMetadata

pytestmark = pytest.mark.gate


class _FallbackAwareManager(BaseRetrieverManager):
    def __init__(self) -> None:
        self.last_execution = RetrieverExecutionMetadata(
            requested_retriever_id="fusion",
            used_retriever_id="hybrid",
            attempted_retriever_ids=["fusion", "hybrid"],
            fallback_applied=True,
        )

    def retrieve(
        self,
        query_text: str,
        *,
        retriever_id: str,
        query_embedding: Sequence[float] | None = None,
        top_k: int = 5,
        metadata_filter=None,
        include_embeddings: bool = False,
    ) -> List[RetrieverCandidate]:
        return [
            RetrieverCandidate(
                id="c1",
                content=f"answer for {query_text}",
                metadata={},
                score=0.9,
            )
        ]

    def retrieve_query(self, query: RetrieverQuery, retriever_id: str) -> List[RetrieverCandidate]:
        return self.retrieve(query.query_text, retriever_id=retriever_id, top_k=query.top_k)


class _FailingManager(BaseRetrieverManager):
    def retrieve(self, query_text: str, **kwargs: object) -> List[RetrieverCandidate]:
        raise RetrievalError(
            kind=RetrievalErrorKind.RETRIEVER_EXHAUSTED,
            message="all retrievers failed",
            retriever_id="fusion",
            attempted_retriever_ids=("fusion", "hybrid", "vector_similarity"),
            retryable=False,
        )

    def retrieve_query(self, query: RetrieverQuery, retriever_id: str) -> List[RetrieverCandidate]:
        return self.retrieve(query.query_text, retriever_id=retriever_id)


def test_retrieval_service_records_fallback_trace() -> None:
    service = RetrievalService(
        retriever_manager=_FallbackAwareManager(),
        profile=RagProfile(enable_rerank=False, retriever_id="fusion"),
    )
    result = service.retrieve(RetrievalRequest(query="hello world"))

    assert result.used is True
    assert result.trace.retriever_id == "hybrid"
    assert result.trace.fallback_applied is True
    assert result.trace.attempted_retriever_ids == ["fusion", "hybrid"]


def test_retrieval_service_maps_retrieval_error_to_result() -> None:
    service = RetrievalService(
        retriever_manager=_FailingManager(),
        profile=RagProfile(enable_rerank=False),
    )
    result = service.retrieve(RetrievalRequest(query="hello world"))

    assert result.used is False
    assert result.reason == "retriever_failed"
    assert result.trace.retrieval_error_kind == RetrievalErrorKind.RETRIEVER_EXHAUSTED.value
    assert result.trace.attempted_retriever_ids == ["fusion", "hybrid", "vector_similarity"]
