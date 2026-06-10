# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from typing import List, Type

import pytest

from intergrax.integrations._shared.circuit_breaker import IntegrationCircuitBreakerConfig
from intergrax.rag.retrieval.retrieval_errors import RetrievalError, RetrievalErrorKind
from intergrax.rag.retrievers.contracts.base_retriever import (
    BaseRetriever,
    RetrieverCandidate,
    RetrieverQuery,
)
from intergrax.rag.retrievers.engine.retriever_engine import RetrieverEngine
from intergrax.rag.retrievers.registry.retriever_registry import RetrieverRegistry
from intergrax.rag.retrievers.resilience.vector_store_circuit_breaker import (
    RetrieverVectorCircuitBreaker,
)

pytestmark = pytest.mark.gate


def _flaky_retriever_class(retriever_name: str, *, fail_times: int) -> Type[BaseRetriever]:
    class _FlakyRetriever(BaseRetriever):
        _calls = 0

        @classmethod
        def name(cls) -> str:
            return retriever_name

        def retrieve(self, query: RetrieverQuery) -> List[RetrieverCandidate]:
            type(self)._calls += 1
            if type(self)._calls <= fail_times:
                raise TimeoutError(f"{retriever_name} timeout")
            return [
                RetrieverCandidate(
                    id=f"{retriever_name}-1",
                    content="hit",
                    metadata={},
                    score=0.8,
                )
            ]

    return _FlakyRetriever


def test_retriever_engine_retries_before_fallback() -> None:
    registry = RetrieverRegistry()
    fusion_cls = _flaky_retriever_class("fusion", fail_times=1)
    registry.register(fusion_cls())
    registry.register(_flaky_retriever_class("hybrid", fail_times=0)())

    engine = RetrieverEngine(registry, max_retries=2)
    query = RetrieverQuery(query_text="hello", query_embedding=None, top_k=3)
    candidates = engine.retrieve(query, "fusion")

    assert candidates[0].id == "fusion-1"
    assert fusion_cls._calls == 2
    assert engine.last_execution is not None
    assert engine.last_execution.used_retriever_id == "fusion"
    assert engine.last_execution.fallback_applied is False


def test_retriever_engine_falls_back_fusion_to_hybrid() -> None:
    registry = RetrieverRegistry()
    registry.register(_flaky_retriever_class("fusion", fail_times=10)())
    hybrid_cls = _flaky_retriever_class("hybrid", fail_times=0)
    registry.register(hybrid_cls())
    registry.register(_flaky_retriever_class("vector_similarity", fail_times=0)())

    engine = RetrieverEngine(registry, max_retries=1)
    query = RetrieverQuery(query_text="hello", query_embedding=None, top_k=3)
    candidates = engine.retrieve(query, "fusion")

    assert candidates[0].id == "hybrid-1"
    assert hybrid_cls._calls == 1
    assert engine.last_execution is not None
    assert engine.last_execution.fallback_applied is True
    assert engine.last_execution.attempted_retriever_ids == ["fusion", "hybrid"]


def test_retriever_engine_raises_structured_error_when_chain_exhausted() -> None:
    registry = RetrieverRegistry()
    registry.register(_flaky_retriever_class("fusion", fail_times=10)())
    registry.register(_flaky_retriever_class("hybrid", fail_times=10)())

    engine = RetrieverEngine(registry, max_retries=0)
    query = RetrieverQuery(query_text="hello", query_embedding=None, top_k=3)

    with pytest.raises(RetrievalError) as exc_info:
        engine.retrieve(query, "fusion")

    exc = exc_info.value
    assert exc.kind in {RetrievalErrorKind.UNKNOWN, RetrievalErrorKind.VECTOR_BACKEND_FAILURE}
    assert exc.attempted_retriever_ids == ("fusion", "hybrid")


def test_vector_circuit_breaker_opens_after_threshold() -> None:
    registry = RetrieverRegistry()
    registry.register(_flaky_retriever_class("vector_similarity", fail_times=10)())

    breaker = RetrieverVectorCircuitBreaker(
        config=IntegrationCircuitBreakerConfig(failure_threshold=2, recovery_timeout_seconds=30.0),
    )
    engine = RetrieverEngine(
        registry,
        max_retries=0,
        fallback_enabled=False,
        vector_circuit_breaker=breaker,
    )
    query = RetrieverQuery(query_text="hello", query_embedding=None, top_k=3)

    with pytest.raises(RetrievalError):
        engine.retrieve(query, "vector_similarity")
    with pytest.raises(RetrievalError):
        engine.retrieve(query, "vector_similarity")

    with pytest.raises(RetrievalError) as exc_info:
        engine.retrieve(query, "vector_similarity")

    assert exc_info.value.kind == RetrievalErrorKind.CIRCUIT_OPEN
