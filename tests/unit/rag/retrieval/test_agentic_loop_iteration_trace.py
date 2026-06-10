# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import time

import pytest

from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.retrieval.agentic_loop import AgenticRetrievalLoop
from intergrax.rag.retrieval.retrieval_request import RetrievalRequest
from intergrax.rag.retrieval.retrieval_result import RetrievalChunk, RetrievalResult, RetrievalTrace
from intergrax.rag.retrieval.retrieval_service import RetrievalService

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _WeakResultService(RetrievalService):
    def __init__(self) -> None:
        self.retriever_ids: list[str | None] = []

    def retrieve_single_pass(self, request: RetrievalRequest, *, route_tier=None) -> RetrievalResult:
        self.retriever_ids.append(request.retriever_id)
        rid = request.retriever_id or "profile_default"
        return RetrievalResult(
            chunks=[RetrievalChunk(id="1", text="weak", score=0.1, metadata={})],
            used=True,
            reason="ok",
            trace=RetrievalTrace(retriever_id=rid, retrieval_latency_ms=12.5),
        )


def test_agentic_loop_per_iteration_retriever_schedule() -> None:
    profile = RagProfile(
        agentic_enabled=True,
        agentic_max_iterations=3,
        agentic_min_chunks=5,
        agentic_min_score=0.9,
        agentic_iteration_retriever_ids=("fusion", "hybrid", "vector_similarity"),
    )
    service = _WeakResultService()
    loop = AgenticRetrievalLoop(service, profile)  # type: ignore[arg-type]
    result = loop.run(RetrievalRequest(query="initial"))

    assert service.retriever_ids == ["fusion", "hybrid", "vector_similarity"]
    assert result.trace.agentic_per_iteration_retriever_ids == [
        "fusion",
        "hybrid",
        "vector_similarity",
    ]
    assert result.trace.agentic_per_iteration_latency_ms == [12.5, 12.5, 12.5]
    assert result.trace.agentic_refine_calls == 3
    assert result.trace.agentic_stopped == "max_iterations"
    assert result.trace.agentic_total_latency_ms is not None


class _LatencyAdvancingService(_WeakResultService):
    def __init__(self, clock: dict[str, float]) -> None:
        super().__init__()
        self._clock = clock

    def retrieve_single_pass(self, request: RetrievalRequest, *, route_tier=None) -> RetrievalResult:
        self._clock["t"] += 0.02
        return super().retrieve_single_pass(request, route_tier=route_tier)


def test_agentic_loop_latency_budget_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = {"t": 0.0}

    def fake_perf_counter() -> float:
        return clock["t"]

    monkeypatch.setattr(time, "perf_counter", fake_perf_counter)

    profile = RagProfile(
        agentic_enabled=True,
        agentic_max_iterations=5,
        agentic_min_chunks=5,
        agentic_min_score=0.9,
        agentic_max_total_latency_ms=15.0,
    )
    service = _LatencyAdvancingService(clock)
    loop = AgenticRetrievalLoop(service, profile)  # type: ignore[arg-type]
    result = loop.run(RetrievalRequest(query="initial"))

    assert len(result.trace.agentic_per_iteration_retriever_ids) == 1
    assert result.trace.agentic_latency_budget_ms == 15.0
    assert result.trace.agentic_stopped == "latency_budget"
    assert result.trace.agentic_refine_calls == 1
