# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.retrieval.agentic_loop import AgenticRetrievalLoop
from intergrax.rag.retrieval.query_refiner import LlmQueryRefiner
from intergrax.rag.retrieval.retrieval_request import RetrievalRequest
from intergrax.rag.retrieval.retrieval_result import RetrievalChunk, RetrievalResult, RetrievalTrace
from intergrax.rag.retrieval.retrieval_service import RetrievalService

pytestmark = pytest.mark.unit


class _StubService(RetrievalService):
    def __init__(self) -> None:
        self._calls = 0

    def retrieve_single_pass(self, request: RetrievalRequest, *, route_tier=None) -> RetrievalResult:
        self._calls += 1
        if self._calls >= 2:
            return RetrievalResult(
                chunks=[
                    RetrievalChunk(id="1", text="ok", score=0.9, metadata={}),
                    RetrievalChunk(id="2", text="ok2", score=0.8, metadata={}),
                ],
                used=True,
                reason="ok",
                trace=RetrievalTrace(),
            )
        return RetrievalResult(
            chunks=[RetrievalChunk(id="1", text="weak", score=0.1, metadata={})],
            used=True,
            reason="ok",
            trace=RetrievalTrace(),
        )


class _FakeLlmRefiner:
    def generate_messages(self, messages, run_id: str = ""):
        del messages, run_id

        class _R:
            content = "refined search query with more terms"

        return _R()


def test_agentic_loop_with_llm_refiner() -> None:
    profile = RagProfile(
        agentic_enabled=True,
        agentic_max_iterations=3,
        agentic_min_chunks=2,
        agentic_min_score=0.5,
        agentic_query_mode="llm",
    )
    loop = AgenticRetrievalLoop(
        _StubService(),
        profile,
        query_refiner=LlmQueryRefiner(_FakeLlmRefiner()),
    )
    result = loop.run(RetrievalRequest(query="initial"))
    assert result.used
    assert len(result.chunks) >= 2


def test_agentic_loop_refines_until_min_chunks() -> None:
    profile = RagProfile(
        agentic_enabled=True,
        agentic_max_iterations=3,
        agentic_min_chunks=2,
        agentic_min_score=0.5,
    )
    loop = AgenticRetrievalLoop(_StubService(), profile)  # type: ignore[arg-type]
    result = loop.run(RetrievalRequest(query="initial"))
    assert result.used
    assert len(result.chunks) >= 2
    assert result.trace.agentic_stopped == "sufficient_context"
