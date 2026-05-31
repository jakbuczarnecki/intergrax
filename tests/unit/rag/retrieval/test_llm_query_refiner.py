# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.retrieval.query_refiner import LlmQueryRefiner, resolve_query_refiner
from intergrax.rag.retrieval.retrieval_result import RetrievalChunk, RetrievalResult, RetrievalTrace

pytestmark = pytest.mark.unit


class _FakeLlm:
    def generate_messages(self, messages, run_id: str = ""):
        del messages, run_id

        class _R:
            content = "Intergrax RAG RetrievalService hybrid"

        return _R()


def test_llm_query_refiner_returns_rewritten_query() -> None:
    refiner = LlmQueryRefiner(_FakeLlm())
    result = RetrievalResult(
        chunks=[RetrievalChunk(id="1", text="weak context", score=0.1, metadata={})],
        used=True,
        reason="ok",
        trace=RetrievalTrace(),
    )
    out = refiner.refine("initial query", result)
    assert "Intergrax" in out


def test_resolve_query_refiner_llm_mode() -> None:
    profile = RagProfile(agentic_query_mode="llm")
    refiner = resolve_query_refiner(profile, llm=_FakeLlm())
    assert isinstance(refiner, LlmQueryRefiner)
