# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
RAG contract test: no documents -> no grounded answer.

The system must not hallucinate answers when retrieval yields no documents.
It must return an explicit fallback answer and report zero LLM work.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from intergrax.rag.rag_answerer import RagAnswerer
from intergrax.rag.rag_retriever import RagRetriever
from tests._support.builder import FakeLLMAdapter


pytestmark = pytest.mark.unit


class _DummyVectorStore:
    pass


class _DummyEmbeddingManager:
    pass


class _EmptyRetriever(RagRetriever):
    """
    Deterministic retriever that returns no hits.

    Note: RagAnswerer.run() calls retrieve(question=..., top_k=..., where=...).
    We only implement the minimal signature needed for this contract test.
    """

    def retrieve(
        self,
        *,
        question: str,
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> List[Dict[str, Any]]:
        return []


def test_rag_no_documents_returns_fallback_and_zero_llm_work() -> None:
    retriever = _EmptyRetriever(
        vector_store=_DummyVectorStore(),
        embedding_manager=_DummyEmbeddingManager(),
    )
    answerer = RagAnswerer(retriever=retriever, llm=FakeLLMAdapter())

    result = answerer.run("What is Intergrax?")

    # Contract: result is a dict with a stable schema.
    assert isinstance(result, dict)

    # Contract: no hits => explicit fallback answer, no context, no sources.
    assert result["answer"] == "No sufficiently relevant context fragments were found to answer the question."
    assert result["output_structure"] is None
    assert result["sources"] == []
    assert result["summary"] is None
    assert result["context"] == ""
    assert result["messages"] == []

    # Contract: no hits => no LLM work (time/cost must be zeroed).
    stats = result["stats"]
    assert stats["hits_in"] == 0
    assert stats["context_chars"] == 0
    assert stats["rerank_s"] == 0.0
    assert stats["llm_s"] == 0.0


def test_rag_single_hit_calls_llm_once_and_uses_context() -> None:
    class _OneHitRetriever(RagRetriever):
        def retrieve(
            self,
            *,
            question: str,
            top_k: int = 5,
            where: Optional[dict] = None,
        ) -> List[Dict[str, Any]]:
            return [
                {
                    "text": "Intergrax is an AI-native runtime framework.",
                    "source": "doc-1",
                    "page": None,
                    "score": 0.9,
                }
            ]

    retriever = _OneHitRetriever(
        vector_store=_DummyVectorStore(),
        embedding_manager=_DummyEmbeddingManager(),
    )

    llm = FakeLLMAdapter(fixed_text="ANSWER_FROM_LLM")

    answerer = RagAnswerer(
        retriever=retriever,
        llm=llm,
    )

    result = answerer.run("What is Intergrax?")

    # ---- Contract assertions ----
    assert isinstance(result, dict)

    # Context must include retrieved text
    assert "Intergrax is an AI-native runtime framework." in result["context"]

    # Answer must come from LLM
    assert result["answer"] == "ANSWER_FROM_LLM"

    # Exactly one LLM call
    stats = llm.usage.get_run_stats()
    assert stats.calls == 1
    assert stats.errors == 0

    # Sources must be propagated
    assert len(result["sources"]) == 1

    source = result["sources"][0]
    assert source.source == "doc-1"
