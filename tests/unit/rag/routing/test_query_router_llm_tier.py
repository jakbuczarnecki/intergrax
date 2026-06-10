# © Artur Czarnecki. All rights reserved.

"""M-RAG.32 — optional LLM QueryRouter tier classifier."""

from __future__ import annotations

from typing import List, Sequence
from unittest.mock import MagicMock

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.retrieval.retrieval_request import RetrievalRequest
from intergrax.rag.retrieval.retrieval_service import RetrievalService
from intergrax.rag.retrievers.contracts.base_retriever import (
    BaseRetriever,
    RetrieverCandidate,
    RetrieverQuery,
)
from intergrax.rag.retrievers.contracts.base_retriever_manager import BaseRetrieverManager
from intergrax.rag.routing.llm_tier_classifier import parse_route_tier_response
from intergrax.rag.routing.query_router import QueryRouter

pytestmark = pytest.mark.gate


class _StubLLM:
    def __init__(self, content: str) -> None:
        self._content = content
        self.calls: list[list[ChatMessage]] = []

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        run_id: str = "",
        **kwargs: object,
    ) -> LLMAdapterResponse:
        self.calls.append(list(messages))
        return LLMAdapterResponse(content=self._content, model="stub")


class _TierTrackingRetrieverManager(BaseRetrieverManager):
    def __init__(self) -> None:
        self.last_retriever_id = ""

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
        self.last_retriever_id = retriever_id
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


def test_parse_route_tier_response_accepts_single_token() -> None:
    assert parse_route_tier_response("deep") == "deep"
    assert parse_route_tier_response("Answer: standard.") == "standard"


def test_query_router_uses_heuristic_when_llm_route_disabled() -> None:
    llm = _StubLLM("deep")
    router = QueryRouter(RagProfile(route_mode="auto", llm_route_enabled=False), llm=llm)

    assert router.route("hi there") == "fast"
    assert router.last_route_classifier == "heuristic"
    assert llm.calls == []


def test_query_router_llm_routes_ambiguous_query_to_deep() -> None:
    llm = _StubLLM("deep")
    router = QueryRouter(
        RagProfile(route_mode="auto", llm_route_enabled=True, deep_query_min_words=20),
        llm=llm,
    )

    query = "What are the implications?"
    assert router.route(query) == "deep"
    assert router.last_route_classifier == "llm"
    assert len(llm.calls) == 1


def test_query_router_falls_back_to_heuristic_when_llm_fails() -> None:
    llm = MagicMock(spec=LLMAdapter)
    llm.generate_messages.side_effect = RuntimeError("provider down")
    router = QueryRouter(RagProfile(route_mode="auto", llm_route_enabled=True), llm=llm)

    assert router.route("hi there") == "fast"
    assert router.last_route_classifier == "heuristic"


def test_retrieval_service_uses_deep_retriever_when_llm_routes_deep() -> None:
    manager = _TierTrackingRetrieverManager()
    profile = RagProfile(
        route_mode="auto",
        llm_route_enabled=True,
        deep_retriever_id="fusion",
        fast_retriever_id="vector_similarity",
        retriever_id="hybrid",
        enable_rerank=False,
        query_expansion="off",
    )
    service = RetrievalService(
        retriever_manager=manager,
        profile=profile,
        llm_for_routing=_StubLLM("deep"),
    )

    result = service.retrieve(RetrievalRequest(query="What are the implications?"))

    assert result.used is True
    assert result.trace.route_tier == "deep"
    assert result.trace.route_classifier == "llm"
    assert manager.last_retriever_id == "fusion"
