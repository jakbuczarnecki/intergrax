# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from typing import List, Optional, Sequence

import pytest

from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.retrieval.retrieval_request import RetrievalRequest
from intergrax.rag.retrieval.retrieval_service import RetrievalService
from intergrax.rag.retrievers.contracts.base_retriever import (
    BaseRetriever,
    RetrieverCandidate,
    RetrieverQuery,
)
from intergrax.rag.retrievers.contracts.base_retriever_manager import BaseRetrieverManager
from intergrax.rag.retrievers.registry.retriever_registry import RetrieverRegistry
from intergrax.rag.routing.query_router import QueryRouter

pytestmark = pytest.mark.unit


class StubRetriever(BaseRetriever):
    @classmethod
    def name(cls) -> str:
        return "stub"

    def retrieve(self, query: RetrieverQuery) -> List[RetrieverCandidate]:
        return [
            RetrieverCandidate(
                id="c1",
                content=f"answer for {query.query_text}",
                metadata={},
                score=0.9,
            )
        ]


class StubRetrieverManager(BaseRetrieverManager):
    def __init__(self) -> None:
        self.last_retriever_id: str = ""

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
        return StubRetriever().retrieve(
            RetrieverQuery(
                query_text=query_text,
                query_embedding=None,
                top_k=top_k,
                metadata_filter=metadata_filter,
            )
        )

    def retrieve_query(self, query: RetrieverQuery, retriever_id: str) -> List[RetrieverCandidate]:
        self.last_retriever_id = retriever_id
        return StubRetriever().retrieve(query)


def test_query_router_fast_tier() -> None:
    profile = RagProfile(route_mode="auto", deep_query_min_words=20)
    router = QueryRouter(profile)
    assert router.route("hi there") == "fast"


def test_retrieval_service_uses_profile_retriever() -> None:
    manager = StubRetrieverManager()
    profile = RagProfile(
        retriever_id="hybrid",
        fast_retriever_id="vector_similarity",
        enable_rerank=False,
        route_mode="off",
    )
    service = RetrievalService(retriever_manager=manager, profile=profile)
    result = service.retrieve(RetrievalRequest(query="What is Intergrax?"))
    assert result.used is True
    assert result.chunks[0].text.startswith("answer for")
    assert manager.last_retriever_id == "hybrid"


def test_recall_at_k_metric() -> None:
    from intergrax.rag.evaluation.metrics import recall_at_k

    assert recall_at_k(["a", "b", "c"], {"b", "d"}, k=2) == 0.5
