# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from typing import List, Sequence

import pytest
from langchain_core.documents import Document

from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.rag.retrievers.contracts.base_retriever import RetrieverCandidate
from intergrax.rag.profiles.rag_profile import RagProfile, rag_profile_from_env
from intergrax.rag.retrieval.retrieval_request import RetrievalRequest
from intergrax.rag.retrieval.retrieval_service import RetrievalService
from intergrax.rag.retrievers.bootstrap.retriever_bootstrap import create_default_retriever_registry
from intergrax.rag.retrievers.providers.multiquery_retriever import MultiQueryRetriever
from intergrax.rag.routing.query_router import QueryRouter
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager
from intergrax.rag.vectorstore.contracts.vector_store import MetadataFilter, VectorStoreHit

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _FakeLlm:
    def __init__(self, variants: List[str]) -> None:
        self._variants = variants
        self.calls = 0

    def generate_messages(self, messages: object, *, run_id: str = "", **kwargs: object) -> LLMAdapterResponse:
        del messages, run_id, kwargs
        self.calls += 1
        return LLMAdapterResponse(content="\n".join(self._variants), model="stub")


class CountingVectorManager(BaseVectorstoreManager):
    def __init__(self) -> None:
        self.query_calls = 0

    def query(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int,
        metadata_filter: MetadataFilter | None = None,
        include_embeddings: bool = False,
    ) -> Sequence[VectorStoreHit]:
        self.query_calls += 1
        return []

    def add_documents(
        self,
        documents: Sequence[Document],
        embeddings: Sequence[Sequence[float]],
        *,
        ids: Sequence[str] | None = None,
        base_metadata: dict | None = None,
        batch_size: int | None = None,
    ) -> Sequence[str] | None:
        return None

    def delete(self, ids: Sequence[str]) -> None:
        return None

    def count(self) -> int:
        return 0


def test_effective_retriever_deep_uses_multiquery_when_expansion_enabled() -> None:
    profile = RagProfile(query_expansion="deterministic", deep_retriever_id="fusion")
    assert profile.effective_retriever(route_tier="deep") == "multiquery"


def test_effective_retriever_deep_uses_fusion_when_expansion_off() -> None:
    profile = RagProfile(query_expansion="off", deep_retriever_id="fusion")
    assert profile.effective_retriever(route_tier="deep") == "fusion"


def test_effective_retriever_deep_prefers_graph_rag_over_multiquery() -> None:
    profile = RagProfile(
        query_expansion="deterministic",
        graph_rag_enabled=True,
        deep_retriever_id="fusion",
    )
    assert profile.effective_retriever(route_tier="deep") == "graph_rag"


class _TrackingRetrieverManager:
    def __init__(self) -> None:
        self.last_retriever_id = ""

    def retrieve(
        self,
        query_text: str,
        *,
        retriever_id: str,
        query_embedding: Sequence[float] | None = None,
        top_k: int = 5,
        metadata_filter: MetadataFilter | None = None,
        include_embeddings: bool = False,
    ) -> List[RetrieverCandidate]:
        del query_text, query_embedding, top_k, metadata_filter, include_embeddings
        self.last_retriever_id = retriever_id
        return [
            RetrieverCandidate(id="c1", content="hit", metadata={}, score=0.9),
        ]


def test_retrieval_service_routes_deep_tier_to_multiquery() -> None:
    profile = RagProfile(
        query_expansion="deterministic",
        route_mode="auto",
        deep_query_min_words=3,
        enable_rerank=False,
    )
    manager = _TrackingRetrieverManager()
    service = RetrievalService(retriever_manager=manager, profile=profile)  # type: ignore[arg-type]

    long_query = " ".join(["word"] * 15)
    assert QueryRouter(profile).route(long_query) == "deep"

    result = service.retrieve(RetrievalRequest(query=long_query, top_k=3))
    assert result.used is True
    assert result.trace.retriever_id == "multiquery"
    assert manager.last_retriever_id == "multiquery"


def test_bootstrap_injects_llm_query_expander_from_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERGRAX_RAG_QUERY_EXPANSION", "llm")
    profile = rag_profile_from_env()
    assert profile.query_expansion == "llm"

    llm = _FakeLlm(["expanded variant alpha", "expanded variant beta"])
    vs = CountingVectorManager()
    from intergrax.rag.embedding.bootstrap.default_embedding_engine import create_default_embedding_manager

    registry = create_default_retriever_registry(
        vector_store=vs,
        embedding_manager=create_default_embedding_manager(),
        profile=profile,
        llm_for_query_expansion=llm,  # type: ignore[arg-type]
    )
    multi = registry.get("multiquery")
    assert isinstance(multi, MultiQueryRetriever)

    from intergrax.rag.retrievers.contracts.base_retriever import RetrieverQuery

    multi.retrieve(
        RetrieverQuery(
            query_text="enterprise knowledge retrieval platform",
            query_embedding=None,
            top_k=2,
            metadata_filter=None,
            include_embeddings=False,
        )
    )
    assert llm.calls == 1
    assert vs.query_calls >= 2
