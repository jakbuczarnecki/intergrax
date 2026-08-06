# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.retrieval.retrieval_request import RetrievalRequest
from intergrax.rag.retrieval.retrieval_service import RetrievalService
from intergrax.rag.retrievers.contracts.base_retriever import RetrieverQuery
from intergrax.rag.retrievers.pipeline.retriever_pipeline import RetrieverPipeline
from intergrax.rag.retrievers.retriever_manager import RetrieverManager
from intergrax.rag.vectorstore.contracts.native_vectorstore import VectorStoreScope

pytestmark = pytest.mark.gate


class _EmbeddingManager:
    def embed_one(self, text: str):
        raise AssertionError("embedding should not be needed")


class _Retriever:
    requires_query_embedding = False


class _Engine:
    last_execution = None

    def __init__(self) -> None:
        self.query: RetrieverQuery | None = None
        self.retriever = _Retriever()

    def get_retriever(self, retriever_id: str):
        return self.retriever

    def retrieve(self, query: RetrieverQuery, *, retriever_id: str):
        self.query = query
        return []


def test_scope_survives_retrieval_service_manager_pipeline_query() -> None:
    scope = VectorStoreScope(
        tenant_id="tenant-a",
        namespace="rag",
        workspace_id="workspace-a",
    )
    engine = _Engine()
    pipeline = RetrieverPipeline(
        engine,
        embedding_manager=_EmbeddingManager(),
    )
    manager = RetrieverManager(pipeline)
    service = RetrievalService(
        retriever_manager=manager,
        profile=RagProfile(
            route_mode="off",
            retriever_id="vector_similarity",
            enable_rerank=False,
        ),
    )

    result = service.retrieve(RetrievalRequest(query="hello", scope=scope))

    assert result.used is False
    assert engine.query is not None
    assert engine.query.scope is scope


def test_retriever_query_scope_is_immutable() -> None:
    query = RetrieverQuery(
        query_text="hello",
        query_embedding=None,
        top_k=3,
        scope=VectorStoreScope(tenant_id="tenant-a"),
    )

    with pytest.raises(FrozenInstanceError):
        query.scope = VectorStoreScope(tenant_id="tenant-b")
