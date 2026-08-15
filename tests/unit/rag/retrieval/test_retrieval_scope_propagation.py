# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.retrieval.agentic_loop import AgenticRetrievalLoop
from intergrax.rag.retrieval.retrieval_request import RetrievalRequest
from intergrax.rag.retrieval.retrieval_result import RetrievalResult, RetrievalTrace
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
    assert manager.supports_scoped_retrieval is True
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


def test_agentic_reconstructed_requests_preserve_exact_scope() -> None:
    scope = VectorStoreScope(
        tenant_id="tenant-a",
        namespace="namespace-a",
        workspace_id="workspace-a",
    )

    class _Service:
        def __init__(self) -> None:
            self.requests: list[RetrievalRequest] = []

        def retrieve_single_pass(self, request: RetrievalRequest, *, route_tier: str):
            assert route_tier == "deep"
            self.requests.append(request)
            return RetrievalResult(
                chunks=[],
                used=False,
                reason="no_hits",
                trace=RetrievalTrace(),
            )

    class _Refiner:
        def refine(self, query: str, result: RetrievalResult) -> str:
            return f"{query} refined"

    service = _Service()
    loop = AgenticRetrievalLoop(
        service,  # type: ignore[arg-type]
        RagProfile(
            agentic_max_iterations=2,
            agentic_min_chunks=1,
            agentic_min_score=0.5,
        ),
        query_refiner=_Refiner(),
    )

    loop.run(RetrievalRequest(query="initial", top_k=3, scope=scope))

    assert len(service.requests) == 2
    assert all(step.scope is scope for step in service.requests)
    assert all(
        (
            step.scope.tenant_id,
            step.scope.namespace,
            step.scope.workspace_id,
        )
        == ("tenant-a", "namespace-a", "workspace-a")
        for step in service.requests
    )


def test_scoped_request_rejects_manager_without_scoped_contract() -> None:
    scope = VectorStoreScope(
        tenant_id="tenant-a",
        namespace="namespace-a",
        workspace_id="workspace-a",
    )

    class _UnscopedManager:
        calls = 0

        def retrieve(
            self,
            query_text: str,
            *,
            retriever_id: str,
            top_k: int,
            metadata_filter=None,
            include_embeddings: bool = False,
        ):
            self.calls += 1
            return []

    manager = _UnscopedManager()
    service = RetrievalService(
        retriever_manager=manager,  # type: ignore[arg-type]
        profile=RagProfile(route_mode="off", retriever_id="vector_similarity"),
    )

    result = service.retrieve(RetrievalRequest(query="hello", scope=scope))

    assert result.used is False
    assert result.reason == "retriever_failed"
    assert result.trace.retrieval_error_kind == "scoped_retrieval_unsupported"
    assert manager.calls == 0


def test_scoped_request_rejects_manager_with_false_scoped_contract() -> None:
    scope = VectorStoreScope(tenant_id="tenant-a")

    class _FalseCapabilityManager:
        supports_scoped_retrieval = False
        calls = 0

        def retrieve(self, *args, **kwargs):
            self.calls += 1
            return []

    manager = _FalseCapabilityManager()
    service = RetrievalService(
        retriever_manager=manager,  # type: ignore[arg-type]
        profile=RagProfile(route_mode="off", retriever_id="vector_similarity"),
    )

    result = service.retrieve(RetrievalRequest(query="hello", scope=scope))

    assert result.used is False
    assert result.reason == "retriever_failed"
    assert result.trace.retrieval_error_kind == "scoped_retrieval_unsupported"
    assert manager.calls == 0


def test_supported_custom_manager_receives_exact_scope() -> None:
    scope = VectorStoreScope(
        tenant_id="tenant-a",
        namespace="namespace-a",
        workspace_id="workspace-a",
    )

    class _ScopedManager:
        supports_scoped_retrieval = True

        def __init__(self) -> None:
            self.received_scope = None

        def retrieve(
            self,
            query_text: str,
            *,
            retriever_id: str,
            top_k: int,
            metadata_filter=None,
            scope: VectorStoreScope,
            include_embeddings: bool = False,
        ):
            self.received_scope = scope
            return []

    manager = _ScopedManager()
    service = RetrievalService(
        retriever_manager=manager,  # type: ignore[arg-type]
        profile=RagProfile(route_mode="off", retriever_id="vector_similarity"),
    )

    result = service.retrieve(RetrievalRequest(query="hello", scope=scope))

    assert result.reason == "no_hits"
    assert manager.received_scope is scope


def test_internal_type_error_from_supported_manager_remains_visible() -> None:
    scope = VectorStoreScope(tenant_id="tenant-a")

    class _BrokenScopedManager:
        supports_scoped_retrieval = True

        def retrieve(self, *args, **kwargs):
            raise TypeError("internal retriever defect")

    service = RetrievalService(
        retriever_manager=_BrokenScopedManager(),  # type: ignore[arg-type]
        profile=RagProfile(route_mode="off", retriever_id="vector_similarity"),
    )

    with pytest.raises(TypeError, match="internal retriever defect"):
        service.retrieve(RetrievalRequest(query="hello", scope=scope))
