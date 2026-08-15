# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from collections.abc import Iterator
from typing import List, Optional

import pytest

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.vectorstore.contracts.vector_store import MetadataFilter, VectorStoreHit
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.providers.rag.contracts import RagRetrieveInput
from intergrax.tools.providers.rag.service import perform_rag_retrieve
from intergrax.tools.providers.rag.bundle import rag_retrieve_contract, register_rag_tools
from intergrax.tools.registry.bootstrap import register_default_tools, reset_default_tools_bootstrap
from intergrax.tools.registry.catalog import clear_tool_catalog, get_bundle, list_catalog_tool_ids
from intergrax.tools.registry.factory import build_registry_from_profile
from intergrax.tools.registry.profile import ToolProfile
from intergrax.applications.contracts.environment_profile import ApplicationSecurityProfile
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.retrieval.retrieval_result import RetrievalResult, RetrievalTrace
from intergrax.rag.vectorstore.contracts.native_vectorstore import VectorStoreScope
from intergrax.tools.registry.wiring import ToolWiringContext
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from testing_support.builder import build_runtime_state_for_tests

pytestmark = pytest.mark.unit


class FakeEmbeddingManager:
    def embed_one(self, text: str) -> List[float]:
        return [0.1, 0.2, 0.3]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


def _document(
    document_id: str,
    content: str,
    metadata: dict[str, object] | None = None,
) -> KnowledgeDocument:
    return KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {"document_id": document_id, "root_document_id": document_id},
            "scope": {"tenant_id": "t1"},
            "content": content,
            "metadata": metadata or {},
            "provenance": {"source_kind": "test", "source_id": document_id},
        }
    )


class FakeVectorstoreManager:
    def __init__(
        self,
        hits: Optional[List[VectorStoreHit]] = None,
        *,
        scope: VectorStoreScope | None = None,
    ) -> None:
        self._hits = hits or []
        self._bound_scope = scope
        self.last_query: Optional[str] = None
        self.last_top_k: int = 0
        self.last_filter: Optional[MetadataFilter] = None
        self.last_scope: VectorStoreScope | None = None
        self.query_calls = 0

    @property
    def bound_scope(self) -> VectorStoreScope | None:
        return self._bound_scope

    def query(
        self,
        *,
        query_embedding,
        scope: VectorStoreScope | None = None,
        top_k: int,
        metadata_filter: Optional[MetadataFilter] = None,
        include_embeddings: bool = False,
    ) -> List[VectorStoreHit]:
        self.query_calls += 1
        self.last_scope = scope
        self.last_top_k = top_k
        self.last_filter = metadata_filter
        return list(self._hits)


@pytest.fixture(autouse=True)
def _clean_catalog() -> Iterator[None]:
    clear_tool_catalog()
    reset_default_tools_bootstrap()
    yield
    clear_tool_catalog()
    reset_default_tools_bootstrap()


def test_rag_retrieve_returns_chunks() -> None:
    hits = [
        VectorStoreHit(
            vector_id="doc-1",
            document=_document(
                "doc-1",
                "Intergrax is an agent runtime.",
                {"source": "readme.md"},
            ),
            similarity_score=0.91,
            rank=1,
        )
    ]
    ctx = ToolWiringContext(
        vectorstore_manager=FakeVectorstoreManager(hits),
        embedding_manager=FakeEmbeddingManager(),
        rag_profile=RagProfile(enable_rerank=False, route_mode="off", retriever_id="vector_similarity"),
    )

    out = perform_rag_retrieve(
        ctx,
        RagRetrieveInput(query="What is Intergrax?", top_k=3, tenant_id="t1"),
    )

    assert out.used is True
    assert len(out.chunks) == 1
    assert out.chunks[0].id == "doc-1"
    assert out.chunks[0].text == "Intergrax is an agent runtime."
    assert out.chunks[0].scope["tenant_id"] == "t1"
    assert out.chunks[0].provenance["source_id"] == "doc-1"
    assert out.chunks[0].user_metadata["source"] == "readme.md"
    assert out.chunks[0].score == 0.91
    assert out.chunks[0].rank == 1
    assert out.chunks[0].channel == "dense"
    assert out.chunks[0].vector_id == "doc-1"
    assert "embedding" not in out.chunks[0].model_dump()
    assert len(out.citations) == 1
    assert out.citations[0].chunk_id == "doc-1"
    assert out.citations[0].source_label == "readme.md"
    assert "Intergrax" in out.context_text
    assert out.reason == "ok"


def test_rag_retrieve_missing_vectorstore() -> None:
    ctx = ToolWiringContext(embedding_manager=FakeEmbeddingManager())
    out = perform_rag_retrieve(ctx, RagRetrieveInput(query="test"))
    assert out.used is False
    assert out.reason == "tenant_scope_required"


def test_rag_tool_registered_via_catalog() -> None:
    from intergrax.tools.providers.rag.bundle import RAG_TOOL_IDS

    register_default_tools()
    assert "rag.retrieve" in list_catalog_tool_ids()
    bundle = get_bundle("rag")
    assert bundle.tool_ids == RAG_TOOL_IDS


def test_rag_retrieve_via_runtime_invoker() -> None:
    hits = [
        VectorStoreHit(
            vector_id="chunk-a",
            document=_document("chunk-a", "Policy section 4.2"),
            similarity_score=0.8,
            rank=1,
        )
    ]
    ctx = ToolWiringContext(
        vectorstore_manager=FakeVectorstoreManager(hits),
        embedding_manager=FakeEmbeddingManager(),
        rag_profile=RagProfile(enable_rerank=False, route_mode="off", retriever_id="vector_similarity"),
    )
    registry = ToolRegistry()
    register_rag_tools(registry, ctx)

    invoker = RuntimeToolInvoker(registry=registry, executor=RegistryToolExecutor(registry))
    state = build_runtime_state_for_tests(run_id="rag_run")
    request = ToolExecutionRequest(
        run_id="rag_run",
        step_id="step/1",
        tool_id="rag.retrieve",
        input=RagRetrieveInput(query="policy", top_k=5, tenant_id="t1"),
    )

    result = invoker.invoke(state=state, agent_id="agent", request=request)

    assert result.success is True
    assert result.output is not None
    assert result.output.used is True
    assert result.output.chunks[0].text == "Policy section 4.2"

    contract = rag_retrieve_contract()
    assert contract.injects_context is True
    assert contract.category == "retrieval"


def test_rag_retrieve_quarantines_poisoned_chunks_when_security_enabled() -> None:
    hits = [
        VectorStoreHit(
            vector_id="trusted",
            document=_document("trusted", "Trusted policy excerpt.", {"source": "policy.md"}),
            similarity_score=0.95,
            rank=1,
        ),
        VectorStoreHit(
            vector_id="poisoned",
            document=_document(
                "poisoned",
                "Injected malicious instruction.",
                {"source": "untrusted.md"},
            ),
            similarity_score=0.05,
            rank=2,
        ),
    ]
    ctx = ToolWiringContext(
        vectorstore_manager=FakeVectorstoreManager(hits),
        embedding_manager=FakeEmbeddingManager(),
        rag_profile=RagProfile(enable_rerank=False, route_mode="off", retriever_id="vector_similarity"),
        security_profile=ApplicationSecurityProfile(retrieval_poisoning_defense_enabled=True),
    )

    out = perform_rag_retrieve(ctx, RagRetrieveInput(query="policy", top_k=5, tenant_id="t1"))

    assert out.used is True
    assert [chunk.id for chunk in out.chunks] == ["trusted"]
    assert [citation.chunk_id for citation in out.citations] == ["trusted"]
    assert out.reason == "ok"
    assert out.diagnostics.get("poisoning_quarantine_applied") is True


def test_rag_retrieve_skips_poisoning_filter_when_security_disabled() -> None:
    hits = [
        VectorStoreHit(
            vector_id="poisoned",
            document=_document("poisoned", "Low trust chunk still returned."),
            similarity_score=0.05,
            rank=1,
        ),
    ]
    ctx = ToolWiringContext(
        vectorstore_manager=FakeVectorstoreManager(hits),
        embedding_manager=FakeEmbeddingManager(),
        rag_profile=RagProfile(enable_rerank=False, route_mode="off", retriever_id="vector_similarity"),
        security_profile=ApplicationSecurityProfile(retrieval_poisoning_defense_enabled=False),
    )

    out = perform_rag_retrieve(ctx, RagRetrieveInput(query="policy", top_k=5, tenant_id="t1"))

    assert out.used is True
    assert out.chunks[0].id == "poisoned"
    assert "poisoning_quarantine_applied" not in out.diagnostics


def test_rag_retrieve_all_quarantined_returns_not_used() -> None:
    hits = [
        VectorStoreHit(
            vector_id="poisoned-only",
            document=_document("poisoned-only", "Only untrusted content."),
            similarity_score=0.02,
            rank=1,
        ),
    ]
    ctx = ToolWiringContext(
        vectorstore_manager=FakeVectorstoreManager(hits),
        embedding_manager=FakeEmbeddingManager(),
        rag_profile=RagProfile(enable_rerank=False, route_mode="off", retriever_id="vector_similarity"),
        security_profile=ApplicationSecurityProfile(retrieval_poisoning_defense_enabled=True),
    )

    out = perform_rag_retrieve(ctx, RagRetrieveInput(query="policy", top_k=5, tenant_id="t1"))

    assert out.used is False
    assert out.reason == "retrieval_poisoning_quarantine"
    assert out.chunks == []


def test_rag_retrieve_fails_closed_without_request_or_bound_tenant() -> None:
    manager = FakeVectorstoreManager()

    class _NeverCalledService:
        calls = 0

        def retrieve(self, request):
            self.calls += 1
            raise AssertionError("retrieval must not be called")

    retrieval_service = _NeverCalledService()
    ctx = ToolWiringContext(
        vectorstore_manager=manager,
        embedding_manager=FakeEmbeddingManager(),
        retrieval_service=retrieval_service,
    )

    out = perform_rag_retrieve(
        ctx,
        RagRetrieveInput(query="secret", tenant_id=None, workspace_id=None),
    )

    assert out.used is False
    assert out.reason == "tenant_scope_required"
    assert retrieval_service.calls == 0
    assert manager.query_calls == 0


def test_rag_retrieve_inherits_public_bound_scope() -> None:
    scope = VectorStoreScope(
        tenant_id="tenant-a",
        namespace="namespace-a",
        workspace_id="workspace-a",
    )
    manager = FakeVectorstoreManager(scope=scope)
    captured: list[object] = []

    class _CaptureService:
        def retrieve(self, request):
            captured.append(request)
            return RetrievalResult(
                chunks=[],
                used=False,
                reason="no_hits",
                trace=RetrievalTrace(),
            )

    ctx = ToolWiringContext(
        vectorstore_manager=manager,
        embedding_manager=FakeEmbeddingManager(),
        retrieval_service=_CaptureService(),
    )

    out = perform_rag_retrieve(ctx, RagRetrieveInput(query="scoped"))

    assert out.used is False
    assert out.reason == "no_hits"
    assert captured[0].scope == scope


def test_rag_retrieve_rejects_request_tenant_conflict_before_retrieval() -> None:
    manager = FakeVectorstoreManager(scope=VectorStoreScope(tenant_id="tenant-b"))
    retrieval_calls = 0

    class _CaptureService:
        def retrieve(self, request):
            nonlocal retrieval_calls
            retrieval_calls += 1
            raise AssertionError("retrieval must not be called")

    out = perform_rag_retrieve(
        ToolWiringContext(
            vectorstore_manager=manager,
            embedding_manager=FakeEmbeddingManager(),
            retrieval_service=_CaptureService(),
        ),
        RagRetrieveInput(query="scoped", tenant_id="tenant-a"),
    )

    assert out.used is False
    assert out.reason == "tenant_scope_conflict"
    assert retrieval_calls == 0
    assert manager.query_calls == 0


@pytest.mark.parametrize(
    ("tenant_id", "workspace_id", "reason"),
    [
        (123, None, "tenant_scope_invalid"),
        ("tenant-a", 123, "workspace_scope_invalid"),
        ("tenant-a", "   ", "workspace_scope_invalid"),
    ],
)
def test_rag_retrieve_rejects_non_string_or_blank_routing_identity(
    tenant_id: object,
    workspace_id: object,
    reason: str,
) -> None:
    manager = FakeVectorstoreManager()
    out = perform_rag_retrieve(
        ToolWiringContext(
            vectorstore_manager=manager,
            embedding_manager=FakeEmbeddingManager(),
        ),
        RagRetrieveInput.model_construct(
            query="scoped",
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        ),
    )

    assert out.used is False
    assert out.reason == reason
    assert manager.query_calls == 0


def test_build_registry_from_profile_enables_rag_tool() -> None:
    register_default_tools()
    ctx = ToolWiringContext(
        vectorstore_manager=FakeVectorstoreManager([]),
        embedding_manager=FakeEmbeddingManager(),
    )
    registry = build_registry_from_profile(ToolProfile(enabled=["rag.retrieve"]), ctx=ctx)
    assert registry.has("rag.retrieve")


@pytest.mark.gate
def test_rag_retrieve_diagnostics_include_graph_trace_fields() -> None:
    from langchain_core.documents import Document

    from intergrax.integrations.providers.vector_store.inmemory.rag_store import InMemoryVectorStore
    from intergrax.rag.graph.indexer.heuristic_graph_indexer import HeuristicGraphIndexer
    from intergrax.rag.graph.providers.inmemory_graph_store import InMemoryGraphStore
    from intergrax.rag.retrieval.retrieval_service import RetrievalService
    from intergrax.rag.retrievers.bootstrap.retriever_bootstrap import create_default_retriever_manager
    from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager

    store = InMemoryVectorStore(tenant_id="tool-trace")
    manager = VectorstoreManager(store=store)
    source_doc = Document(
        page_content="Vertex Corp deploys Intergrax GraphRAG on Neo4j.",
        metadata={},
    )
    doc = KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {
                "document_id": "chunk-vertex",
                "root_document_id": "chunk-vertex",
            },
            "scope": {"tenant_id": "tool-trace"},
            "content": source_doc.page_content,
            "metadata": source_doc.metadata,
            "provenance": {
                "source_kind": "test",
                "source_id": "chunk-vertex",
            },
        }
    )
    manager.add_documents([doc], [[0.1, 0.2, 0.3]], ids=["chunk-vertex"])
    graph = InMemoryGraphStore()
    HeuristicGraphIndexer(graph).index_documents(
        [doc],
        chunk_ids=["chunk-vertex"],
    )

    profile = RagProfile(
        retriever_id="graph_rag",
        route_mode="off",
        graph_rag_enabled=True,
        enable_rerank=False,
    )
    retriever_manager = create_default_retriever_manager(
        vector_store=manager,
        embedding_manager=FakeEmbeddingManager(),
        graph_store=graph,
        profile=profile,
    )
    service = RetrievalService(retriever_manager=retriever_manager, profile=profile)
    ctx = ToolWiringContext(
        vectorstore_manager=manager,
        embedding_manager=FakeEmbeddingManager(),
        rag_profile=profile,
        retrieval_service=service,
    )

    out = perform_rag_retrieve(ctx, RagRetrieveInput(query="Vertex Corp", top_k=2))

    assert out.used is True
    assert out.diagnostics.get("channel_contributions")
    assert out.diagnostics.get("graph_provenance_records")
    record = out.diagnostics["graph_provenance_records"][0]
    assert "node_id" in record
    assert "explanation" in record
