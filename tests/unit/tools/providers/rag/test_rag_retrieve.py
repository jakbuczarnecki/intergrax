# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Any, List, Optional

import pytest

from intergrax.rag.vectorstore.contracts.vector_store import MetadataFilter, VectorStoreHit
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.providers.rag.contracts import RagRetrieveInput
from intergrax.tools.providers.rag.handler import RagRetrieveHandler
from intergrax.tools.providers.rag.service import perform_rag_retrieve
from intergrax.tools.providers.rag.bundle import rag_retrieve_contract, register_rag_tools
from intergrax.tools.registry.bootstrap import register_default_tools, reset_default_tools_bootstrap
from intergrax.tools.registry.catalog import clear_tool_catalog, get_bundle, list_catalog_tool_ids
from intergrax.tools.registry.factory import build_registry_from_profile
from intergrax.tools.registry.profile import ToolProfile
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


class FakeVectorstoreManager:
    def __init__(self, hits: Optional[List[VectorStoreHit]] = None) -> None:
        self._hits = hits or []
        self.last_query: Optional[str] = None
        self.last_top_k: int = 0
        self.last_filter: Optional[MetadataFilter] = None

    def query(
        self,
        *,
        query_embedding,
        top_k: int,
        metadata_filter: Optional[MetadataFilter] = None,
        include_embeddings: bool = False,
    ) -> List[VectorStoreHit]:
        self.last_top_k = top_k
        self.last_filter = metadata_filter
        return list(self._hits)


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_tool_catalog()
    reset_default_tools_bootstrap()
    yield
    clear_tool_catalog()
    reset_default_tools_bootstrap()


def test_rag_retrieve_returns_chunks() -> None:
    hits = [
        VectorStoreHit(
            id="doc-1",
            content="Intergrax is an agent runtime.",
            metadata={"source": "readme.md"},
            similarity_score=0.91,
            rank=1,
        )
    ]
    ctx = ToolWiringContext(
        vectorstore_manager=FakeVectorstoreManager(hits),
        embedding_manager=FakeEmbeddingManager(),
    )

    out = perform_rag_retrieve(
        ctx,
        RagRetrieveInput(query="What is Intergrax?", top_k=3, tenant_id="t1"),
    )

    assert out.used is True
    assert len(out.chunks) == 1
    assert out.chunks[0].id == "doc-1"
    assert "Intergrax" in out.context_text
    assert out.reason == "ok"


def test_rag_retrieve_missing_vectorstore() -> None:
    ctx = ToolWiringContext(embedding_manager=FakeEmbeddingManager())
    out = perform_rag_retrieve(ctx, RagRetrieveInput(query="test"))
    assert out.used is False
    assert out.reason == "vectorstore_manager_not_configured"


def test_rag_tool_registered_via_catalog() -> None:
    register_default_tools()
    assert "rag.retrieve" in list_catalog_tool_ids()
    bundle = get_bundle("rag")
    assert bundle.tool_ids == ("rag.retrieve",)


def test_rag_retrieve_via_runtime_invoker() -> None:
    hits = [
        VectorStoreHit(
            id="chunk-a",
            content="Policy section 4.2",
            metadata={},
            similarity_score=0.8,
            rank=1,
        )
    ]
    ctx = ToolWiringContext(
        vectorstore_manager=FakeVectorstoreManager(hits),
        embedding_manager=FakeEmbeddingManager(),
    )
    registry = ToolRegistry()
    register_rag_tools(registry, ctx)

    invoker = RuntimeToolInvoker(registry=registry, executor=RegistryToolExecutor(registry))
    state = build_runtime_state_for_tests(run_id="rag_run")
    request = ToolExecutionRequest(
        run_id="rag_run",
        step_id="step/1",
        tool_id="rag.retrieve",
        input=RagRetrieveInput(query="policy", top_k=5),
    )

    result = invoker.invoke(state=state, agent_id="agent", request=request)

    assert result.success is True
    assert result.output is not None
    assert result.output.used is True
    assert result.output.chunks[0].text == "Policy section 4.2"

    contract = rag_retrieve_contract()
    assert contract.injects_context is True
    assert contract.category == "retrieval"


def test_build_registry_from_profile_enables_rag_tool() -> None:
    register_default_tools()
    ctx = ToolWiringContext(
        vectorstore_manager=FakeVectorstoreManager([]),
        embedding_manager=FakeEmbeddingManager(),
    )
    registry = build_registry_from_profile(ToolProfile(enabled=["rag.retrieve"]), ctx=ctx)
    assert registry.has("rag.retrieve")
