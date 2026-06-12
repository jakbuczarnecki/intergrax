# © Artur Czarnecki. All rights reserved.

"""CE S6–S12 module gates."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.context_presets import (
    codebase_context_profile,
    explore_child_context_profile,
    production_context_profile,
)
from intergrax.context.contracts import (
    ContextAssemblyRequest,
    ContextBudgetSnapshot,
    ContextDecisionSnapshot,
    ContextFragmentSource,
    ContextProviderContext,
)
from intergrax.context.dedup import dedup_fragments_by_hash
from intergrax.context.orchestrator import ContextOrchestrator, ContextOrchestratorConfig
from intergrax.context.providers.session_semantic_recall import SessionSemanticRecallProvider
from intergrax.context.providers.workspace import WorkspaceContextProvider
from intergrax.context.providers.workspace_index import build_workspace_index
from intergrax.context.tracking.context_spans import CE_OTEL_SPAN_NAMES, context_span
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions
from intergrax.runtime.nexus.context.codebase_engine import CodebaseContextEngine

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.mark.asyncio
async def test_session_semantic_recall_when_enabled() -> None:
    provider = SessionSemanticRecallProvider()
    request = ContextAssemblyRequest(
        trace_id="t",
        run_id="r",
        task_id="task",
        tenant_id="tenant",
        assembly_scope="acp_step",
        objective="find prior discussion",
        decision_profile=ContextDecisionSnapshot(),
        budget_policy=ContextBudgetSnapshot(),
        assembly_options=TaskContextAssemblyOptions(),
    )
    ctx = ContextProviderContext(
        handles={
            "enable_session_vector_index": True,
            "session_vector_hits": [{"text": "prior answer", "score": 0.9}],
        }
    )
    fragments = await provider.collect(request, ctx)
    assert fragments
    assert fragments[0].source == ContextFragmentSource.SESSION_HISTORY_SEMANTIC


def test_workspace_index_and_provider_chunks() -> None:
    files = {f"f{i}.py": f"print({i})\n" for i in range(1000)}
    index = build_workspace_index(files)
    assert index.root_merkle
    assert len(index.chunks) == 1000


@pytest.mark.asyncio
async def test_one_k_workspace_assemble_stays_under_budget() -> None:
    from intergrax.llm.messages import ChatMessage
    from intergrax.runtime.nexus.config import RuntimeConfig
    from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
    from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
    from intergrax.context.bootstrap import materialize_context_plugin_registry
    from intergrax.runtime.nexus.context.context_engine import DefaultNexusContextEngine

    class _WindowAdapter(LLMAdapter):
        provider = "fake"
        model = "fake-1k"

        @property
        def context_window_tokens(self) -> int:
            return 2048

        def generate_messages(self, messages, **kwargs) -> LLMAdapterResponse:
            _ = messages, kwargs
            return LLMAdapterResponse(content="ok")

    files = {f"f{i}.py": f"print({i})\n" for i in range(1000)}
    registry = materialize_context_plugin_registry(["intergrax.builtin"])
    engine = CodebaseContextEngine(registry=registry)
    adapter = _WindowAdapter()
    config = RuntimeConfig(llm_adapter=adapter, production_mode=False)
    request = ContextAssemblyRequest(
        trace_id="t",
        run_id="r",
        task_id="task",
        tenant_id="tenant",
        assembly_scope="graph_node",
        objective="review codebase",
        decision_profile=ContextDecisionSnapshot(),
        budget_policy=ContextBudgetSnapshot(max_tokens_estimate=512),
        assembly_options=TaskContextAssemblyOptions(),
    )
    ctx = ContextProviderContext(
        handles={
            "runtime_config": config,
            "messages": [ChatMessage(role="user", content="summarize")],
            "workspace_files": files,
            "workspace_max_chunks": 8,
        }
    )
    assembled = await engine.assemble(request, provider_ctx=ctx)
    assert assembled.total_tokens <= assembled.budget_tokens
    assert assembled.fragments_included
    assert any("f" in (msg.content or "") for msg in assembled.messages)


@pytest.mark.asyncio
async def test_workspace_provider_respects_max_chunks() -> None:
    provider = WorkspaceContextProvider()
    files = {f"f{i}.py": "x\n" for i in range(50)}
    request = ContextAssemblyRequest(
        trace_id="t",
        run_id="r",
        task_id="task",
        tenant_id="tenant",
        assembly_scope="graph_node",
        objective="code",
        decision_profile=ContextDecisionSnapshot(),
        budget_policy=ContextBudgetSnapshot(),
        assembly_options=TaskContextAssemblyOptions(),
    )
    ctx = ContextProviderContext(handles={"workspace_files": files, "workspace_max_chunks": 8})
    fragments = await provider.collect(request, ctx)
    assert len(fragments) == 8


def test_codebase_engine_id() -> None:
    engine = CodebaseContextEngine()
    assert engine.engine_id == "codebase"


def test_dedup_fragments() -> None:
    from intergrax.context.contracts import ContextFragment

    first = ContextFragment(
        fragment_id="a",
        source=ContextFragmentSource.RAG,
        source_id="1",
        content="same",
        token_estimate=1,
        relevance_score=0.8,
        freshness_score=0.8,
        confidence_score=0.8,
        mandatory=False,
        content_hash="abc",
    )
    second = ContextFragment(
        fragment_id="b",
        source=ContextFragmentSource.RAG,
        source_id="2",
        content="same",
        token_estimate=1,
        relevance_score=0.8,
        freshness_score=0.8,
        confidence_score=0.8,
        mandatory=False,
        content_hash="abc",
    )
    kept, dropped = dedup_fragments_by_hash([first, second])
    assert len(kept) == 1
    assert dropped


def test_context_spans_registry() -> None:
    assert "context.engine.assemble" in CE_OTEL_SPAN_NAMES
    with context_span("context.engine.assemble"):
        pass


def test_context_presets() -> None:
    assert production_context_profile().engine_preset == "default"
    assert codebase_context_profile().engine_preset == "codebase"
    assert explore_child_context_profile().engine_preset == "explore_child"


@pytest.mark.asyncio
async def test_orchestrator_bounded_hops() -> None:
    from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
    from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
    from intergrax.runtime.nexus.config import RuntimeConfig
    from intergrax.runtime.nexus.context.context_engine import DefaultNexusContextEngine

    class _Adapter(LLMAdapter):
        provider = "fake"
        model = "fake"

        @property
        def context_window_tokens(self) -> int:
            return 4096

        def generate_messages(self, messages, **kwargs) -> LLMAdapterResponse:
            _ = messages, kwargs
            return LLMAdapterResponse(content="ok")

    engine = DefaultNexusContextEngine(engine_id="codebase")
    orchestrator = ContextOrchestrator(engine, config=ContextOrchestratorConfig(max_hops=2))
    request = ContextAssemblyRequest(
        trace_id="t",
        run_id="r",
        task_id="task",
        tenant_id="tenant",
        assembly_scope="graph_node",
        objective="obj",
        decision_profile=ContextDecisionSnapshot(),
        budget_policy=ContextBudgetSnapshot(),
        assembly_options=TaskContextAssemblyOptions(),
    )
    config = RuntimeConfig(llm_adapter=_Adapter(), production_mode=False)
    from intergrax.llm.messages import ChatMessage

    ctx = ContextProviderContext(
        handles={
            "runtime_config": config,
            "messages": [ChatMessage(role="user", content="hello")],
        },
    )
    assembled = await orchestrator.assemble_with_hops(request, provider_ctx=ctx)
    assert assembled.budget_tokens >= 0
