# © Artur Czarnecki. All rights reserved.

"""CE-PROV-BRIDGE / B1: legacy bridge + graph core builtin providers."""

from __future__ import annotations

import pytest

from intergrax.context.bootstrap import bootstrap_context_catalog, reset_context_catalog_bootstrap_for_tests
from intergrax.context.bootstrap import materialize_context_plugin_registry
from intergrax.context.contracts import (
    ContextAssemblyRequest,
    ContextBudgetSnapshot,
    ContextDecisionSnapshot,
    ContextFragmentSource,
    ContextProviderContext,
)
from intergrax.context.providers.legacy_bridge import (
    SESSION_HISTORY_MESSAGES_HANDLE,
    fragments_from_attachment_summaries,
    fragments_from_ltm_entries,
    fragments_from_policy_overlay_fragments,
    fragments_from_prior_output_records,
    fragments_from_rag_chunks,
    fragments_from_session_history,
    fragments_from_shared_context_reads,
    fragments_from_system_instructions,
    fragments_from_task_message,
    fragments_from_tool_output_blocks,
    fragments_from_websearch_blocks,
)
from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.context.context_engine import DefaultNexusContextEngine
from intergrax.runtime.nexus.context.context_manager import ContextManager
from intergrax.runtime.nexus.context.context_models import ContextProvenance, ContextSourceType, PriorOutputRecord
from intergrax.runtime.nexus.context.provider_handles import (
    RAG_CHUNKS_METADATA_KEY,
    SESSION_HISTORY_MESSAGES_METADATA_KEY,
    build_graph_provider_handles,
)
from intergrax.runtime.nexus.execution.execution_graph import ExecutionNode
from intergrax.runtime.task.task import Task, TaskContext

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _WindowAdapter(LLMAdapter):
    provider = "fake"
    model = "fake"

    @property
    def context_window_tokens(self) -> int:
        return 4096

    def generate_messages(self, messages, **kwargs) -> LLMAdapterResponse:
        _ = messages, kwargs
        return LLMAdapterResponse(content="ok")


@pytest.fixture(autouse=True)
def _catalog() -> None:
    reset_context_catalog_bootstrap_for_tests()
    bootstrap_context_catalog(register_shipped=True, discover_entry_points=False)
    yield
    reset_context_catalog_bootstrap_for_tests()


def _assembly_request(*, objective: str = "do work") -> ContextAssemblyRequest:
    return ContextAssemblyRequest(
        trace_id="trace-1",
        run_id="run-1",
        task_id="task-1",
        tenant_id="tenant-1",
        assembly_scope="graph_node",
        objective=objective,
        decision_profile=ContextDecisionSnapshot(),
        budget_policy=ContextBudgetSnapshot(max_chars=8000),
        assembly_options=TaskContextAssemblyOptions(),
        graph_node_id="n1",
        step_kind="cap.test",
    )


def test_fragments_from_task_message_uses_objective() -> None:
    fragments = fragments_from_task_message(_assembly_request(objective="ship feature"))
    assert len(fragments) == 1
    assert fragments[0].source == ContextFragmentSource.TASK_MESSAGE
    assert "ship feature" in fragments[0].content


def test_fragments_from_prior_output_records() -> None:
    record = PriorOutputRecord(
        node_id="dep-1",
        agent_id="agent-a",
        summary="summary text",
        evidence="prior evidence block",
        provenance=ContextProvenance(
            source_type=ContextSourceType.DEPENDENCY_OUTPUT,
            source_id="dep-1",
            agent_id="agent-a",
        ),
    )
    fragments = fragments_from_prior_output_records([record])
    assert len(fragments) == 1
    assert fragments[0].source == ContextFragmentSource.GRAPH_PRIOR
    assert fragments[0].content == "prior evidence block"


def test_fragments_from_session_history_respects_limit() -> None:
    messages = [
        ChatMessage(role="user", content=f"turn-{index}")
        for index in range(10)
    ]
    fragments = fragments_from_session_history(messages, max_entries=3)
    assert len(fragments) == 3
    assert fragments[0].source == ContextFragmentSource.SESSION_HISTORY
    assert "turn-7" in fragments[0].content


@pytest.mark.asyncio
async def test_graph_engine_path_includes_core_provider_fragments() -> None:
    bus = RuntimeEventBus(record_history=True)
    adapter = _WindowAdapter()
    registry = materialize_context_plugin_registry(["intergrax.builtin"])
    engine = DefaultNexusContextEngine(engine_id="default", registry=registry)
    manager = ContextManager(event_bus=bus, context_engine=engine, llm_adapter=adapter)

    prior = {
        "dep-1": AgentExecutionResult(
            agent_id="agent-a",
            run_id="run-dep",
            status=AgentExecutionStatus.COMPLETED,
            summary="dependency summary",
        )
    }
    task = Task(
        tenant_id="t1",
        user_id="u1",
        session_id="sess-1",
        message="main task",
        context=TaskContext(),
        metadata={
            SESSION_HISTORY_MESSAGES_METADATA_KEY: [
                ChatMessage(role="user", content="earlier question", entry_id="hist-u1"),
                ChatMessage(role="assistant", content="earlier answer", entry_id="hist-a1"),
            ]
        },
    )
    node = ExecutionNode(
        node_id="n1",
        agent_id="worker",
        capability="cap.test",
        depends_on=["dep-1"],
    )

    bundle = await manager.build_agent_context_async(task, node, prior)

    assert "[context:task_message:" in bundle.message
    assert "[context:graph_prior:dep-1]" in bundle.message
    assert "dependency summary" in bundle.message
    assert "[context:session_history:" in bundle.message
    assert "earlier question" in bundle.message
    assert bundle.metadata.get("engine_id") == "default"


@pytest.mark.asyncio
async def test_builtin_collectors_read_provider_handles() -> None:
    registry = materialize_context_plugin_registry(["intergrax.builtin"])
    providers = {provider.provider_id: provider for provider in registry.list_providers()}
    request = _assembly_request(objective="handle task")
    runtime_config = RuntimeConfig(llm_adapter=_WindowAdapter(), production_mode=False)
    task = Task(
        tenant_id="t1",
        user_id="u1",
        session_id="sess-2",
        message="handle task",
        context=TaskContext(),
    )
    record = PriorOutputRecord(
        node_id="dep-2",
        agent_id="agent-b",
        summary="from handle",
        provenance=ContextProvenance(
            source_type=ContextSourceType.DEPENDENCY_OUTPUT,
            source_id="dep-2",
        ),
    )
    handles = build_graph_provider_handles(
        task,
        runtime_config=runtime_config,
        messages=[ChatMessage(role="user", content="handle task")],
        event_bus=None,
        node_id="n1",
        agent_id="worker",
        engine_id="default",
        prior_output_records=[record],
        session_history_messages=[
            ChatMessage(role="user", content="history turn", entry_id="hist-1"),
        ],
    )
    ctx = ContextProviderContext(engine_id="default", handles=handles)

    task_frags = await providers["builtin.task_message"].collect(request, ctx)
    prior_frags = await providers["builtin.graph_prior"].collect(request, ctx)
    history_frags = await providers["builtin.session_history"].collect(request, ctx)

    assert task_frags and task_frags[0].source == ContextFragmentSource.TASK_MESSAGE
    assert prior_frags and prior_frags[0].source == ContextFragmentSource.GRAPH_PRIOR
    assert history_frags and history_frags[0].source == ContextFragmentSource.SESSION_HISTORY


def test_fragments_from_rag_chunks_emits_citations() -> None:
    chunks = [
        {"text": "retrieved doc", "metadata": {"doc_id": "d1", "source": "manual.pdf", "page": 3}},
    ]
    fragments = fragments_from_rag_chunks(chunks)
    assert len(fragments) == 1
    assert fragments[0].source == ContextFragmentSource.RAG
    assert fragments[0].metadata.get("citations")


def test_fragments_from_ltm_entries_skips_deleted() -> None:
    entries = [{"entry_id": "e1", "content": "fact", "deleted": True}, {"entry_id": "e2", "content": "ok"}]
    fragments = fragments_from_ltm_entries(entries)
    assert len(fragments) == 1
    assert fragments[0].source == ContextFragmentSource.LONGTERM_MEMORY


def test_fragments_from_websearch_and_tool_output() -> None:
    web = fragments_from_websearch_blocks(["result snippet", {"content": "block", "url": "https://x"}])
    tools = fragments_from_tool_output_blocks([{"content": "tool ok", "tool_call_id": "tc1"}])
    assert web and web[0].source == ContextFragmentSource.WEBSEARCH
    assert tools and tools[0].source == ContextFragmentSource.TOOL_OUTPUT


def test_fragments_from_system_policy_shared_attachment() -> None:
    system = fragments_from_system_instructions("Be concise.")
    policy = fragments_from_policy_overlay_fragments(
        [{"overlay_id": "o2", "content": "second", "priority": 200}, {"overlay_id": "o1", "content": "first", "priority": 50}]
    )
    shared = fragments_from_shared_context_reads({"dep-1": {"summary": "shared payload"}})
    attachments = fragments_from_attachment_summaries([{"attachment_id": "a1", "summary": "image desc"}])
    assert system[0].mandatory is True
    assert policy[0].source_id == "o1"
    assert shared[0].source == ContextFragmentSource.SHARED_CONTEXT
    assert attachments[0].source == ContextFragmentSource.ATTACHMENT


@pytest.mark.asyncio
async def test_builtin_collectors_read_extended_handles() -> None:
    registry = materialize_context_plugin_registry(["intergrax.builtin"])
    providers = {provider.provider_id: provider for provider in registry.list_providers()}
    request = _assembly_request(objective="handle task")
    runtime_config = RuntimeConfig(llm_adapter=_WindowAdapter(), production_mode=False)
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="handle task",
        context=TaskContext(),
        metadata={
            RAG_CHUNKS_METADATA_KEY: [{"text": "rag hit", "metadata": {"doc_id": "doc-1"}}],
            "ltm_entries": [{"entry_id": "ltm-1", "content": "memory fact"}],
            "websearch_blocks": ["search hit"],
            "tool_output_blocks": [{"content": "tool result", "tool_call_id": "t1"}],
            "system_instructions": "Follow policy.",
            "policy_overlay_fragments": [{"overlay_id": "p1", "content": "overlay"}],
            "attachment_summaries": [{"attachment_id": "att-1", "summary": "chart summary"}],
        },
    )
    handles = build_graph_provider_handles(
        task,
        runtime_config=runtime_config,
        messages=[ChatMessage(role="user", content="handle task")],
        event_bus=None,
        node_id="n1",
        agent_id="worker",
        engine_id="default",
        shared_context_reads={"dep-x": {"summary": "shared"}},
    )
    ctx = ContextProviderContext(engine_id="default", handles=handles)

    rag_frags = await providers["builtin.rag"].collect(request, ctx)
    ltm_frags = await providers["builtin.longterm_memory"].collect(request, ctx)
    web_frags = await providers["builtin.websearch"].collect(request, ctx)
    tool_frags = await providers["builtin.tool_output"].collect(request, ctx)
    sys_frags = await providers["builtin.system_instructions"].collect(request, ctx)
    shared_frags = await providers["builtin.shared_context"].collect(request, ctx)
    policy_frags = await providers["builtin.policy_overlay"].collect(request, ctx)
    attach_frags = await providers["builtin.attachments"].collect(request, ctx)

    assert rag_frags and rag_frags[0].source == ContextFragmentSource.RAG
    assert ltm_frags and ltm_frags[0].source == ContextFragmentSource.LONGTERM_MEMORY
    assert web_frags and web_frags[0].source == ContextFragmentSource.WEBSEARCH
    assert tool_frags and tool_frags[0].source == ContextFragmentSource.TOOL_OUTPUT
    assert sys_frags and sys_frags[0].source == ContextFragmentSource.SYSTEM_INSTRUCTIONS
    assert shared_frags and shared_frags[0].source == ContextFragmentSource.SHARED_CONTEXT
    assert policy_frags and policy_frags[0].source == ContextFragmentSource.POLICY_OVERLAY
    assert attach_frags and attach_frags[0].source == ContextFragmentSource.ATTACHMENT


@pytest.mark.asyncio
async def test_missing_stable_revision_preserves_raw_legacy_history_through_builtin_session_history() -> None:
    registry = materialize_context_plugin_registry(["intergrax.builtin"])
    providers = {provider.provider_id: provider for provider in registry.list_providers()}
    request = _assembly_request(objective="legacy task")
    ctx = ContextProviderContext(
        engine_id="default",
        handles={
            SESSION_HISTORY_MESSAGES_HANDLE: [
                ChatMessage(role="user", content="legacy turn", entry_id="legacy-u1"),
                ChatMessage(role="assistant", content="legacy answer", entry_id="legacy-a1"),
            ],
        },
    )
    fragments = await providers["builtin.session_history"].collect(request, ctx)
    assert len(fragments) == 2
    assert fragments[0].source == ContextFragmentSource.SESSION_HISTORY
    assert "legacy turn" in fragments[0].content


@pytest.mark.asyncio
async def test_legacy_fallback_still_applies_max_memory_entries_in_context() -> None:
    registry = materialize_context_plugin_registry(["intergrax.builtin"])
    providers = {provider.provider_id: provider for provider in registry.list_providers()}
    request = ContextAssemblyRequest(
        trace_id="trace-1",
        run_id="run-1",
        task_id="task-1",
        tenant_id="tenant-1",
        assembly_scope="graph_node",
        objective="legacy task",
        decision_profile=ContextDecisionSnapshot(max_memory_entries_in_context=2),
        budget_policy=ContextBudgetSnapshot(max_chars=8000),
        assembly_options=TaskContextAssemblyOptions(),
        graph_node_id="n1",
        step_kind="cap.test",
    )
    messages = [
        ChatMessage(role="user", content=f"turn-{index}", entry_id=f"m{index}")
        for index in range(6)
    ]
    ctx = ContextProviderContext(
        engine_id="default",
        handles={SESSION_HISTORY_MESSAGES_HANDLE: messages},
    )
    fragments = await providers["builtin.session_history"].collect(request, ctx)
    assert len(fragments) == 2
    assert fragments_from_session_history(messages, max_entries=2) == fragments
