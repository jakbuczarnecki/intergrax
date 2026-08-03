# © Artur Czarnecki. All rights reserved.

"""CE-PROV-BRIDGE / B1: legacy bridge + graph core builtin providers."""

from __future__ import annotations

import inspect

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
from intergrax.context.session_history import (
    SESSION_HISTORY_SNAPSHOT_HANDLE,
    SESSION_HISTORY_SNAPSHOT_REQUIRED_REASON,
    SessionHistorySnapshotRequiredError,
    build_session_history_snapshot,
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
    SESSION_CONTEXT_REVISION_METADATA_KEY,
    SESSION_HISTORY_MESSAGES_METADATA_KEY,
    SESSION_HISTORY_SNAPSHOT_METADATA_KEY,
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


def test_legacy_session_history_fragment_builder_fails_closed() -> None:
    messages = [
        ChatMessage(role="user", content=f"turn-{index}", entry_id=f"m{index}")
        for index in range(3)
    ]
    with pytest.raises(SessionHistorySnapshotRequiredError) as exc_info:
        fragments_from_session_history(messages)
    assert str(exc_info.value) == SESSION_HISTORY_SNAPSHOT_REQUIRED_REASON
    assert fragments_from_session_history([], include_session_history=False) == []
    assert fragments_from_session_history([], include_session_history=True) == []


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
            ],
            SESSION_CONTEXT_REVISION_METADATA_KEY: "rev-graph-1",
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
    assert "user: earlier question" not in bundle.message
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
        metadata={SESSION_CONTEXT_REVISION_METADATA_KEY: "rev-handle-1"},
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
async def test_raw_legacy_handle_requires_snapshot_via_builtin() -> None:
    registry = materialize_context_plugin_registry(["intergrax.builtin"])
    providers = {provider.provider_id: provider for provider in registry.list_providers()}
    request = _assembly_request(objective="legacy task")
    ctx = ContextProviderContext(
        engine_id="default",
        handles={
            SESSION_HISTORY_MESSAGES_HANDLE: [
                ChatMessage(role="user", content="legacy turn", entry_id="legacy-u1"),
            ],
        },
    )
    with pytest.raises(SessionHistorySnapshotRequiredError):
        await providers["builtin.session_history"].collect(request, ctx)


def test_graph_handles_build_canonical_snapshot() -> None:
    messages = [
        ChatMessage(
            role="user",
            content="question",
            entry_id="m1",
            tool_calls=[{"id": "tc1", "type": "function", "function": {"name": "search", "arguments": "{}"}}],
        ),
        ChatMessage(role="tool", content="result", entry_id="m2", tool_call_id="tc1", name="search"),
    ]
    task = Task(
        tenant_id="t1",
        user_id="u1",
        session_id="sess-snap",
        message="task",
        context=TaskContext(),
        metadata={SESSION_CONTEXT_REVISION_METADATA_KEY: "rev-1"},
    )
    handles = build_graph_provider_handles(
        task,
        runtime_config=RuntimeConfig(llm_adapter=_WindowAdapter(), production_mode=False),
        messages=[ChatMessage(role="user", content="task")],
        event_bus=None,
        node_id="n1",
        agent_id="worker",
        engine_id="default",
        session_history_messages=messages,
    )
    assert SESSION_HISTORY_SNAPSHOT_HANDLE in handles
    assert SESSION_HISTORY_MESSAGES_HANDLE not in handles
    snapshot = handles[SESSION_HISTORY_SNAPSHOT_HANDLE]
    assert len(snapshot.messages) == 2
    assert snapshot.messages[0].message_id == "m1"
    assert snapshot.messages[1].tool_call_id == "tc1"


def test_graph_handles_reject_raw_history_without_revision() -> None:
    task = Task(
        tenant_id="t1",
        user_id="u1",
        session_id="sess-no-rev",
        message="task",
        context=TaskContext(),
    )
    with pytest.raises(SessionHistorySnapshotRequiredError):
        build_graph_provider_handles(
            task,
            runtime_config=RuntimeConfig(llm_adapter=_WindowAdapter(), production_mode=False),
            messages=[ChatMessage(role="user", content="task")],
            event_bus=None,
            node_id="n1",
            agent_id="worker",
            engine_id="default",
            session_history_messages=[
                ChatMessage(role="user", content="orphan", entry_id="m1"),
            ],
        )


def test_graph_handles_accept_direct_snapshot() -> None:
    snapshot = build_session_history_snapshot(
        tenant_id="t1",
        context_scope_id="scope",
        revision_id="rev-direct",
        messages=[ChatMessage(role="user", content="direct", entry_id="m1")],
    )
    task = Task(
        tenant_id="t1",
        user_id="u1",
        session_id="sess-direct",
        message="task",
        context=TaskContext(),
        metadata={SESSION_HISTORY_SNAPSHOT_METADATA_KEY: snapshot},
    )
    handles = build_graph_provider_handles(
        task,
        runtime_config=RuntimeConfig(llm_adapter=_WindowAdapter(), production_mode=False),
        messages=[ChatMessage(role="user", content="task")],
        event_bus=None,
        node_id="n1",
        agent_id="worker",
        engine_id="default",
        session_history_messages=[
            ChatMessage(role="user", content="ignored", entry_id="ignored"),
        ],
    )
    assert handles[SESSION_HISTORY_SNAPSHOT_HANDLE] is snapshot


def test_structural_guards_session_history_migration() -> None:
    from intergrax.context.providers import builtin as builtin_mod
    from intergrax.context.providers import legacy_bridge as legacy_mod
    from intergrax.runtime.nexus.context import provider_handles as handles_mod
    from intergrax.runtime.nexus.context import runtime_state_handle_bridge as bridge_mod
    from intergrax.runtime.nexus.context import uaep_assemble as uaep_mod

    builtin_source = inspect.getsource(builtin_mod._collect_session_history)
    assert "fragments_from_session_history(" not in builtin_source
    assert "max_memory_entries_in_context" not in builtin_source

    legacy_source = inspect.getsource(legacy_mod.fragments_from_session_history)
    assert "[-max_entries:]" not in legacy_source
    assert 'content=f"{role}: {text}"' not in legacy_source
    assert "SessionHistorySnapshotRequiredError" in legacy_source

    handles_source = inspect.getsource(handles_mod.build_graph_provider_handles)
    assert "SESSION_HISTORY_MESSAGES_HANDLE]" not in handles_source

    bridge_source = inspect.getsource(bridge_mod.extract_provider_metadata_from_runtime_state)
    assert "SESSION_HISTORY_MESSAGES_METADATA_KEY" not in bridge_source

    uaep_source = inspect.getsource(uaep_mod._task_stub_from_request)
    assert "session_history_snapshot" in uaep_source
    assert "session_context_revision_id" in uaep_source
