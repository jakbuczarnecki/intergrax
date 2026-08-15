# © Artur Czarnecki. All rights reserved.

"""CE-HANDLE-FILL: RuntimeState → provider metadata bridge."""

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
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.context.context_builder import BuiltContext, RetrievedChunk
from intergrax.context.session_history import (
    SESSION_HISTORY_SNAPSHOT_HANDLE,
    SESSION_HISTORY_SNAPSHOT_REQUIRED_REASON,
    SessionHistorySnapshotRequiredError,
)
from intergrax.runtime.nexus.context.provider_handles import (
    RAG_CHUNKS_METADATA_KEY,
    SESSION_HISTORY_MESSAGES_METADATA_KEY,
    build_graph_provider_handles,
)
from intergrax.runtime.nexus.context.runtime_state_handle_bridge import (
    extract_provider_metadata_from_runtime_state,
    merge_provider_metadata_into_request,
)
from unittest.mock import MagicMock

from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.task.task import Task, TaskContext

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.fixture(autouse=True)
def _catalog() -> None:
    reset_context_catalog_bootstrap_for_tests()
    bootstrap_context_catalog(register_shipped=True, discover_entry_points=False)
    yield
    reset_context_catalog_bootstrap_for_tests()


def _runtime_state(**metadata: object) -> RuntimeState:
    from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
    from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse

    class _Adapter(LLMAdapter):
        provider = "fake"
        model = "fake"

        @property
        def context_window_tokens(self) -> int:
            return 4096

        def generate_messages(self, messages, **kwargs) -> LLMAdapterResponse:
            _ = messages, kwargs
            return LLMAdapterResponse(content="ok")

    config = RuntimeConfig(llm_adapter=_Adapter(), production_mode=False)
    request = RuntimeRequest(
        agent_id="agent",
        user_id="user",
        session_id="session",
        message="question",
        metadata=dict(metadata),
    )
    return RuntimeState(
        context=RuntimeContext(config=config, session_manager=MagicMock()),
        request=request,
        run_id="run-1",
    )


def test_extract_provider_metadata_from_rag_chunks() -> None:
    state = _runtime_state()
    state.context_builder_result = BuiltContext(
        history_messages=[],
        retrieved_chunks=[
            RetrievedChunk(
                id="chunk-1",
                text="policy text",
                metadata={"doc_id": "doc-1"},
                score=0.9,
            )
        ],
        rag_used=True,
        rag_reason="ok",
    )
    metadata = extract_provider_metadata_from_runtime_state(state)
    assert RAG_CHUNKS_METADATA_KEY in metadata
    assert metadata[RAG_CHUNKS_METADATA_KEY][0]["text"] == "policy text"


def test_merge_provider_metadata_into_request_updates_metadata() -> None:
    state = _runtime_state()
    state.tools_context_parts.append("WEB CONTEXT:\nExample search hit")
    merge_provider_metadata_into_request(state)
    assert "websearch_blocks" in state.request.metadata
    assert state.request.metadata["websearch_blocks"][0]["content"] == "Example search hit"


@pytest.mark.asyncio
async def test_synced_request_metadata_enables_rag_provider_collect() -> None:
    state = _runtime_state()
    state.context_builder_result = BuiltContext(
        history_messages=[],
        retrieved_chunks=[
            RetrievedChunk(id="c1", text="hit", metadata={"doc_id": "d1"}, score=0.8)
        ],
        rag_used=True,
        rag_reason="ok",
    )
    merge_provider_metadata_into_request(state)

    registry = materialize_context_plugin_registry(["intergrax.builtin"])
    provider = next(p for p in registry.list_providers() if p.provider_id == "builtin.rag")
    request = ContextAssemblyRequest(
        trace_id="t1",
        run_id="run-1",
        task_id="task-1",
        tenant_id="tenant-1",
        assembly_scope="uaep_turn",
        objective="question",
        decision_profile=ContextDecisionSnapshot(),
        budget_policy=ContextBudgetSnapshot(max_chars=8000),
        assembly_options=TaskContextAssemblyOptions(),
    )
    task = Task(
        tenant_id="tenant-1",
        user_id="user",
        message="question",
        context=TaskContext(),
        metadata=dict(state.request.metadata),
    )
    handles = build_graph_provider_handles(
        task,
        runtime_config=state.context.config,
        messages=[ChatMessage(role="user", content="question")],
        event_bus=None,
        node_id="agent",
        agent_id="agent",
        engine_id="default",
    )
    ctx = ContextProviderContext(engine_id="default", handles=handles)
    fragments = await provider.collect(request, ctx)
    assert fragments
    assert fragments[0].source == ContextFragmentSource.RAG


def test_runtime_state_history_emits_only_snapshot() -> None:
    state = _runtime_state(session_context_revision_id="rev-rs-1")
    state.request.tenant_id = "tenant-1"
    state.base_history = [
        ChatMessage(
            role="user",
            content="hello",
            entry_id="h1",
            tool_calls=[{"id": "tc1", "type": "function", "function": {"name": "x", "arguments": "{}"}}],
        ),
        ChatMessage(role="tool", content="tool out", entry_id="h2", tool_call_id="tc1", name="x"),
        ChatMessage(role="assistant", content="reply", entry_id="h3"),
    ]
    metadata = extract_provider_metadata_from_runtime_state(state)
    assert SESSION_HISTORY_SNAPSHOT_HANDLE in metadata
    assert SESSION_HISTORY_MESSAGES_METADATA_KEY not in metadata
    snapshot = metadata[SESSION_HISTORY_SNAPSHOT_HANDLE]
    assert len(snapshot.messages) == 3
    assert snapshot.messages[0].message_id == "h1"
    assert snapshot.messages[1].tool_call_id == "tc1"
    assert snapshot.messages[2].message_id == "h3"


def test_runtime_state_history_without_revision_fails_closed() -> None:
    state = _runtime_state()
    state.request.tenant_id = "tenant-1"
    state.base_history = [
        ChatMessage(role="user", content="hello", entry_id="h1"),
    ]
    metadata_before = dict(state.request.metadata)
    with pytest.raises(SessionHistorySnapshotRequiredError) as exc_info:
        extract_provider_metadata_from_runtime_state(state)
    assert str(exc_info.value) == SESSION_HISTORY_SNAPSHOT_REQUIRED_REASON
    assert state.request.metadata == metadata_before

    state2 = _runtime_state()
    state2.request.tenant_id = "tenant-1"
    state2.base_history = [ChatMessage(role="user", content="hello", entry_id="h1")]
    metadata_before_merge = dict(state2.request.metadata)
    with pytest.raises(SessionHistorySnapshotRequiredError):
        merge_provider_metadata_into_request(state2)
    assert state2.request.metadata == metadata_before_merge
