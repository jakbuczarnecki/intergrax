# © Artur Czarnecki. All rights reserved.

"""CTX-UCL-3 session history provider tests."""

from __future__ import annotations

import inspect

import pytest

from intergrax.context.providers.builtin import _collect_session_history
from intergrax.context.providers.legacy_bridge import SESSION_HISTORY_MESSAGES_HANDLE
from intergrax.context.session_history import (
    SESSION_HISTORY_CONTEXT_SCOPE_HANDLE,
    SESSION_HISTORY_REVISION_HANDLE,
    SESSION_HISTORY_SNAPSHOT_BINDING_REASON,
    SESSION_HISTORY_SNAPSHOT_HANDLE,
    SESSION_HISTORY_SNAPSHOT_REQUIRED_REASON,
    HandleSessionHistoryProvider,
    SessionHistorySnapshotBindingError,
    SessionHistorySnapshotRequiredError,
    build_session_history_snapshot,
    fragments_from_session_history_snapshot,
    require_session_history_messages,
    session_history_chat_message_from_fragment,
)
from intergrax.context.contracts import (
    ContextAssemblyRequest,
    ContextBudgetSnapshot,
    ContextDecisionSnapshot,
    ContextProviderContext,
)
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions
from intergrax.llm.messages import ChatMessage

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _request() -> ContextAssemblyRequest:
    return ContextAssemblyRequest(
        trace_id="trace",
        run_id="run",
        task_id="task",
        tenant_id="tenant",
        assembly_scope="graph_node",
        objective="obj",
        decision_profile=ContextDecisionSnapshot(include_session_history=True),
        budget_policy=ContextBudgetSnapshot(),
        assembly_options=TaskContextAssemblyOptions(),
    )


def _bound_handles(
    snapshot: object,
    *,
    scope_id: str = "scope",
    revision_id: str = "rev",
) -> dict[str, object]:
    return {
        SESSION_HISTORY_SNAPSHOT_HANDLE: snapshot,
        SESSION_HISTORY_CONTEXT_SCOPE_HANDLE: scope_id,
        SESSION_HISTORY_REVISION_HANDLE: revision_id,
    }


@pytest.mark.asyncio
async def test_canonical_provider_returns_full_snapshot_fragments() -> None:
    messages = [
        ChatMessage(role="user", content=f"turn-{index}", entry_id=f"m{index}")
        for index in range(12)
    ]
    snapshot = build_session_history_snapshot(
        tenant_id="tenant",
        context_scope_id="scope",
        revision_id="rev",
        messages=messages,
    )
    ctx = ContextProviderContext(handles=_bound_handles(snapshot))
    fragments = await _collect_session_history(_request(), ctx)
    assert len(fragments) == 12
    assert fragments[0].metadata["message_id"] == "m0"
    assert not fragments[0].content.startswith("user:")


@pytest.mark.asyncio
async def test_missing_handle_returns_empty() -> None:
    fragments = await _collect_session_history(_request(), ContextProviderContext())
    assert fragments == []


@pytest.mark.asyncio
async def test_include_session_history_false_returns_none() -> None:
    provider = HandleSessionHistoryProvider()
    snapshot = build_session_history_snapshot(
        tenant_id="tenant",
        context_scope_id="scope",
        revision_id="rev",
        messages=[ChatMessage(role="user", content="x", entry_id="m1")],
    )
    request = ContextAssemblyRequest(
        trace_id="trace",
        run_id="run",
        task_id="task",
        tenant_id="tenant",
        assembly_scope="graph_node",
        objective="obj",
        decision_profile=ContextDecisionSnapshot(include_session_history=False),
        budget_policy=ContextBudgetSnapshot(),
        assembly_options=TaskContextAssemblyOptions(),
    )
    assert await provider.load_snapshot(request, ContextProviderContext(handles={SESSION_HISTORY_SNAPSHOT_HANDLE: snapshot})) is None


def test_canonical_provider_has_no_last_n_slicing() -> None:
    source = inspect.getsource(_collect_session_history)
    assert "[-max_entries:]" not in source
    assert "[-N:]" not in source
    assert "fragments_from_session_history_snapshot(snapshot)" in source
    assert "fragments_from_session_history(" not in source
    assert "max_memory_entries_in_context" not in source


@pytest.mark.asyncio
async def test_raw_legacy_handle_requires_snapshot() -> None:
    ctx = ContextProviderContext(
        handles={
            SESSION_HISTORY_MESSAGES_HANDLE: [
                ChatMessage(role="user", content="legacy", entry_id="legacy-1"),
            ],
        }
    )
    with pytest.raises(SessionHistorySnapshotRequiredError) as exc_info:
        await _collect_session_history(_request(), ctx)
    assert str(exc_info.value) == SESSION_HISTORY_SNAPSHOT_REQUIRED_REASON
    assert exc_info.value.reason == SESSION_HISTORY_SNAPSHOT_REQUIRED_REASON


@pytest.mark.asyncio
async def test_snapshot_has_priority_when_snapshot_and_legacy_handles_both_exist() -> None:
    snapshot = build_session_history_snapshot(
        tenant_id="tenant",
        context_scope_id="scope",
        revision_id="rev",
        messages=[ChatMessage(role="user", content="canonical", entry_id="m1")],
    )
    ctx = ContextProviderContext(
        handles={
            **_bound_handles(snapshot),
            SESSION_HISTORY_MESSAGES_HANDLE: [
                ChatMessage(role="user", content="legacy", entry_id="legacy-1"),
            ],
        }
    )
    fragments = await _collect_session_history(_request(), ctx)
    assert len(fragments) == 1
    assert fragments[0].metadata["message_id"] == "m1"
    assert "legacy" not in fragments[0].content


def _request_include_history(include_history: bool) -> ContextAssemblyRequest:
    return ContextAssemblyRequest(
        trace_id="trace",
        run_id="run",
        task_id="task",
        tenant_id="tenant",
        assembly_scope="graph_node",
        objective="obj",
        decision_profile=ContextDecisionSnapshot(include_session_history=include_history),
        budget_policy=ContextBudgetSnapshot(),
        assembly_options=TaskContextAssemblyOptions(),
    )


@pytest.mark.asyncio
async def test_include_history_false_legacy_handle_returns_empty() -> None:
    ctx = ContextProviderContext(
        handles={
            SESSION_HISTORY_MESSAGES_HANDLE: [
                ChatMessage(role="user", content="legacy", entry_id="legacy-1"),
            ],
        }
    )
    fragments = await _collect_session_history(_request_include_history(False), ctx)
    assert fragments == []


@pytest.mark.asyncio
async def test_include_history_false_snapshot_handle_returns_empty() -> None:
    snapshot = build_session_history_snapshot(
        tenant_id="tenant",
        context_scope_id="scope",
        revision_id="rev",
        messages=[ChatMessage(role="user", content="canonical", entry_id="m1")],
    )
    ctx = ContextProviderContext(
        handles=_bound_handles(snapshot, scope_id="scope", revision_id="rev")
    )
    fragments = await _collect_session_history(_request_include_history(False), ctx)
    assert fragments == []


@pytest.mark.asyncio
async def test_include_history_false_both_handles_returns_empty() -> None:
    snapshot = build_session_history_snapshot(
        tenant_id="tenant",
        context_scope_id="scope",
        revision_id="rev",
        messages=[ChatMessage(role="user", content="canonical", entry_id="m1")],
    )
    ctx = ContextProviderContext(
        handles={
            **_bound_handles(snapshot),
            SESSION_HISTORY_MESSAGES_HANDLE: [
                ChatMessage(role="user", content="legacy", entry_id="legacy-1"),
            ],
        }
    )
    fragments = await _collect_session_history(_request_include_history(False), ctx)
    assert fragments == []


def _snapshot_for_binding() -> object:
    return build_session_history_snapshot(
        tenant_id="tenant",
        context_scope_id="scope",
        revision_id="rev",
        messages=[ChatMessage(role="user", content="x", entry_id="m1")],
    )


@pytest.mark.asyncio
async def test_provider_rejects_snapshot_from_other_tenant() -> None:
    snapshot = _snapshot_for_binding()
    ctx = ContextProviderContext(handles=_bound_handles(snapshot))
    request = ContextAssemblyRequest(
        trace_id="trace",
        run_id="run",
        task_id="task",
        tenant_id="other-tenant",
        assembly_scope="graph_node",
        objective="obj",
        decision_profile=ContextDecisionSnapshot(include_session_history=True),
        budget_policy=ContextBudgetSnapshot(),
        assembly_options=TaskContextAssemblyOptions(),
    )
    with pytest.raises(SessionHistorySnapshotBindingError) as exc_info:
        await HandleSessionHistoryProvider().load_snapshot(request, ctx)
    assert str(exc_info.value) == SESSION_HISTORY_SNAPSHOT_BINDING_REASON
    assert exc_info.value.reason == SESSION_HISTORY_SNAPSHOT_BINDING_REASON
    assert "other-tenant" not in str(exc_info.value)
    assert "tenant" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_provider_rejects_snapshot_from_other_scope() -> None:
    snapshot = _snapshot_for_binding()
    ctx = ContextProviderContext(
        handles=_bound_handles(snapshot, scope_id="wrong-scope", revision_id="rev")
    )
    with pytest.raises(SessionHistorySnapshotBindingError):
        await HandleSessionHistoryProvider().load_snapshot(_request(), ctx)


@pytest.mark.asyncio
async def test_provider_rejects_snapshot_from_other_revision() -> None:
    snapshot = _snapshot_for_binding()
    ctx = ContextProviderContext(
        handles=_bound_handles(snapshot, scope_id="scope", revision_id="wrong-rev")
    )
    with pytest.raises(SessionHistorySnapshotBindingError):
        await HandleSessionHistoryProvider().load_snapshot(_request(), ctx)


@pytest.mark.asyncio
async def test_provider_rejects_unbound_snapshot() -> None:
    snapshot = _snapshot_for_binding()
    ctx = ContextProviderContext(handles={SESSION_HISTORY_SNAPSHOT_HANDLE: snapshot})
    with pytest.raises(SessionHistorySnapshotBindingError):
        await HandleSessionHistoryProvider().load_snapshot(_request(), ctx)


@pytest.mark.asyncio
async def test_provider_accepts_exactly_bound_snapshot() -> None:
    snapshot = _snapshot_for_binding()
    ctx = ContextProviderContext(handles=_bound_handles(snapshot))
    loaded = await HandleSessionHistoryProvider().load_snapshot(_request(), ctx)
    assert loaded is snapshot


@pytest.mark.parametrize(
    "malformed",
    [{}, (), "history", [{"role": "user", "content": "x"}], [object()]],
)
def test_require_session_history_messages_rejects_malformed(malformed: object) -> None:
    with pytest.raises(ValueError, match="session_history_messages"):
        require_session_history_messages(malformed)


def test_fragment_metadata_includes_name_and_tool_calls() -> None:
    tool_calls = [
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": "search", "arguments": "{}"},
        }
    ]
    snapshot = build_session_history_snapshot(
        tenant_id="tenant",
        context_scope_id="scope",
        revision_id="rev",
        messages=[
            ChatMessage(
                role="assistant",
                content="working",
                entry_id="m1",
                name="agent",
                tool_calls=tool_calls,
            )
        ],
    )
    fragment = fragments_from_session_history_snapshot(snapshot)[0]
    assert fragment.metadata["name"] == "agent"
    assert fragment.metadata["tool_calls"] == tool_calls


def test_session_history_chat_message_round_trip_preserves_model_fields() -> None:
    from intergrax.llm.messages import compute_model_facing_messages_hash

    original = ChatMessage(
        role="assistant",
        content="answer",
        entry_id="m1",
        name="agent",
        tool_calls=[
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "search", "arguments": "{}"},
            }
        ],
    )
    snapshot = build_session_history_snapshot(
        tenant_id="tenant",
        context_scope_id="scope",
        revision_id="rev",
        messages=[original],
    )
    fragment = fragments_from_session_history_snapshot(snapshot)[0]
    restored = session_history_chat_message_from_fragment(fragment)
    assert restored.role == original.role
    assert restored.content == original.content
    assert restored.entry_id == original.entry_id
    assert restored.name == original.name
    assert restored.tool_calls == original.tool_calls
    assert compute_model_facing_messages_hash([original]) == compute_model_facing_messages_hash(
        [restored]
    )


def test_formatter_rejects_malformed_session_history_fragment() -> None:
    from intergrax.context.contracts import ContextAssemblyRequest, ContextBudgetSnapshot, ContextDecisionSnapshot, ContextFragment, ContextFragmentSource
    from intergrax.context.formatter import DefaultContextFormatter
    from intergrax.contracts.context_assembly import TaskContextAssemblyOptions

    fragment = ContextFragment(
        fragment_id="frag-1",
        source=ContextFragmentSource.SESSION_HISTORY,
        source_id="msg-1",
        content="history text",
        token_estimate=10,
        relevance_score=1.0,
        freshness_score=1.0,
        confidence_score=1.0,
        mandatory=False,
        metadata={
            "message_id": "msg-1",
            "sequence": 0,
            "role": "user",
            "content_hash": "wrong-hash",
        },
        content_hash="wrong-hash",
    )
    request = ContextAssemblyRequest(
        trace_id="trace-1",
        run_id="run-1",
        task_id="task-1",
        tenant_id="tenant-1",
        assembly_scope="graph_node",
        objective="objective",
        decision_profile=ContextDecisionSnapshot(),
        budget_policy=ContextBudgetSnapshot(max_chars=8000),
        assembly_options=TaskContextAssemblyOptions(),
    )
    formatter = DefaultContextFormatter()
    with pytest.raises(ValueError):
        formatter.format([fragment], request)


def test_formatter_non_history_fragment_still_emits_system_block() -> None:
    from intergrax.context.contracts import ContextAssemblyRequest, ContextBudgetSnapshot, ContextDecisionSnapshot, ContextFragment, ContextFragmentSource
    from intergrax.context.formatter import DefaultContextFormatter
    from intergrax.contracts.context_assembly import TaskContextAssemblyOptions

    fragment = ContextFragment(
        fragment_id="frag-2",
        source=ContextFragmentSource.TASK_MESSAGE,
        source_id="task-1",
        content="objective text",
        token_estimate=10,
        relevance_score=1.0,
        freshness_score=1.0,
        confidence_score=1.0,
        mandatory=False,
    )
    request = ContextAssemblyRequest(
        trace_id="trace-1",
        run_id="run-1",
        task_id="task-1",
        tenant_id="tenant-1",
        assembly_scope="graph_node",
        objective="objective",
        decision_profile=ContextDecisionSnapshot(),
        budget_policy=ContextBudgetSnapshot(max_chars=8000),
        assembly_options=TaskContextAssemblyOptions(),
    )
    messages = DefaultContextFormatter().format([fragment], request)
    assert len(messages) == 1
    assert messages[0].role == "system"
    assert messages[0].content.startswith("[context:task_message:")


def test_structural_guards_session_history_r1() -> None:
    from intergrax.context import dedup as dedup_mod
    from intergrax.context import formatter as formatter_mod
    from intergrax.context import session_history as session_mod
    from intergrax.context.providers import builtin as builtin_mod
    from intergrax.context.providers import legacy_bridge as legacy_mod
    from intergrax.runtime.nexus.context import provider_handles as handles_mod

    builtin_source = inspect.getsource(builtin_mod._collect_session_history)
    assert "require_session_history_messages" in builtin_source
    assert "fragments_from_session_history(" not in builtin_source

    legacy_source = inspect.getsource(legacy_mod.fragments_from_session_history)
    assert "[-max_entries:]" not in legacy_source
    assert 'content=f"{role}: {text}"' not in legacy_source
    assert "require_session_history_messages" in legacy_source

    session_source = inspect.getsource(session_mod)
    assert "validate_session_history_snapshot_binding" in session_source
    assert "session_history_chat_message_from_fragment" in session_source
    assert "SessionHistoryMessage(" in session_source
    assert "session_history_message_to_chat_message" in session_source

    formatter_source = inspect.getsource(formatter_mod.DefaultContextFormatter.format)
    assert "session_history_chat_message_from_fragment" in formatter_source
    assert "ContextFragmentSource.SESSION_HISTORY" in formatter_source

    dedup_source = inspect.getsource(dedup_mod._dedup_identity_key)
    assert "SESSION_HISTORY" in dedup_source
    assert "source_id" in dedup_source

    handles_source = inspect.getsource(handles_mod.build_graph_provider_handles)
    assert "validate_session_history_snapshot_binding" in handles_source
    assert "SESSION_HISTORY_CONTEXT_SCOPE_HANDLE" in handles_source
    assert "SESSION_HISTORY_REVISION_HANDLE" in handles_source
    assert "SESSION_HISTORY_MESSAGES_HANDLE]" not in handles_source
