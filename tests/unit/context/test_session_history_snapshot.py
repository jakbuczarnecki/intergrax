# © Artur Czarnecki. All rights reserved.

"""CTX-UCL-3 session history snapshot tests."""

from __future__ import annotations

import pytest

from intergrax.context.session_history import (
    HandleSessionHistoryProvider,
    build_session_history_snapshot,
    session_history_message_from_chat_message,
    session_history_message_to_chat_message,
)
from intergrax.context.contracts import ContextAssemblyRequest, ContextBudgetSnapshot, ContextDecisionSnapshot, ContextProviderContext
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions
from intergrax.llm.messages import ChatMessage

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _request(*, include_history: bool = True) -> ContextAssemblyRequest:
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


def test_snapshot_preserves_full_history() -> None:
    messages = [
        ChatMessage(role="user", content="one", entry_id="m0"),
        ChatMessage(role="assistant", content="two", entry_id="m1"),
        ChatMessage(role="user", content="three", entry_id="m2"),
    ]
    snapshot = build_session_history_snapshot(
        tenant_id="tenant",
        context_scope_id="scope",
        revision_id="rev-1",
        messages=messages,
    )
    assert len(snapshot.messages) == 3
    assert snapshot.source_refs == ("m0", "m1", "m2")


def test_snapshot_rejects_duplicate_ids() -> None:
    messages = [
        ChatMessage(role="user", content="one", entry_id="dup"),
        ChatMessage(role="user", content="two", entry_id="dup"),
    ]
    with pytest.raises(ValueError, match="duplicate"):
        build_session_history_snapshot(
            tenant_id="tenant",
            context_scope_id="scope",
            revision_id="rev-1",
            messages=messages,
        )


def test_snapshot_hash_deterministic() -> None:
    messages = [ChatMessage(role="user", content="hello", entry_id="m1")]
    first = build_session_history_snapshot(
        tenant_id="tenant",
        context_scope_id="scope",
        revision_id="rev-1",
        messages=messages,
    )
    second = build_session_history_snapshot(
        tenant_id="tenant",
        context_scope_id="scope",
        revision_id="rev-1",
        messages=messages,
    )
    assert first.source_content_hash == second.source_content_hash


def test_snapshot_content_change_changes_hash() -> None:
    first = build_session_history_snapshot(
        tenant_id="tenant",
        context_scope_id="scope",
        revision_id="rev-1",
        messages=[ChatMessage(role="user", content="a", entry_id="m1")],
    )
    second = build_session_history_snapshot(
        tenant_id="tenant",
        context_scope_id="scope",
        revision_id="rev-1",
        messages=[ChatMessage(role="user", content="b", entry_id="m1")],
    )
    assert first.source_content_hash != second.source_content_hash


def test_tool_call_linkage_preserved() -> None:
    assistant = ChatMessage(
        role="assistant",
        content="",
        entry_id="a1",
        tool_calls=[{"id": "call-1", "type": "function", "function": {"name": "search"}}],
    )
    tool = ChatMessage(role="tool", content="result", entry_id="t1", tool_call_id="call-1")
    snapshot = build_session_history_snapshot(
        tenant_id="tenant",
        context_scope_id="scope",
        revision_id="rev-1",
        messages=[assistant, tool],
    )
    assert snapshot.messages[0].tool_calls
    assert snapshot.messages[1].tool_call_id == "call-1"


def test_chat_message_roundtrip() -> None:
    original = ChatMessage(
        role="assistant",
        content="hi",
        entry_id="m1",
        tool_calls=[{"id": "c1", "type": "function", "function": {"name": "x"}}],
    )
    converted = session_history_message_from_chat_message(original, sequence=0)
    restored = session_history_message_to_chat_message(converted)
    assert restored.entry_id == original.entry_id
    assert restored.tool_calls == original.tool_calls


@pytest.mark.asyncio
async def test_provider_rejects_wrong_handle_type() -> None:
    provider = HandleSessionHistoryProvider()
    ctx = ContextProviderContext(handles={"session_history_snapshot": ["not-a-snapshot"]})
    with pytest.raises(ValueError, match="SessionHistorySnapshot"):
        await provider.load_snapshot(_request(), ctx)
