# © Artur Czarnecki. All rights reserved.

"""CTX-UCL-3 session history snapshot tests."""

from __future__ import annotations

import pytest

from intergrax.context.session_history import (
    HandleSessionHistoryProvider,
    SessionHistoryMessage,
    SessionHistorySnapshot,
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


def test_supplied_correct_content_hash_accepted() -> None:
    message = ChatMessage(role="user", content="hello", entry_id="m1")
    converted = session_history_message_from_chat_message(message, sequence=0)
    restored = SessionHistoryMessage(
        message_id=converted.message_id,
        sequence=converted.sequence,
        role=converted.role,
        content=converted.content,
        content_hash=converted.content_hash,
    )
    assert restored.content_hash == converted.content_hash


def test_supplied_wrong_content_hash_rejected() -> None:
    with pytest.raises(ValueError, match="content_hash does not match canonical message content"):
        SessionHistoryMessage(
            message_id="m1",
            sequence=0,
            role="user",
            content="hello",
            content_hash="deadbeef",
        )


def test_supplied_wrong_snapshot_hash_rejected() -> None:
    message = ChatMessage(role="user", content="hello", entry_id="m1")
    history_message = session_history_message_from_chat_message(message, sequence=0)
    with pytest.raises(ValueError, match="source_content_hash does not match snapshot messages"):
        SessionHistorySnapshot(
            tenant_id="tenant",
            context_scope_id="scope",
            revision_id="rev-1",
            messages=(history_message,),
            source_content_hash="deadbeef",
        )


def test_nested_tool_call_input_mutation_does_not_modify_snapshot() -> None:
    nested = {"args": {"query": "weather"}}
    assistant = ChatMessage(
        role="assistant",
        content="",
        entry_id="a1",
        tool_calls=[{"id": "call-1", "type": "function", "function": {"name": "search", "arguments": nested}}],
    )
    snapshot = build_session_history_snapshot(
        tenant_id="tenant",
        context_scope_id="scope",
        revision_id="rev-1",
        messages=[assistant],
    )
    nested["args"]["query"] = "changed"
    assert snapshot.messages[0].tool_calls[0]["function"]["arguments"]["args"]["query"] == "weather"  # type: ignore[index]


def test_nested_snapshot_tool_calls_are_immutable() -> None:
    assistant = ChatMessage(
        role="assistant",
        content="",
        entry_id="a1",
        tool_calls=[{"id": "call-1", "type": "function", "function": {"name": "search"}}],
    )
    snapshot = build_session_history_snapshot(
        tenant_id="tenant",
        context_scope_id="scope",
        revision_id="rev-1",
        messages=[assistant],
    )
    with pytest.raises(TypeError):
        snapshot.messages[0].tool_calls[0]["id"] = "changed"  # type: ignore[index]


def test_roundtrip_restores_plain_dict_list() -> None:
    original = ChatMessage(
        role="assistant",
        content="hi",
        entry_id="m1",
        tool_calls=[{"id": "c1", "type": "function", "function": {"name": "x", "args": [1, 2]}}],
    )
    converted = session_history_message_from_chat_message(original, sequence=0)
    restored = session_history_message_to_chat_message(converted)
    assert isinstance(restored.tool_calls[0], dict)
    assert isinstance(restored.tool_calls[0]["function"]["args"], list)


def test_non_json_safe_nested_value_rejected() -> None:
    with pytest.raises(ValueError, match="non-JSON-safe"):
        session_history_message_from_chat_message(
            ChatMessage(
                role="assistant",
                content="",
                entry_id="a1",
                tool_calls=[{"id": "c1", "payload": {1, 2, 3}}],  # type: ignore[list-item]
            ),
            sequence=0,
        )


def test_messages_item_wrong_type_rejected() -> None:
    message = session_history_message_from_chat_message(
        ChatMessage(role="user", content="hello", entry_id="m1"),
        sequence=0,
    )
    with pytest.raises(ValueError, match="SessionHistoryMessage instances"):
        SessionHistorySnapshot(
            tenant_id="tenant",
            context_scope_id="scope",
            revision_id="rev-1",
            messages=(message, {"not": "a message"}),  # type: ignore[arg-type]
        )


def test_numeric_message_id_rejected() -> None:
    with pytest.raises(ValueError, match="message_id must be a non-empty string"):
        SessionHistoryMessage(
            message_id=1,  # type: ignore[arg-type]
            sequence=0,
            role="user",
            content="hello",
        )


def test_numeric_tenant_id_rejected() -> None:
    message = session_history_message_from_chat_message(
        ChatMessage(role="user", content="hello", entry_id="m1"),
        sequence=0,
    )
    with pytest.raises(ValueError, match="tenant_id must be a non-empty string"):
        SessionHistorySnapshot(
            tenant_id=1,  # type: ignore[arg-type]
            context_scope_id="scope",
            revision_id="rev-1",
            messages=(message,),
        )


def test_numeric_revision_id_rejected() -> None:
    message = session_history_message_from_chat_message(
        ChatMessage(role="user", content="hello", entry_id="m1"),
        sequence=0,
    )
    with pytest.raises(ValueError, match="revision_id must be a non-empty string"):
        SessionHistorySnapshot(
            tenant_id="tenant",
            context_scope_id="scope",
            revision_id=1,  # type: ignore[arg-type]
            messages=(message,),
        )


def test_numeric_name_rejected() -> None:
    with pytest.raises(ValueError, match="name must be a non-empty string"):
        SessionHistoryMessage(
            message_id="m1",
            sequence=0,
            role="user",
            content="hello",
            name=1,  # type: ignore[arg-type]
        )


def test_numeric_tool_call_id_rejected() -> None:
    with pytest.raises(ValueError, match="tool_call_id must be a non-empty string"):
        SessionHistoryMessage(
            message_id="m1",
            sequence=0,
            role="tool",
            content="result",
            tool_call_id=1,  # type: ignore[arg-type]
        )


def test_strenum_nested_value_rejected() -> None:
    from enum import StrEnum

    class SampleStrEnum(StrEnum):
        VALUE = "value"

    with pytest.raises(ValueError, match="non-JSON-safe"):
        session_history_message_from_chat_message(
            ChatMessage(
                role="assistant",
                content="",
                entry_id="a1",
                tool_calls=[{"id": "c1", "payload": SampleStrEnum.VALUE}],
            ),
            sequence=0,
        )


def test_intenum_nested_value_rejected() -> None:
    from enum import IntEnum

    class SampleIntEnum(IntEnum):
        VALUE = 1

    with pytest.raises(ValueError, match="non-JSON-safe"):
        session_history_message_from_chat_message(
            ChatMessage(
                role="assistant",
                content="",
                entry_id="a1",
                tool_calls=[{"id": "c1", "payload": SampleIntEnum.VALUE}],
            ),
            sequence=0,
        )


def test_content_hash_zero_rejected() -> None:
    with pytest.raises(ValueError, match="content_hash must be a non-empty string"):
        SessionHistoryMessage(
            message_id="m1",
            sequence=0,
            role="user",
            content="hello",
            content_hash=0,  # type: ignore[arg-type]
        )


def test_content_hash_false_rejected() -> None:
    with pytest.raises(ValueError, match="content_hash must be a non-empty string"):
        SessionHistoryMessage(
            message_id="m1",
            sequence=0,
            role="user",
            content="hello",
            content_hash=False,  # type: ignore[arg-type]
        )


def test_source_content_hash_zero_rejected() -> None:
    message = session_history_message_from_chat_message(
        ChatMessage(role="user", content="hello", entry_id="m1"),
        sequence=0,
    )
    with pytest.raises(ValueError, match="source_content_hash must be a non-empty string"):
        SessionHistorySnapshot(
            tenant_id="tenant",
            context_scope_id="scope",
            revision_id="rev-1",
            messages=(message,),
            source_content_hash=0,  # type: ignore[arg-type]
        )


def test_source_content_hash_false_rejected() -> None:
    message = session_history_message_from_chat_message(
        ChatMessage(role="user", content="hello", entry_id="m1"),
        sequence=0,
    )
    with pytest.raises(ValueError, match="source_content_hash must be a non-empty string"):
        SessionHistorySnapshot(
            tenant_id="tenant",
            context_scope_id="scope",
            revision_id="rev-1",
            messages=(message,),
            source_content_hash=False,  # type: ignore[arg-type]
        )
