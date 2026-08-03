# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Unit tests for intergrax.llm.messages.

These tests define the behavioral contract for:
- ChatMessage.to_dict(): stable projection to provider-compatible payloads,
- append_chat_messages(): deterministic, side-effect-free reducer semantics.

Why this matters:
- to_dict() is a boundary contract with external LLM APIs. Regressions here
  break integrations in subtle ways.
- append_chat_messages() is a state merge reducer used across graph/pipeline flows.
  It must be stable, deterministic, and must not mutate inputs.
"""

from __future__ import annotations

import pytest

from intergrax.llm.messages import (
    ChatMessage,
    append_chat_messages,
    build_model_input_messages_envelope,
    compute_model_facing_messages_hash,
    model_input_messages_from_envelope,
    replace_final_user_message,
    requires_structured_model_input,
    STRUCTURED_MODEL_INPUT_REQUIRED_REASON,
    StructuredModelInputRequiredError,
)


pytestmark = pytest.mark.unit


def test_append_chat_messages_when_existing_none_returns_copy_of_new() -> None:
    """
    When existing is None, reducer must return a new list containing all new messages.

    Contract:
    - return value is a list,
    - contains items in the same order as `new`,
    - and is NOT the same list object as `new` (defensive copy).
    """
    new = [ChatMessage(role="user", content="a"), ChatMessage(role="assistant", content="b")]

    out = append_chat_messages(None, new)

    assert out == new
    assert out is not new


def test_append_chat_messages_appends_preserving_order_and_not_mutating_inputs() -> None:
    """
    Reducer must append new messages to existing messages, preserving order.

    Contract:
    - output == existing + new
    - existing and new lists are not mutated
    """
    existing = [ChatMessage(role="user", content="e1")]
    new = [ChatMessage(role="assistant", content="n1"), ChatMessage(role="user", content="n2")]

    existing_snapshot = list(existing)
    new_snapshot = list(new)

    out = append_chat_messages(existing, new)

    assert out == [*existing_snapshot, *new_snapshot]
    assert existing == existing_snapshot
    assert new == new_snapshot


def test_append_chat_messages_returns_new_list_object() -> None:
    """
    Reducer must return a new list object (no aliasing with existing).
    """
    existing = [ChatMessage(role="user", content="e1")]
    new = [ChatMessage(role="assistant", content="n1")]

    out = append_chat_messages(existing, new)

    assert out is not existing
    assert out is not new


def test_chat_message_to_dict_minimal_fields() -> None:
    """
    to_dict() must return the minimal provider-compatible payload.

    Contract:
    - includes only role/content by default
    - does not leak internal fields (entry_id, created_at, metadata, attachments, etc.)
    """
    msg = ChatMessage(role="user", content="hello")
    out = msg.to_dict()

    assert out == {"role": "user", "content": "hello"}
    assert "entry_id" not in out
    assert "created_at" not in out
    assert "metadata" not in out
    assert "attachments" not in out


def test_chat_message_to_dict_includes_name_when_set() -> None:
    """
    If name is set, to_dict() must include it.
    """
    msg = ChatMessage(role="assistant", content="x", name="agent")
    assert msg.to_dict() == {"role": "assistant", "content": "x", "name": "agent"}


def test_chat_message_to_dict_includes_tool_call_id_when_set() -> None:
    """
    If tool_call_id is set, to_dict() must include it.
    """
    msg = ChatMessage(role="tool", content="result", tool_call_id="call_123")
    assert msg.to_dict() == {"role": "tool", "content": "result", "tool_call_id": "call_123"}


def test_chat_message_to_dict_includes_tool_calls_when_set() -> None:
    """
    If tool_calls is set, to_dict() must include it.
    """
    tool_calls = [{"id": "t1", "type": "function", "function": {"name": "x", "arguments": "{}"}}]
    msg = ChatMessage(role="assistant", content="x", tool_calls=tool_calls)

    assert msg.to_dict() == {"role": "assistant", "content": "x", "tool_calls": tool_calls}


def test_chat_message_to_dict_does_not_include_optional_fields_when_empty_or_none() -> None:
    """
    Optional fields must not appear in the payload when unset/empty.

    Contract:
    - name/tool_call_id/tool_calls are omitted when None or empty.
    """
    msg = ChatMessage(role="assistant", content="x", name=None, tool_call_id=None, tool_calls=None)
    assert msg.to_dict() == {"role": "assistant", "content": "x"}

    msg2 = ChatMessage(role="assistant", content="x", tool_calls=[])
    assert msg2.to_dict() == {"role": "assistant", "content": "x"}


def test_compute_model_facing_messages_hash_is_stable_for_equivalent_messages() -> None:
    first = [
        ChatMessage(role="system", content="SYNTH-A", entry_id="one"),
        ChatMessage(role="user", content="SYNTH-B", created_at="2020-01-01T00:00:00"),
    ]
    second = [
        ChatMessage(role="system", content="SYNTH-A", entry_id="two"),
        ChatMessage(role="user", content="SYNTH-B", created_at="2026-01-01T00:00:00"),
    ]
    assert compute_model_facing_messages_hash(first) == compute_model_facing_messages_hash(second)


def test_compute_model_facing_messages_hash_changes_when_content_changes() -> None:
    baseline = [ChatMessage(role="user", content="SYNTH-A")]
    changed = [ChatMessage(role="user", content="SYNTH-B")]
    assert compute_model_facing_messages_hash(baseline) != compute_model_facing_messages_hash(changed)


def _structured_round_trip_messages() -> list[ChatMessage]:
    return [
        ChatMessage(role="system", content="[context:task_message:task-1] objective"),
        ChatMessage(role="user", content="history user", entry_id="hist-user"),
        ChatMessage(
            role="assistant",
            content="assistant reply",
            entry_id="hist-assistant",
            name="agent",
            tool_calls=[{"id": "call-1", "type": "function", "function": {"name": "search", "arguments": "{}"}}],
        ),
        ChatMessage(
            role="tool",
            content="tool result",
            entry_id="hist-tool",
            name="search",
            tool_call_id="call-1",
        ),
        ChatMessage(role="user", content="final user", entry_id="final-user"),
    ]


def test_model_input_envelope_round_trip_preserves_exact_send() -> None:
    messages = _structured_round_trip_messages()
    envelope = build_model_input_messages_envelope(messages)
    restored = model_input_messages_from_envelope(envelope)
    assert len(restored) == len(messages)
    for original, round_trip in zip(messages, restored, strict=True):
        assert round_trip.role == original.role
        assert round_trip.content == original.content
        assert round_trip.entry_id == original.entry_id
        assert round_trip.name == original.name
        assert round_trip.tool_call_id == original.tool_call_id
        assert round_trip.tool_calls == original.tool_calls
    assert compute_model_facing_messages_hash(messages) == compute_model_facing_messages_hash(restored)


def test_model_input_envelope_rejects_hash_mutation() -> None:
    envelope = build_model_input_messages_envelope(_structured_round_trip_messages())
    envelope["messages_hash"] = "tampered"
    with pytest.raises(ValueError, match="hash mismatch"):
        model_input_messages_from_envelope(envelope)


def test_model_input_envelope_rejects_tool_call_mutation() -> None:
    messages = _structured_round_trip_messages()
    envelope = build_model_input_messages_envelope(messages)
    rows = envelope["messages"]
    assert isinstance(rows, list)
    assistant_row = rows[2]
    assert isinstance(assistant_row, dict)
    assistant_row["tool_calls"] = [{"id": "mutated", "type": "function", "function": {"name": "x", "arguments": "{}"}}]
    with pytest.raises(ValueError, match="hash mismatch"):
        model_input_messages_from_envelope(envelope)


def test_model_input_envelope_rejects_duplicate_entry_ids() -> None:
    messages = [
        ChatMessage(role="user", content="one", entry_id="dup"),
        ChatMessage(role="user", content="two", entry_id="dup"),
    ]
    envelope = build_model_input_messages_envelope(messages)
    with pytest.raises(ValueError, match="duplicate"):
        model_input_messages_from_envelope(envelope)


def test_model_input_envelope_rejects_non_json_tool_calls() -> None:
    messages = [
        ChatMessage(
            role="assistant",
            content="x",
            entry_id="a1",
            tool_calls=[{"id": float("nan"), "type": "function", "function": {"name": "x", "arguments": "{}"}}],
        ),
        ChatMessage(role="user", content="final", entry_id="u1"),
    ]
    with pytest.raises(ValueError):
        build_model_input_messages_envelope(messages)


def test_replace_final_user_message_changes_only_content() -> None:
    messages = _structured_round_trip_messages()
    updated = replace_final_user_message(messages, "agent-specific prompt")
    assert len(updated) == len(messages)
    for index in range(len(messages) - 1):
        assert updated[index].role == messages[index].role
        assert updated[index].content == messages[index].content
        assert updated[index].entry_id == messages[index].entry_id
    final = updated[-1]
    assert final.role == "user"
    assert final.content == "agent-specific prompt"
    assert final.entry_id == messages[-1].entry_id
    assert final.name == messages[-1].name
    assert final.tool_call_id == messages[-1].tool_call_id
    assert final.tool_calls == messages[-1].tool_calls


def test_structured_transport_detection() -> None:
    simple = [
        ChatMessage(role="system", content="[context:task_message:t1] objective"),
        ChatMessage(role="user", content="final only"),
    ]
    assert requires_structured_model_input(simple) is False
    structured = _structured_round_trip_messages()
    assert requires_structured_model_input(structured) is True
    assert requires_structured_model_input([]) is False
    assert StructuredModelInputRequiredError().reason == STRUCTURED_MODEL_INPUT_REQUIRED_REASON
