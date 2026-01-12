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

from intergrax.llm.messages import ChatMessage, append_chat_messages


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
