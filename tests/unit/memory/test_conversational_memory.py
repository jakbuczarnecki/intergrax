# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Unit tests for ConversationalMemory.

These tests define the behavioral contract for the conversation history aggregate:
- append semantics preserve order,
- max_messages trimming keeps the most recent messages (enforced on mutation),
- get_all() returns a defensive copy (callers cannot mutate internal state),
- get_recent(n) handles edge cases deterministically,
- get_for_model(native_tools=...) returns a stable, predictable projection of history,
- clear() removes all messages.

Why this matters:
ConversationalMemory is used across multiple layers (planning, prompting, stores).
Regressions here can silently corrupt the effective prompt context and break
higher-level behavior in non-obvious ways.
"""

from __future__ import annotations

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.memory.conversational_memory import ConversationalMemory


pytestmark = pytest.mark.unit


def test_add_preserves_order() -> None:
    """
    Messages must be stored in the exact order they are added.
    """
    mem = ConversationalMemory(session_id="s1")
    mem.add("user", "u1")
    mem.add("assistant", "a1")
    mem.add("user", "u2")

    assert [m.content for m in mem.get_all()] == ["u1", "a1", "u2"]


def test_get_all_returns_defensive_copy() -> None:
    """
    get_all() must return a copy of the internal list so callers cannot mutate state.
    """
    mem = ConversationalMemory(session_id="s1")
    mem.add("user", "u1")

    out = mem.get_all()
    out.append(ChatMessage(role="user", content="mutate"))

    assert [m.content for m in mem.get_all()] == ["u1"]


def test_max_messages_trims_to_most_recent_on_add() -> None:
    """
    When max_messages is set, the aggregate must keep only the most recent messages.

    Contract:
    - trimming is enforced on mutation (add/add_message/extend),
      not via a separate public trim() method.
    """
    mem = ConversationalMemory(session_id="s1", max_messages=3)

    mem.add("user", "m1")
    mem.add("user", "m2")
    mem.add("user", "m3")
    mem.add("user", "m4")

    assert [m.content for m in mem.get_all()] == ["m2", "m3", "m4"]


def test_extend_appends_and_trims_if_needed() -> None:
    """
    extend() must append messages in order and enforce max_messages trimming.
    """
    mem = ConversationalMemory(session_id="s1", max_messages=3)

    mem.extend(
        [
            ChatMessage(role="user", content="m1"),
            ChatMessage(role="user", content="m2"),
            ChatMessage(role="user", content="m3"),
        ]
    )
    mem.extend(
        [
            ChatMessage(role="user", content="m4"),
            ChatMessage(role="user", content="m5"),
        ]
    )

    assert [m.content for m in mem.get_all()] == ["m3", "m4", "m5"]


def test_get_recent_n_leq_zero_returns_empty_list() -> None:
    """
    get_recent(n) must return [] for n <= 0.
    """
    mem = ConversationalMemory(session_id="s1")
    mem.add("user", "x")

    assert mem.get_recent(0) == []
    assert mem.get_recent(-1) == []


def test_get_recent_returns_last_n_preserving_order() -> None:
    """
    get_recent(n) must return the last n messages in the original order.
    """
    mem = ConversationalMemory(session_id="s1")
    mem.add("user", "m1")
    mem.add("user", "m2")
    mem.add("user", "m3")
    mem.add("user", "m4")

    assert [m.content for m in mem.get_recent(2)] == ["m3", "m4"]
    assert [m.content for m in mem.get_recent(10)] == ["m1", "m2", "m3", "m4"]


def test_get_for_model_native_tools_false_returns_full_history() -> None:
    """
    get_for_model(native_tools=False) must return the full history projection.
    """
    mem = ConversationalMemory(session_id="s1")
    mem.add("system", "s")
    mem.add("user", "u")
    mem.add("assistant", "a")
    mem.add("tool", "t")  # should remain when native_tools=False

    out = mem.get_for_model(native_tools=False)
    assert [m.content for m in out] == ["s", "u", "a", "t"]


def test_get_for_model_native_tools_true_filters_out_tool_messages() -> None:
    """
    get_for_model(native_tools=True) must filter out historical 'tool' messages
    and keep only system/user/assistant.

    This is critical to avoid leaking legacy tool logs into model context when
    using native tool calling backends.
    """
    mem = ConversationalMemory(session_id="s1")
    mem.add("system", "s")
    mem.add("user", "u")
    mem.add("assistant", "a")
    mem.add("tool", "internal-tool-log")

    out = mem.get_for_model(native_tools=True)
    assert [m.content for m in out] == ["s", "u", "a"]


def test_clear_removes_all_messages() -> None:
    """
    clear() must remove all messages from the aggregate.
    """
    mem = ConversationalMemory(session_id="s1")
    mem.add("user", "m1")
    mem.add("assistant", "m2")

    mem.clear()

    assert mem.get_all() == []
    assert mem.get_recent(10) == []
    assert mem.get_for_model(native_tools=False) == []
