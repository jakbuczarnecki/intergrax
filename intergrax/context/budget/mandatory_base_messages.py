# © Artur Czarnecki. All rights reserved.

"""Canonical mandatory base model-facing message classification (CE-02-R1F)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from intergrax.llm.messages import ChatMessage


def last_user_message_index(messages: Sequence[ChatMessage]) -> int:
    """Index of the final user turn in model-facing base messages."""
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].role == "user":
            return index
    return max(0, len(messages) - 1)


def _strict_tool_call_id(raw: object) -> str | None:
    if type(raw) is not str:
        return None
    stripped = raw.strip()
    if not stripped:
        return None
    return stripped


def _tool_call_group_complete(
    assistant: ChatMessage,
    tool_messages: Sequence[ChatMessage],
) -> bool:
    tool_calls = assistant.tool_calls
    if not tool_calls:
        return False
    call_ids: list[str] = []
    for call in tool_calls:
        if not isinstance(call, Mapping):
            return False
        call_id = _strict_tool_call_id(call.get("id"))
        if call_id is None:
            return False
        call_ids.append(call_id)
    if not call_ids or len(call_ids) != len(set(call_ids)):
        return False
    received: dict[str, int] = {}
    expected = set(call_ids)
    for tool_message in tool_messages:
        if tool_message.role != "tool":
            return False
        call_id = _strict_tool_call_id(tool_message.tool_call_id)
        if call_id is None or call_id not in expected or call_id in received:
            return False
        received[call_id] = 1
    return set(received.keys()) == expected


def mandatory_base_message_indices(messages: Sequence[ChatMessage]) -> frozenset[int]:
    """Message indices reserved as mandatory base context (planner-aligned, fail-closed)."""
    if not messages:
        return frozenset()

    mandatory: set[int] = set()
    last_user = last_user_message_index(messages)
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.role == "assistant" and message.tool_calls:
            start = index
            index += 1
            tool_messages: list[ChatMessage] = []
            while index < len(messages) and messages[index].role == "tool":
                tool_messages.append(messages[index])
                index += 1
            if not _tool_call_group_complete(message, tool_messages):
                mandatory.update(range(start, index))
            continue

        if message.role == "tool":
            mandatory.add(index)
            index += 1
            continue

        if message.role == "system":
            mandatory.add(index)
        elif message.role == "user" and index == last_user:
            mandatory.add(index)

        index += 1

    return frozenset(mandatory)


def estimate_mandatory_base_message_tokens(
    messages: Sequence[ChatMessage],
    *,
    count_text: Callable[[str], int],
) -> int:
    """Token reserve for mandatory base messages only (no fragments)."""
    indices = mandatory_base_message_indices(messages)
    total = 0
    for index in sorted(indices):
        total += max(0, count_text(messages[index].content or ""))
    return total
