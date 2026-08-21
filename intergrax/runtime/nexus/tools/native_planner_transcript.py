# © Artur Czarnecki. All rights reserved.

"""ENG-5/ENG-6 — canonical model-visible native planner transcript semantics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

from intergrax.llm.messages import ChatMessage


class _AssistantToolCallSpec(NamedTuple):
    valid: bool
    required_ids: frozenset[str]


def _parse_assistant_tool_call_ids(
    tool_calls: Sequence[dict] | None,
) -> _AssistantToolCallSpec:
    if not tool_calls:
        return _AssistantToolCallSpec(valid=True, required_ids=frozenset())
    seen: set[str] = set()
    declared: set[str] = set()
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        call_id = tool_call.get("id")
        if not isinstance(call_id, str) or not call_id:
            continue
        if call_id in seen:
            return _AssistantToolCallSpec(valid=False, required_ids=frozenset())
        seen.add(call_id)
        declared.add(call_id)
    return _AssistantToolCallSpec(valid=True, required_ids=frozenset(declared))


def canonical_native_planner_messages(messages: Sequence[ChatMessage]) -> list[ChatMessage]:
    """
    Provider-safe transcript: complete ordered assistant→tool exchanges only.

    Does not mutate the caller-owned message list.
    """
    pruned: list[ChatMessage] = []
    index = 0
    message_count = len(messages)
    while index < message_count:
        message = messages[index]

        if message.role == "tool":
            index += 1
            continue

        if message.role == "assistant" and message.tool_calls:
            spec = _parse_assistant_tool_call_ids(message.tool_calls)
            if not spec.valid:
                scan_index = index + 1
                while scan_index < message_count and messages[scan_index].role == "tool":
                    scan_index += 1
                index = scan_index
                continue

            required_ids = spec.required_ids
            if not required_ids:
                pruned.append(message)
                index += 1
                continue

            scan_index = index + 1
            observed_ids: set[str] = set()
            tool_group: list[ChatMessage] = []
            group_valid = True
            while scan_index < message_count and messages[scan_index].role == "tool":
                tool_message = messages[scan_index]
                tool_call_id = tool_message.tool_call_id
                if (
                    not isinstance(tool_call_id, str)
                    or not tool_call_id
                    or tool_call_id not in required_ids
                    or tool_call_id in observed_ids
                ):
                    group_valid = False
                else:
                    observed_ids.add(tool_call_id)
                    tool_group.append(tool_message)
                scan_index += 1

            if group_valid and observed_ids == set(required_ids):
                pruned.append(message)
                pruned.extend(tool_group)
                index = scan_index
                continue

            index = scan_index if scan_index > index + 1 else index + 1
            continue

        if message.role in ("system", "user", "assistant"):
            pruned.append(message)
        index += 1

    return pruned
