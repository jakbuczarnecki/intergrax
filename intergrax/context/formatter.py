# © Artur Czarnecki. All rights reserved.

"""Default context fragment formatter (CE-FMT-1)."""

from __future__ import annotations

from intergrax.context.contracts import ContextAssemblyRequest, ContextFragment, ContextFragmentSource
from intergrax.context.session_history import session_history_chat_message_from_fragment
from intergrax.llm.messages import ChatMessage


def _last_user_index(messages: list[ChatMessage]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].role == "user":
            return index
    return len(messages)


class DefaultContextFormatter:
    """Formats ranked fragments as system injection blocks merged into the base window."""

    def format(
        self,
        fragments: list[ContextFragment],
        request: ContextAssemblyRequest,
    ) -> list[ChatMessage]:
        _ = request
        formatted: list[ChatMessage] = []
        for fragment in fragments:
            if fragment.source is ContextFragmentSource.SESSION_HISTORY:
                formatted.append(session_history_chat_message_from_fragment(fragment))
                continue
            if fragment.source is ContextFragmentSource.TOOL_OUTPUT:
                tool_call_id = fragment.metadata.get("tool_call_id")
                tool_name = fragment.metadata.get("tool_name")
                if isinstance(tool_call_id, str) and tool_call_id.strip():
                    formatted.append(
                        ChatMessage(
                            role="tool",
                            content=fragment.content,
                            tool_call_id=tool_call_id.strip(),
                            name=tool_name.strip() if isinstance(tool_name, str) and tool_name.strip() else None,
                        )
                    )
                    continue
            formatted.append(
                ChatMessage(
                    role="system",
                    content=f"[context:{fragment.source.value}:{fragment.source_id}] {fragment.content}",
                )
            )
        return formatted


def _ordered_assistant_tool_call_ids(message: ChatMessage) -> tuple[str, ...]:
    if message.role != "assistant" or not message.tool_calls:
        return ()
    ordered: list[str] = []
    for tool_call in message.tool_calls:
        if not isinstance(tool_call, dict):
            continue
        call_id = tool_call.get("id")
        if isinstance(call_id, str) and call_id.strip():
            ordered.append(call_id.strip())
    return tuple(ordered)


def merge_iterative_tool_feedback_messages(
    base_messages: list[ChatMessage],
    fragment_messages: list[ChatMessage],
) -> list[ChatMessage]:
    """Interleave native tool-feedback immediately after each assistant tool-call turn."""
    if not fragment_messages:
        return list(base_messages)
    tool_pool: dict[str, ChatMessage] = {}
    non_tool_fragments: list[ChatMessage] = []
    for message in fragment_messages:
        if message.role == "tool" and message.tool_call_id:
            tool_pool[message.tool_call_id] = message
            continue
        non_tool_fragments.append(message)

    if not base_messages:
        return non_tool_fragments + list(tool_pool.values())

    merged: list[ChatMessage] = []
    for message in base_messages:
        merged.append(message)
        for call_id in _ordered_assistant_tool_call_ids(message):
            tool_message = tool_pool.pop(call_id, None)
            if tool_message is not None:
                merged.append(tool_message)

    if tool_pool:
        merged.extend(tool_pool.values())
    if non_tool_fragments:
        merged.extend(non_tool_fragments)
    return merged


def merge_fragment_messages(
    base_messages: list[ChatMessage],
    fragment_messages: list[ChatMessage],
) -> list[ChatMessage]:
    """Insert fragment messages immediately before the last user turn (CE-FMT-1)."""
    if not fragment_messages:
        return list(base_messages)
    if not base_messages:
        return list(fragment_messages)
    insert_at = _last_user_index(base_messages)
    return (
        list(base_messages[:insert_at])
        + list(fragment_messages)
        + list(base_messages[insert_at:])
    )
