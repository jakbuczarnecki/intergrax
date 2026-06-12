# © Artur Czarnecki. All rights reserved.

"""Default context fragment formatter (CE-FMT-1)."""

from __future__ import annotations

from intergrax.context.contracts import ContextAssemblyRequest, ContextFragment
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
        return [
            ChatMessage(
                role="system",
                content=f"[context:{fragment.source.value}:{fragment.source_id}] {fragment.content}",
            )
            for fragment in fragments
        ]


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
