# © Artur Czarnecki. All rights reserved.

"""Per-message token cost for the hard model-window budget (PDC-D2-R010, PDC-D2-R011).

A chat message occupies the context window with its text *and* with the structured fields the
provider serializes next to it: assistant ``tool_calls`` (id, function name, arguments), the tool
``name`` and the ``tool_call_id``. Sizing a message by ``content`` alone lets an assistant turn with
empty content and a huge arguments payload slip through every budget count.

The recipe matches the adapter estimator (``BaseLLMAdapter.estimate_tokens_for_messages``): the
``tool_calls`` list is serialized with ``json.dumps(..., sort_keys=True, default=str)`` (``str()``
fallback when the value cannot be serialized), followed by ``name`` and ``tool_call_id``. A message
without those fields costs exactly its content, as before.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from intergrax.llm.messages import ChatMessage

__all__ = ["message_non_content_cost", "message_token_cost", "non_content_texts"]


def non_content_texts(message: ChatMessage) -> tuple[str, ...]:
    """Deterministic text of the structured fields that are sent next to ``content``."""
    parts: list[str] = []
    if message.tool_calls:
        try:
            parts.append(json.dumps(message.tool_calls, sort_keys=True, default=str))
        except (TypeError, ValueError):
            parts.append(str(message.tool_calls))
    if message.name:
        parts.append(message.name)
    if message.tool_call_id:
        parts.append(message.tool_call_id)
    return tuple(parts)


def message_non_content_cost(message: ChatMessage, count_text: Callable[[str], int]) -> int:
    """Tokens of ``tool_calls``, ``name`` and ``tool_call_id``; content truncation cannot reduce them."""
    return sum(max(0, count_text(text)) for text in non_content_texts(message))


def message_token_cost(message: ChatMessage, count_text: Callable[[str], int]) -> int:
    """Tokens a message occupies in the model window: content plus its structured fields."""
    return max(0, count_text(message.content or "")) + message_non_content_cost(message, count_text)
