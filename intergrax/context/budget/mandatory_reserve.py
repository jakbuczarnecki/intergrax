# © Artur Czarnecki. All rights reserved.

"""Mandatory model-facing token reserve estimation (CE-02-R1)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from intergrax.context.budget.mandatory_base_messages import estimate_mandatory_base_message_tokens
from intergrax.context.contracts import ContextFragment
from intergrax.llm.messages import ChatMessage


def estimate_mandatory_reserve_tokens(
    *,
    base_messages: Sequence[ChatMessage],
    collected_fragments: Sequence[ContextFragment],
    count_text: Callable[[str], int],
) -> int:
    """Single mandatory reserve estimate for authoritative budget resolution."""
    message_mandatory = estimate_mandatory_base_message_tokens(
        base_messages,
        count_text=count_text,
    )
    fragment_mandatory = sum(
        max(0, fragment.token_estimate)
        for fragment in collected_fragments
        if fragment.mandatory
    )
    return message_mandatory + fragment_mandatory
