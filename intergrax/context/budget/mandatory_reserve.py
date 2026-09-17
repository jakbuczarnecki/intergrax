# © Artur Czarnecki. All rights reserved.

"""Mandatory model-facing token reserve estimation (CE-02-R1)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from intergrax.context.contracts import ContextFragment
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.context.context_compiler import classify_candidates


def estimate_mandatory_reserve_tokens(
    *,
    base_messages: Sequence[ChatMessage],
    collected_fragments: Sequence[ContextFragment],
    count_text: Callable[[str], int],
) -> int:
    """Single mandatory reserve estimate for authoritative budget resolution."""
    message_mandatory = 0
    if base_messages:
        for candidate in classify_candidates(base_messages, count_tokens=count_text):
            if candidate.mandatory:
                message_mandatory += candidate.token_estimate
    fragment_mandatory = sum(
        max(0, fragment.token_estimate)
        for fragment in collected_fragments
        if fragment.mandatory
    )
    return message_mandatory + fragment_mandatory
