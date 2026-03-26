# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Tier-2 conversation memory policy for Legal Agent.

Defines how much of the session transcript is surfaced to LLM calls that only need a short
multi-turn hint (pipeline routing, tool-intent decision, replanner). Full chat persistence and
long-term memory remain governed by Nexus session storage and RuntimeConfig; this module is the
product-level contract for those trimmed snippets.
"""

from __future__ import annotations

from typing import Any, List, Sequence

from pydantic import BaseModel, Field

from intergrax.runtime.nexus.engine.runtime_state import RuntimeState

NO_PRIOR_TURNS_PLACEHOLDER = "(no prior turns in context)"


class LegalMemoryPolicy(BaseModel):
    """
    Tunable limits for legal-tier use of recent conversation in routing / tool-plan LLM prompts.

    Session store retention and ``RuntimeConfig.enable_user_longterm_memory`` are orthogonal;
    adjust those at deployment; use this model for per-SKU control of snippet shape.
    """

    conversation_tail_message_limit: int = Field(
        default=12,
        ge=1,
        le=256,
        description="Max recent chat messages included in routing/tool-decision conversation snippets.",
    )
    conversation_snippet_max_chars_per_message: int = Field(
        default=500,
        ge=1,
        le=32_768,
        description="Per-message cap when formatting snippets (limits contract prose in router context).",
    )


def build_legal_conversation_snippet(state: RuntimeState, *, policy: LegalMemoryPolicy) -> str:
    """
    Build a newline-separated ``role: excerpt`` block from ``state`` for legal routing/tool prompts.

    Uses ``messages_for_llm`` when set, otherwise ``built_history_messages`` (same precedence as
    the previous inline helper).
    """
    lines: List[str] = []
    msgs: Sequence[Any] = state.messages_for_llm or state.built_history_messages or []
    tail_n = policy.conversation_tail_message_limit
    cap = policy.conversation_snippet_max_chars_per_message
    tail = list(msgs)[-tail_n:]
    for m in tail:
        role = getattr(m, "role", "?")
        content = getattr(m, "content", "") or ""
        text = content if isinstance(content, str) else str(content)
        excerpt = text if len(text) <= cap else text[:cap]
        lines.append(f"{role}: {excerpt}")
    return "\n".join(lines) if lines else NO_PRIOR_TURNS_PLACEHOLDER
