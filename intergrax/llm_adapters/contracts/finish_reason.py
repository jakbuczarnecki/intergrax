# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from enum import Enum


class LLMFinishReason(str, Enum):
    """Normalized completion stop reason across LLM providers."""

    COMPLETED = "completed"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    REFUSAL = "refusal"
    ERROR = "error"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


def parse_finish_reason(raw: str | None) -> LLMFinishReason:
    """Map provider-specific finish/stop reason strings to ``LLMFinishReason``."""
    if not raw:
        return LLMFinishReason.COMPLETED
    normalized = str(raw).strip().lower().replace("-", "_")
    aliases = {
        "stop": LLMFinishReason.COMPLETED,
        "end_turn": LLMFinishReason.COMPLETED,
        "complete": LLMFinishReason.COMPLETED,
        "max_tokens": LLMFinishReason.LENGTH,
        "tool_use": LLMFinishReason.TOOL_CALLS,
        "function_call": LLMFinishReason.TOOL_CALLS,
    }
    if normalized in aliases:
        return aliases[normalized]
    try:
        return LLMFinishReason(normalized)
    except ValueError:
        return LLMFinishReason.UNKNOWN
