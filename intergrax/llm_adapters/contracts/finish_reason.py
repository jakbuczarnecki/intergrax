# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from enum import Enum


class LLMFinishReason(str, Enum):  # noqa: UP042 - str-mixin kept for wire/serialization compatibility
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
        # Provider-native refusal / safety vocabularies (PDC-D3-R009).
        "refused": LLMFinishReason.REFUSAL,
        "content_filtered": LLMFinishReason.CONTENT_FILTER,
        "guardrail_intervened": LLMFinishReason.CONTENT_FILTER,
        "safety": LLMFinishReason.CONTENT_FILTER,
        "prohibited_content": LLMFinishReason.CONTENT_FILTER,
        "blocklist": LLMFinishReason.CONTENT_FILTER,
        "spii": LLMFinishReason.CONTENT_FILTER,
        "recitation": LLMFinishReason.CONTENT_FILTER,
        "error_toxic": LLMFinishReason.CONTENT_FILTER,
    }
    if normalized in aliases:
        return aliases[normalized]
    try:
        return LLMFinishReason(normalized)
    except ValueError:
        return LLMFinishReason.UNKNOWN


#: Canonical ``refusal`` payload when a provider signals a refusal without any message text.
REFUSAL_UNSPECIFIED = "refused_without_message"


def refusal_text(value: object) -> str | None:
    """Return a provider refusal payload only when it is a non-blank ``str``."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def normalize_refusal(
    finish_reason: LLMFinishReason,
    refusal: object,
) -> tuple[LLMFinishReason, str | None]:
    """Normalize a refusal across providers (PDC-D3-R009).

    Invariant of the result: ``finish_reason is REFUSAL`` if and only if ``refusal`` is a
    non-empty string. A refusal payload upgrades any non-error finish reason to ``REFUSAL``; a
    ``REFUSAL`` finish without text receives ``REFUSAL_UNSPECIFIED``; a payload that is not a
    non-blank string is dropped. Content-filter stops stay ``CONTENT_FILTER`` and carry no
    refusal text.
    """
    text = refusal_text(refusal)
    if text is not None and finish_reason is not LLMFinishReason.ERROR:
        return LLMFinishReason.REFUSAL, text
    if finish_reason is LLMFinishReason.REFUSAL:
        return LLMFinishReason.REFUSAL, text or REFUSAL_UNSPECIFIED
    return finish_reason, None
