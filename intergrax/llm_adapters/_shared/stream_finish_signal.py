# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Provider stop / finish / refusal signal for stream and buffered paths (PDC-D3 F-D3-02).

``build_adapter_response`` normalizes whatever it is given. These helpers collect the provider's
terminal signal (a stop-reason string or enum, a refusal text or refusal deltas) so every adapter path
can pass it on instead of reporting ``COMPLETED`` with empty content. Pure functions, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from intergrax.llm_adapters.contracts.finish_reason import LLMFinishReason, parse_finish_reason
from intergrax.utils import attribute_access

#: Provider stop-reason spellings that ``parse_finish_reason`` does not know, mapped to its vocabulary.
_EXTRA_ALIASES: dict[str, str] = {
    "max_output_tokens": "length",
    "model_context_window_exceeded": "length",
    "stop_sequence": "stop",
    "tool_call": "tool_calls",
    "finish_reason_unspecified": "stop",
    "finish": "stop",
    "image_safety": "content_filter",
}


def raw_finish_text(value: object) -> str | None:
    """Return a provider stop reason as text; accepts ``str`` and SDK enums (``.name`` / ``.value``)."""
    if isinstance(value, str):
        return value.strip() or None
    for attribute in ("name", "value"):
        candidate = attribute_access.optional(value, attribute, None)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def finish_reason_from_raw(raw: object, *, has_tool_calls: bool = False) -> LLMFinishReason:
    """Map a provider stop reason to ``LLMFinishReason`` (missing signal means ``COMPLETED``)."""
    text = raw_finish_text(raw)
    if text is None:
        finish = LLMFinishReason.COMPLETED
    else:
        normalized = text.lower().replace("-", "_")
        finish = parse_finish_reason(_EXTRA_ALIASES.get(normalized, normalized))
    if has_tool_calls and finish is LLMFinishReason.COMPLETED:
        return LLMFinishReason.TOOL_CALLS
    return finish


@dataclass
class StreamFinishSignal:
    """Accumulates the terminal signal while a provider stream is consumed."""

    raw_finish: str | None = None
    _refusal: list[str] = field(default_factory=list)

    def observe_finish(self, raw: object) -> None:
        """Remember the latest non-empty stop reason."""
        text = raw_finish_text(raw)
        if text is not None:
            self.raw_finish = text

    def observe_refusal(self, piece: object) -> None:
        """Append a refusal text or refusal delta; non-string and empty pieces are ignored."""
        if isinstance(piece, str) and piece:
            self._refusal.append(piece)

    def resolve(self, *, has_tool_calls: bool = False) -> tuple[LLMFinishReason, str | None]:
        """Finish reason and refusal text for ``build_adapter_response`` (which normalizes both)."""
        refusal = "".join(self._refusal)
        return finish_reason_from_raw(self.raw_finish, has_tool_calls=has_tool_calls), refusal or None


def converse_stop_signal(response_or_event: object) -> object:
    """Stop reason of a Bedrock Converse response (``stopReason``) or stream event (``messageStop``)."""
    if not isinstance(response_or_event, dict):
        return None
    direct = response_or_event.get("stopReason")
    if direct:
        return direct
    message_stop = response_or_event.get("messageStop")
    if isinstance(message_stop, dict):
        return message_stop.get("stopReason")
    return None


def langchain_done_reason(message: object) -> object:
    """``done_reason`` of a LangChain chat message or chunk (``response_metadata``), if any."""
    metadata = attribute_access.optional(message, "response_metadata", None)
    if isinstance(metadata, dict):
        return metadata.get("done_reason")
    return None


def gemini_stop_signal(response: object) -> object:
    """Stop signal of a Gemini response or chunk: a prompt block reason, else the candidate finish reason."""
    feedback = attribute_access.optional(response, "prompt_feedback", None)
    block = raw_finish_text(attribute_access.optional(feedback, "block_reason", None))
    if block is not None:
        return block
    candidates = attribute_access.optional(response, "candidates", None)
    if isinstance(candidates, (list, tuple)) and candidates:
        return attribute_access.optional(candidates[0], "finish_reason", None)
    return None


def invoke_stop_signal(payload: object) -> object:
    """Stop reason of a Bedrock ``invoke_model`` body or stream chunk (family-specific spellings)."""
    if not isinstance(payload, dict):
        return None
    for key in ("stop_reason", "finish_reason", "stopReason", "completionReason"):
        value = payload.get(key)
        if value:
            return value
    delta = payload.get("delta")
    if isinstance(delta, dict):
        nested = delta.get("stop_reason")
        if nested:
            return nested
    results = payload.get("results")
    if isinstance(results, list) and results and isinstance(results[0], dict):
        return results[0].get("completionReason")
    return None
