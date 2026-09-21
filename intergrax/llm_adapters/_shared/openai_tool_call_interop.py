# © Artur Czarnecki. All rights reserved.

"""OpenAI-family SDK and wire parsing into canonical ``LLMToolCall``."""

from __future__ import annotations

import json
from collections.abc import Iterable

from intergrax.llm_adapters.contracts.serialized_value import JsonObject
from intergrax.llm_adapters.contracts.tool_call import (
    LLMToolCall,
    finalize_accepted_tool_call_identities,
)
from intergrax.utils import attribute_access


def tool_calls_from_openai_message(message: object) -> tuple[LLMToolCall, ...]:
    """Extract typed tool calls from an OpenAI-style chat completion message."""
    raw = attribute_access.optional(message, "tool_calls", None) or []
    out: list[LLMToolCall] = []
    for tc in raw:
        fn = attribute_access.optional(tc, "function", None)
        if fn is None and isinstance(tc, dict):
            fn = tc.get("function")
        name = attribute_access.optional(fn, "name", None) if fn is not None else None
        args = attribute_access.optional(fn, "arguments", None) if fn is not None else None
        if name is None and isinstance(fn, dict):
            name = fn.get("name")
            args = fn.get("arguments")
        tc_id = attribute_access.optional(tc, "id", None) or (
            tc.get("id") if isinstance(tc, dict) else None
        )
        if not name:
            continue
        out.append(
            LLMToolCall.from_native_parts(
                call_id=str(tc_id or ""),
                name=str(name),
                arguments=args,
            )
        )
    return finalize_accepted_tool_call_identities(out)


def tool_calls_from_openai_dicts(items: Iterable[object]) -> tuple[LLMToolCall, ...]:
    """Convert accumulated OpenAI-style tool call dicts to typed calls."""
    out: list[LLMToolCall] = []
    for tc in items:
        if isinstance(tc, LLMToolCall):
            out.append(tc)
            continue
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        name = fn.get("name") or tc.get("name")
        if not name:
            continue
        out.append(
            LLMToolCall.from_native_parts(
                call_id=str(tc.get("id") or ""),
                name=str(name),
                arguments=fn.get("arguments") or tc.get("arguments"),
            )
        )
    return finalize_accepted_tool_call_identities(out)


def llm_tool_call_from_openai_shape(
    *,
    call_id: str,
    name: str,
    arguments: str | JsonObject | None,
) -> LLMToolCall:
    """Compatibility alias for callers still using the OpenAI-shaped factory name."""
    return LLMToolCall.from_native_parts(
        call_id=call_id,
        name=name,
        arguments=arguments,
    )
