# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from intergrax.utils import attribute_access

import json
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


@dataclass(frozen=True, slots=True)
class LLMToolCall:
    """Typed native tool call returned by an LLM adapter."""

    id: str
    name: str
    arguments_json: str

    @classmethod
    def from_openai_shape(
        cls,
        *,
        call_id: str,
        name: str,
        arguments: str | dict[str, Any] | None,
    ) -> LLMToolCall:
        if isinstance(arguments, str):
            args_json = arguments or "{}"
        else:
            args_json = json.dumps(arguments or {}, ensure_ascii=False)
        return cls(
            id=str(call_id or ""),
            name=str(name or ""),
            arguments_json=args_json,
        )


def tool_calls_from_openai_message(message: Any) -> tuple[LLMToolCall, ...]:
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
        tc_id = attribute_access.optional(tc, "id", None) or (tc.get("id") if isinstance(tc, dict) else None)
        if not name:
            continue
        out.append(
            LLMToolCall.from_openai_shape(
                call_id=str(tc_id or ""),
                name=str(name),
                arguments=args,
            )
        )
    return tuple(out)


def merge_streaming_tool_calls(chunks: Sequence[LLMToolCall]) -> tuple[LLMToolCall, ...]:
    """Merge incremental streaming tool-call fragments by id."""
    by_id: dict[str, list[LLMToolCall]] = {}
    order: list[str] = []
    for tc in chunks:
        key = tc.id or tc.name or f"idx-{len(order)}"
        if key not in by_id:
            by_id[key] = []
            order.append(key)
        by_id[key].append(tc)
    merged: list[LLMToolCall] = []
    for key in order:
        parts = by_id[key]
        name = next((p.name for p in parts if p.name), "")
        args = "".join(p.arguments_json for p in parts)
        merged.append(
            LLMToolCall.from_openai_shape(
                call_id=key,
                name=name,
                arguments=args or "{}",
            )
        )
    return tuple(merged)


def tool_calls_from_openai_dicts(items: Iterable[Any]) -> tuple[LLMToolCall, ...]:
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
            LLMToolCall.from_openai_shape(
                call_id=str(tc.get("id") or ""),
                name=str(name),
                arguments=fn.get("arguments") or tc.get("arguments"),
            )
        )
    return tuple(out)
