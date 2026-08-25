# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from intergrax.utils import attribute_access

import json
from dataclasses import dataclass, replace
from typing import Any, Iterable, Sequence
from uuid import uuid4


_TOOL_CALL_ID_PREFIX = "toolcall-"


class ToolCallIdentityError(ValueError):
    """Accepted tool call is missing a stable non-empty identity."""


def _is_blank_tool_call_id(call_id: str) -> bool:
    return not call_id or not call_id.strip()


def mint_tool_call_id() -> str:
    """Mint a non-empty tool-call identity safe for logs and provenance."""
    return f"{_TOOL_CALL_ID_PREFIX}{uuid4().hex}"


def finalize_accepted_tool_call_identities(
    calls: Sequence[LLMToolCall],
) -> tuple[LLMToolCall, ...]:
    """Normalize accepted adapter output — sole minting owner for tool-call IDs."""
    seen: set[str] = set()
    finalized: list[LLMToolCall] = []
    for index, call in enumerate(calls):
        call_id = call.id
        if _is_blank_tool_call_id(call_id):
            while True:
                candidate = mint_tool_call_id()
                if candidate not in seen:
                    call_id = candidate
                    break
        elif call_id in seen:
            raise ToolCallIdentityError(
                f"duplicate tool call identity at index {index}: {call_id!r}"
            )
        seen.add(call_id)
        if call_id == call.id:
            finalized.append(call)
        else:
            finalized.append(replace(call, id=call_id))
    return tuple(finalized)


def validate_tool_call_identities(calls: Sequence[LLMToolCall]) -> None:
    """Fail closed when an invalid tool-call identity reaches planner/runtime."""
    seen: set[str] = set()
    for index, call in enumerate(calls):
        if _is_blank_tool_call_id(call.id):
            raise ToolCallIdentityError(
                f"tool call at index {index} has empty identity"
            )
        if call.id in seen:
            raise ToolCallIdentityError(
                f"duplicate tool call identity at index {index}: {call.id!r}"
            )
        seen.add(call.id)


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
    return finalize_accepted_tool_call_identities(out)


def merge_streaming_tool_calls(chunks: Sequence[LLMToolCall]) -> tuple[LLMToolCall, ...]:
    """Merge incremental streaming tool-call fragments by id."""
    by_key: dict[str, list[LLMToolCall]] = {}
    key_had_provider_id: dict[str, bool] = {}
    order: list[str] = []
    for tc in chunks:
        if not _is_blank_tool_call_id(tc.id):
            key = tc.id
            key_had_provider_id[key] = True
        else:
            key = tc.name or f"idx-{len(order)}"
            key_had_provider_id.setdefault(key, False)
        if key not in by_key:
            by_key[key] = []
            order.append(key)
        by_key[key].append(tc)
    merged: list[LLMToolCall] = []
    for key in order:
        parts = by_key[key]
        name = next((p.name for p in parts if p.name), "")
        args = "".join(p.arguments_json for p in parts)
        provider_call_id = key if key_had_provider_id.get(key) else ""
        merged.append(
            LLMToolCall.from_openai_shape(
                call_id=provider_call_id,
                name=name,
                arguments=args or "{}",
            )
        )
    return finalize_accepted_tool_call_identities(merged)


def tool_calls_from_langchain_message(message: Any) -> tuple[LLMToolCall, ...]:
    """Compatibility shim for the provider-local LangChain tool-call parser."""
    from intergrax.llm_adapters.providers._langchain_compat import (
        tool_calls_from_langchain_message as _parse_langchain_tool_calls,
    )

    return _parse_langchain_tool_calls(message)


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
    return finalize_accepted_tool_call_identities(out)
