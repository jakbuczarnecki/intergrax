# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Any

from intergrax.llm_adapters.contracts.tool_call import LLMToolCall


def tool_calls_from_langchain_message(message: Any) -> tuple[LLMToolCall, ...]:
    """Extract typed tool calls from a LangChain AIMessage or compatible object."""
    raw = getattr(message, "tool_calls", None) or []
    out: list[LLMToolCall] = []
    for tool_call in raw:
        if isinstance(tool_call, dict):
            name = tool_call.get("name")
            args = tool_call.get("args")
            call_id = tool_call.get("id")
        else:
            name = getattr(tool_call, "name", None)
            args = getattr(tool_call, "args", None)
            call_id = getattr(tool_call, "id", None)
        if not name or not str(name).strip():
            continue
        if args is not None and not isinstance(args, (dict, str)):
            raise ValueError("langchain tool call args must be a dictionary or JSON string")
        out.append(
            LLMToolCall.from_openai_shape(
                call_id=str(call_id or ""),
                name=str(name),
                arguments=args if isinstance(args, (dict, str)) else {},
            )
        )
    return tuple(out)
