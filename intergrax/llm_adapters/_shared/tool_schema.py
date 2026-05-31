# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import json
from typing import Any, Dict, List


def openai_tools_to_anthropic(tools_schema: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert OpenAI function tools schema to Anthropic tools format."""
    out: List[Dict[str, Any]] = []
    for t in tools_schema:
        fn = t.get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        out.append(
            {
                "name": name,
                "description": fn.get("description") or "",
                "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
            }
        )
    return out


def openai_tools_to_gemini(tools_schema: List[Dict[str, Any]]) -> List[Any]:
    """Convert OpenAI tools schema to google.genai Tool list."""
    from google.genai import types

    declarations: List[Any] = []
    for t in tools_schema:
        fn = t.get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        declarations.append(
            types.FunctionDeclaration(
                name=name,
                description=fn.get("description") or "",
                parameters=fn.get("parameters") or {"type": "object", "properties": {}},
            )
        )
    if not declarations:
        return []
    return [types.Tool(function_declarations=declarations)]


def extract_openai_tool_calls(message: Any) -> List[Dict[str, Any]]:
    """Extract tool_calls from an OpenAI-style chat completion message object."""
    raw = getattr(message, "tool_calls", None) or []
    out: List[Dict[str, Any]] = []
    for tc in raw:
        fn = getattr(tc, "function", None)
        if fn is None and isinstance(tc, dict):
            fn = tc.get("function")
        name = getattr(fn, "name", None) if fn is not None else None
        args = getattr(fn, "arguments", None) if fn is not None else None
        if name is None and isinstance(fn, dict):
            name = fn.get("name")
            args = fn.get("arguments")
        tc_id = getattr(tc, "id", None) or (tc.get("id") if isinstance(tc, dict) else None)
        if not name:
            continue
        out.append(
            {
                "id": tc_id or "",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": args if isinstance(args, str) else json.dumps(args or {}, ensure_ascii=False),
                },
            }
        )
    return out
