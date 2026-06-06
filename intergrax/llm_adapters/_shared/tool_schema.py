# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Any, Dict, List

from intergrax.llm_adapters.contracts.tool_call import LLMToolCall, tool_calls_from_openai_message


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


def openai_tools_to_bedrock_converse(tools_schema: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert OpenAI tools schema to Bedrock Converse ``toolConfig.tools`` entries."""
    out: List[Dict[str, Any]] = []
    for t in tools_schema:
        fn = t.get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        out.append(
            {
                "toolSpec": {
                    "name": name,
                    "description": fn.get("description") or "",
                    "inputSchema": {
                        "json": fn.get("parameters") or {"type": "object", "properties": {}},
                    },
                }
            }
        )
    return out


def extract_openai_tool_calls(message: Any) -> tuple[LLMToolCall, ...]:
    """Extract typed tool_calls from an OpenAI-style chat completion message object."""
    return tool_calls_from_openai_message(message)
