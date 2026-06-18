# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from intergrax.utils import attribute_access

import json
from typing import Any, Dict, List, Sequence

from intergrax.llm.messages import ChatMessage


def map_anthropic_messages(msgs: Sequence[ChatMessage]) -> List[Dict[str, Any]]:
    """
    Map ChatMessage list to Anthropic Messages API content blocks.

    Handles assistant tool_use and user tool_result blocks for multi-turn tools.
    """
    out: List[Dict[str, Any]] = []

    for m in msgs:
        if m.role == "user":
            blocks: List[Dict[str, Any]] = []
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            if blocks:
                out.append({"role": "user", "content": blocks})
            continue

        if m.role == "assistant":
            blocks = []
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            for tc in m.tool_calls or []:
                fn = tc.get("function") or {}
                args_raw = fn.get("arguments") or "{}"
                try:
                    args_obj = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except json.JSONDecodeError:
                    args_obj = {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id") or "",
                        "name": fn.get("name") or "",
                        "input": args_obj if isinstance(args_obj, dict) else {},
                    }
                )
            if blocks:
                out.append({"role": "assistant", "content": blocks})
            continue

        if m.role == "tool":
            out.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": m.tool_call_id or "",
                            "content": m.content or "",
                        }
                    ],
                }
            )

    return out


def extract_anthropic_tool_calls(response: Any) -> List[Dict[str, Any]]:
    """Extract OpenAI-compatible tool_calls from an Anthropic Message response."""
    out: List[Dict[str, Any]] = []
    for block in attribute_access.optional(response, "content", None) or []:
        if attribute_access.optional(block, "type", None) != "tool_use":
            continue
        out.append(
            {
                "id": attribute_access.optional(block, "id", "") or "",
                "type": "function",
                "function": {
                    "name": attribute_access.optional(block, "name", "") or "",
                    "arguments": json.dumps(attribute_access.optional(block, "input", None) or {}, ensure_ascii=False),
                },
            }
        )
    return out


def extract_anthropic_text(response: Any) -> str:
    parts: List[str] = []
    for block in attribute_access.optional(response, "content", None) or []:
        if attribute_access.optional(block, "type", None) == "text":
            parts.append(attribute_access.optional(block, "text", "") or "")
    return "".join(parts)
