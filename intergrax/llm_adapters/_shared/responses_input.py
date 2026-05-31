# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import json
from typing import Any, Dict, List


def messages_to_responses_input(mapped_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert Chat Completions-style message dicts to OpenAI Responses API input items.

    Supports multi-turn tool history: function_call + function_call_output items.
    """
    items: List[Dict[str, Any]] = []

    for m in mapped_messages:
        role = m.get("role", "user")

        if role == "assistant" and m.get("tool_calls"):
            if m.get("content"):
                items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": m.get("content", ""),
                    }
                )
            for tc in m["tool_calls"]:
                fn = tc.get("function") or {}
                args = fn.get("arguments") or "{}"
                if not isinstance(args, str):
                    args = json.dumps(args, ensure_ascii=False)
                items.append(
                    {
                        "type": "function_call",
                        "call_id": tc.get("id") or "",
                        "name": fn.get("name") or "",
                        "arguments": args,
                    }
                )
            continue

        if role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": m.get("tool_call_id") or "",
                    "output": m.get("content") or "",
                }
            )
            continue

        items.append(
            {
                "type": "message",
                "role": role if role in ("user", "assistant", "system") else "user",
                "content": m.get("content") or "",
            }
        )

    return items
