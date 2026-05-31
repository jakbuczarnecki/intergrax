# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""
AWS Bedrock Converse API helpers (unified messages/tools interface).

Falls back to InvokeModel codecs when Converse is unavailable in the runtime client.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.messages import split_system_messages


def _to_converse_message(m: ChatMessage) -> Dict[str, Any]:
    if m.role == "user":
        return {"role": "user", "content": [{"text": m.content or ""}]}
    if m.role == "assistant":
        blocks: List[Dict[str, Any]] = []
        if m.content:
            blocks.append({"text": m.content})
        for tc in m.tool_calls or []:
            fn = tc.get("function") or {}
            blocks.append(
                {
                    "toolUse": {
                        "toolUseId": tc.get("id") or "",
                        "name": fn.get("name") or "",
                        "input": json.loads(fn.get("arguments") or "{}")
                        if isinstance(fn.get("arguments"), str)
                        else (fn.get("arguments") or {}),
                    }
                }
            )
        return {"role": "assistant", "content": blocks or [{"text": ""}]}
    if m.role == "tool":
        return {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": m.tool_call_id or "",
                        "content": [{"text": m.content or ""}],
                    }
                }
            ],
        }
    return {"role": "user", "content": [{"text": m.content or ""}]}


def build_converse_request(
    messages: Sequence[ChatMessage],
    *,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    system_text, convo = split_system_messages(messages)
    req: Dict[str, Any] = {
        "messages": [_to_converse_message(m) for m in convo if m.role != "system"],
    }
    if system_text:
        req["system"] = [{"text": system_text}]
    inference: Dict[str, Any] = {}
    if max_tokens is not None:
        inference["maxTokens"] = int(max_tokens)
    if temperature is not None:
        inference["temperature"] = float(temperature)
    if inference:
        req["inferenceConfig"] = inference
    if tools:
        req["toolConfig"] = {"tools": tools}
    return req


def extract_converse_text(response: Dict[str, Any]) -> str:
    output = response.get("output") or {}
    message = output.get("message") or {}
    parts: List[str] = []
    for block in message.get("content") or []:
        if isinstance(block, dict) and "text" in block:
            parts.append(str(block["text"]))
    return "".join(parts)


def extract_converse_tool_calls(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    output = response.get("output") or {}
    message = output.get("message") or {}
    out: List[Dict[str, Any]] = []
    for block in message.get("content") or []:
        if not isinstance(block, dict) or "toolUse" not in block:
            continue
        tu = block["toolUse"] or {}
        out.append(
            {
                "id": tu.get("toolUseId") or "",
                "type": "function",
                "function": {
                    "name": tu.get("name") or "",
                    "arguments": json.dumps(tu.get("input") or {}, ensure_ascii=False),
                },
            }
        )
    return out


def converse_supported(client: Any) -> bool:
    return callable(getattr(client, "converse", None))
