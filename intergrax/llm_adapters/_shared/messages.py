# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from intergrax.llm.messages import AttachmentRef, ChatMessage
from intergrax.llm_adapters._shared.multimodal_content import (
    AttachmentContentMapper,
    DefaultAttachmentContentMapper,
    build_openai_user_content,
)


def split_system_messages(
    messages: Sequence[ChatMessage],
) -> Tuple[str, List[ChatMessage]]:
    """Extract concatenated system prompt and non-system conversation."""
    system_parts: List[str] = []
    convo: List[ChatMessage] = []
    for m in messages:
        if m.role == "system":
            if m.content:
                system_parts.append(m.content)
            continue
        convo.append(m)
    return ("\n\n".join(system_parts).strip(), convo)


def map_chat_completion_messages(
    *,
    system_text: str,
    convo: Sequence[ChatMessage],
    include_multimodal: bool = False,
    attachment_mapper: AttachmentContentMapper | None = None,
) -> List[Dict[str, Any]]:
    """
    Map ChatMessage list to OpenAI Chat Completions message dicts.

    Preserves tool_call_id, name, and assistant tool_calls for multi-turn tool loops.
    """
    out: List[Dict[str, Any]] = []
    mapper = attachment_mapper or DefaultAttachmentContentMapper()
    if system_text:
        out.append({"role": "system", "content": system_text})

    for m in convo:
        if m.role == "assistant":
            d: Dict[str, Any] = {"role": "assistant"}
            if m.content:
                d["content"] = m.content
            if m.tool_calls:
                d["tool_calls"] = m.tool_calls
            if d.get("content") is not None or d.get("tool_calls"):
                out.append(d)
            continue

        if m.role == "tool":
            d = {"role": "tool", "content": m.content or ""}
            if m.tool_call_id:
                d["tool_call_id"] = m.tool_call_id
            if m.name:
                d["name"] = m.name
            out.append(d)
            continue

        if m.role == "user" and include_multimodal and m.attachments:
            out.append(
                {
                    "role": "user",
                    "content": build_openai_user_content(
                        text=m.content or "",
                        attachments=list(m.attachments),
                        mapper=mapper,
                    ),
                }
            )
            continue

        if not m.content and m.role != "user":
            continue

        out.append({"role": m.role, "content": m.content or ""})

    return out
