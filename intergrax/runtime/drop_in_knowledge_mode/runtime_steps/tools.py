# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from typing import Any, Iterable, List, Optional

from intergrax.llm.messages import ChatMessage
from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_state import RuntimeState


def insert_context_before_last_user(        
        state: RuntimeState,
        context_messages: List[ChatMessage],
    ) -> None:
        """
        Insert context messages right before the last user message.
        If no user message exists, append to the end.
        """
        if not context_messages:
            return

        msgs = state.messages_for_llm
        last_user_idx: Optional[int] = None

        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].role == "user":
                last_user_idx = i
                break

        if last_user_idx is None:
            msgs.extend(context_messages)
        else:
            state.messages_for_llm = msgs[:last_user_idx] + context_messages + msgs[last_user_idx:]


def format_rag_context(chunks: Iterable[Any], *, max_chars: int = 4000) -> str:
    """
    Best-effort compact formatter that is resilient to different chunk shapes.
    """
    lines: List[str] = []
    total = 0

    for idx, ch in enumerate(chunks, start=1):
        text = _chunk_text(ch)
        meta = _chunk_meta(ch)

        header_parts: List[str] = [f"[{idx}]"]
        src = meta.get("source") or meta.get("url") or meta.get("doc_id") or meta.get("file")
        if src:
            header_parts.append(str(src))
        page = meta.get("page") or meta.get("page_number")
        if page is not None:
            header_parts.append(f"p={page}")

        header = " ".join(header_parts)
        block = header + "\n" + (text.strip() if text else "").strip()

        if not block.strip():
            continue

        # Enforce cap
        if total + len(block) + 2 > max_chars:
            remaining = max_chars - total
            if remaining > 80:
                lines.append(block[:remaining].rstrip() + "…")
            break

        lines.append(block)
        total += len(block) + 2

    return "\n\n".join(lines).strip()


def _chunk_text(ch: Any) -> str:
    for attr in ("text", "content", "page_content", "chunk", "value"):
        v = getattr(ch, attr, None)
        if isinstance(v, str) and v.strip():
            return v
    if isinstance(ch, dict):
        for k in ("text", "content", "page_content", "chunk", "value"):
            v = ch.get(k)
            if isinstance(v, str) and v.strip():
                return v
    return ""


def _chunk_meta(ch: Any) -> dict:
    meta = getattr(ch, "metadata", None)
    if isinstance(meta, dict):
        return meta
    if isinstance(ch, dict):
        m = ch.get("metadata")
        if isinstance(m, dict):
            return m
    return {}