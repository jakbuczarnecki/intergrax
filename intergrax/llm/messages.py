# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

MessageRole = Literal["system", "user", "assistant", "tool"]


@dataclass
class AttachmentRef:
    """
    Lightweight reference to an attachment associated with a message or session.

    The actual binary content is stored elsewhere (filesystem, object storage,
    database BLOB, etc.). Here we only keep stable identifiers and metadata.
    """

    id: str
    type: str  # e.g. "pdf", "docx", "image", "audio", "video", "code", "json"
    uri: str   # e.g. "file://...", "s3://...", "db://attachments/<id>"
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# ChatMessage – extended with tool_calls and to_dict() method
# ============================================================
@dataclass
class ChatMessage:
    """
    Universal chat message compatible with the OpenAI Responses API.
    Supports fields:
      - tool_call_id  → for single tool calls (from field 'id'),
      - tool_calls    → list of calls (for assistant.tool_calls).
    """

    role: MessageRole
    content: str

    entry_id: Optional[int] = None
    deleted: bool = False
    modified: bool = False    
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[dict]] = None
    attachments: List[AttachmentRef] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """
        Converts the object to a dict compatible with OpenAI Responses API / ChatCompletions.
        """
        msg = {
            "role": self.role,
            "content": self.content,
        }
        if self.name:
            msg["name"] = self.name
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        return msg

    def __repr__(self):
        extras = []
        if self.name:
            extras.append(f"name={self.name}")
        if self.tool_call_id:
            extras.append(f"tool_call_id={self.tool_call_id}")
        if self.tool_calls:
            extras.append(f"tool_calls={len(self.tool_calls)}")
        extras_str = ", ".join(extras)
        return f"<ChatMessage role={self.role} {extras_str}>"


def append_chat_messages(
        existing: Optional[List[ChatMessage]],
        new: List[ChatMessage]
) -> List[ChatMessage]:
    """
    Custom reducer for LangGraph state.

    LangGraph calls this function when merging state updates:
      - `existing`: the current list of messages in the state (may be None)
      - `new`: the list of messages provided by a node update

    We simply append the new messages to the existing ones.
    """
    if existing is None:
        return list(new)
    return [*existing, *new]
