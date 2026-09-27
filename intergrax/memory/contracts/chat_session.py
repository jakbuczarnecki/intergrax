# © Artur Czarnecki. All rights reserved.

"""Chat session domain model (Memory / session contract surface)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from intergrax.llm.messages import AttachmentRef
from intergrax.memory.contracts.clock import utc_now


class SessionStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"


class SessionCloseReason(str, Enum):
    EXPLICIT = "explicit"
    TIMEOUT = "timeout"
    CONTEXT_SWITCH = "context_switch"
    UNKNOWN = "unknown"


@dataclass
class ChatSession:
    """Domain model for session metadata (messages live in session storage)."""

    id: str
    tenant_id: str
    user_id: str | None = None
    workspace_id: str | None = None
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    attachments: list[AttachmentRef] = field(default_factory=list)
    status: SessionStatus = SessionStatus.OPEN
    closed_reason: SessionCloseReason | None = None
    user_turns: int = 0
    last_consolidated_at: datetime | None = None
    last_consolidated_reason: str | None = None
    last_consolidated_turn: int | None = None
    user_profile_instructions: str | None = None
    org_profile_instructions: str | None = None
    needs_user_instructions_refresh: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        self.updated_at = utc_now()

    @property
    def is_closed(self) -> bool:
        return self.status == SessionStatus.CLOSED

    def mark_closed(self, reason: SessionCloseReason | None = None) -> None:
        self.status = SessionStatus.CLOSED
        self.closed_reason = reason or SessionCloseReason.EXPLICIT
        self.touch()

    def increment_user_turns(self) -> int:
        self.user_turns += 1
        self.touch()
        return self.user_turns
