# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from intergrax.llm.messages import AttachmentRef
from enum import Enum


class SessionStatus(str, Enum):
    """
    Domain-level status of a chat session.

    Kept as `str` + `Enum` so that the value is JSON-serializable and
    safe to store in DB or metadata.

    Typical lifecycle:
      - OPEN   → session is active and can receive messages
      - CLOSED → session was finalized and should not be appended to
    """

    OPEN = "open"
    CLOSED = "closed"


class SessionCloseReason(str, Enum):
    """
    Optional domain-level reason for closing a chat session.

    This is intentionally minimal for now; additional categories can
    be added when the domain evolves (e.g. timeout, user_logout, etc.)
    """

    # The session was explicitly closed (default)
    EXPLICIT = "explicit"

    # Could be triggered by inactivity / timeout (future use)
    TIMEOUT = "timeout"

    # Session closed because the tenant/user context changed (future use)
    CONTEXT_SWITCH = "context_switch"

    # Generic fallback reason
    UNKNOWN = "unknown"


@dataclass
class ChatSession:
    """
    Domain model describing a single chat session.

    Important:
      - This object does NOT store messages. The single source of truth for
        conversation history is maintained by session-level storage
        (e.g. ConversationalMemory, database, Redis, etc.).
      - This model is intentionally I/O-free. It should not talk directly
        to any storage backend. All persistence is handled by a manager
        or storage component above it.
    """

    # Stable session identifier used throughout the runtime.
    id: str

    # Optional identifiers for user, tenant and workspace.
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None     # can be used as organization/tenant identifier
    workspace_id: Optional[str] = None  # workspace/project/context within a tenant

    # Timestamps for auditing and retention policies.
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Optional per-session attachments (not tied directly to a single message).
    attachments: List[AttachmentRef] = field(default_factory=list)

    # Core domain state (typed)
    status: SessionStatus = SessionStatus.OPEN  # "open" / "closed" – można kiedyś zamienić na Enum
    closed_reason: Optional[SessionCloseReason] = None

    user_turns: int = 0

    # Consolidation-related state
    last_consolidated_at: Optional[datetime] = None
    last_consolidated_reason: Optional[str] = None  # wartości z SessionConsolidationReason.value
    last_consolidated_turn: Optional[int] = None
    last_consolidation_debug: Optional[Dict[str, Any]] = None

    # Per-session instructions snapshot + refresh flag
    user_profile_instructions: Optional[str] = None
    org_profile_instructions: Optional[str] = None
    needs_user_instructions_refresh: bool = False

    # Arbitrary metadata (could contain tags, profile instruction cache, counters, etc.).
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Domain helpers (no I/O)
    # ------------------------------------------------------------------

    def touch(self) -> None:
        """
        Refresh modification timestamp.

        Managers / storage components should call this before/after
        mutating the session, but this method itself does not persist
        anything.
        """
        self.updated_at = datetime.now(timezone.utc)

    @property
    def is_closed(self) -> bool:
        """
        Return True if this session is marked as closed at the domain level.
        """
        return self.status == SessionStatus.CLOSED

    def mark_closed(self, reason: SessionCloseReason = None) -> None:
        """
        Mark this session as closed at the domain level.

        This method does not persist changes. The caller is responsible
        for saving the session via the session manager / storage.
        """
        self.status = SessionStatus.CLOSED        
        self.closed_reason = reason or SessionCloseReason.EXPLICIT

        self.touch()

    def increment_user_turns(self) -> int:
        """
        Increment and return the per-session counter of user turns.

        This is useful for heuristics like "synthesize memory every N user
        messages". The counter is stored in user_turns.

        This method updates in-memory state only. Persistence is the
        responsibility of the manager / storage layer.
        """        
        self.user_turns += 1
        self.touch()
        return self.user_turns