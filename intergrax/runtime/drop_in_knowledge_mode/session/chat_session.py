# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from intergrax.llm.messages import AttachmentRef

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
        return self.metadata.get("status") == "closed"

    def mark_closed(self, reason: Optional[str] = None) -> None:
        """
        Mark this session as closed at the domain level.

        This method does not persist changes. The caller is responsible
        for saving the session via the session manager / storage.
        """
        self.metadata["status"] = "closed"
        if reason:
            self.metadata["closed_reason"] = reason
        self.touch()

    def increment_user_turns(self) -> int:
        """
        Increment and return the per-session counter of user turns.

        This is useful for heuristics like "synthesize memory every N user
        messages". The counter is stored in metadata["user_turns"].

        This method updates in-memory state only. Persistence is the
        responsibility of the manager / storage layer.
        """
        current = int(self.metadata.get("user_turns", 0) or 0)
        current += 1
        self.metadata["user_turns"] = current
        self.touch()
        return current