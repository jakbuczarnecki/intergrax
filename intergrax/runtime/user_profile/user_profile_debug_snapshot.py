# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from intergrax.memory.user_profile_memory import (
    UserProfileMemoryEntry,
    MemoryKind,
)
from intergrax.runtime.drop_in_knowledge_mode.session.chat_session import ChatSession

UNKNOWN:str = "UNKNOWN"

@dataclass
class SessionDebugView:
    """
    Lightweight, debug-friendly view of a single ChatSession.

    This structure is intentionally small and only contains fields that
    are useful for understanding how this user's sessions behave in
    the context of long-term memory and system instructions.
    """

    session_id: str
    status: str
    user_turns: int
    last_consolidated_at: Optional[datetime]
    last_consolidated_reason: Optional[str]
    last_consolidated_turn: Optional[int]


@dataclass
class MemoryEntryDebugView:
    """
    Lightweight, debug-friendly view of a single UserProfileMemoryEntry.

    Only exposes the most relevant fields for inspection and UI display.
    The full UserProfileMemoryEntry remains owned by the memory layer.
    """

    entry_id: Optional[int]
    kind: str
    title: Optional[str]
    content: str
    importance: str
    session_id: Optional[str]
    created_at: Optional[datetime]
    modified: bool
    deleted: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserProfileDebugSnapshot:
    """
    Immutable snapshot of user profile state for debugging and observability.

    This model does NOT introduce any new behavior in the runtime. It is
    a read-only view that can be constructed by a higher-level service
    (e.g. UserProfileDebugService) from:

      - UserProfile (identity, preferences, system_instructions, memory),
      - SessionManager / SessionStorage (recent sessions metadata).

    Typical usage:
      - exposing a "debug user profile" API endpoint,
      - rendering an inspector panel in a UI,
      - logging / tracing long-term memory behavior.
    """

    # Core identifiers
    user_id: str

    # Basic identity and preferences flattened into dicts so we do not depend
    # on concrete domain models in the debug API surface.
    identity: Dict[str, Any]
    preferences: Dict[str, Any]

    # Current user-level system instructions (as stored on the profile).
    system_instructions: Optional[str]

    # Memory statistics
    memory_entries_total: int
    memory_entries_by_kind: Dict[str, int]

    # Recent memory entries (e.g. latest N items) in a compact view.
    recent_memory_entries: List[MemoryEntryDebugView] = field(default_factory=list)

    # Recent sessions related to this user (e.g. latest N sessions).
    recent_sessions: List[SessionDebugView] = field(default_factory=list)

    # Timestamp when this snapshot was generated (UTC).
    generated_at: datetime = field(default_factory=datetime.utcnow)

    # Optional bag for future extensions (debug-only metadata, feature flags, etc.).
    extra: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def build_memory_kind_counters(
        entries: List[UserProfileMemoryEntry],
    ) -> Dict[str, int]:
        """
        Helper that can be used by the debug service to calculate how many
        entries exist per MemoryKind.

        This method is purely computational and does not touch storage.
        """
        counters: Dict[str, int] = {}

        for entry in entries:
            # We treat the enum name as the canonical string representation.
            kind_name = UNKNOWN
            if isinstance(entry.kind, MemoryKind):
                kind_name = entry.kind.name

            current = counters.get(kind_name, 0)
            counters[kind_name] = current + 1

        return counters

    @staticmethod
    def from_domain_session(session: ChatSession) -> SessionDebugView:
        """
        Build a SessionDebugView from a ChatSession domain object.

        This method is intended to be used by a higher-level debug service
        when assembling the snapshot. It does not perform any I/O.
        """
        return SessionDebugView(
            session_id=session.id,
            status=session.status.name if session.status is not None else UNKNOWN,
            user_turns=session.user_turns,
            last_consolidated_at=session.last_consolidated_at,
            last_consolidated_reason=session.last_consolidated_reason,
            last_consolidated_turn=session.last_consolidated_turn,
        )

    @staticmethod
    def from_memory_entry(
        entry: UserProfileMemoryEntry,
    ) -> MemoryEntryDebugView:
        """
        Build a MemoryEntryDebugView from a UserProfileMemoryEntry.

        This keeps the debug surface stable even if the underlying domain
        model grows additional fields in the future.
        """
        importance_name = entry.importance.name if entry.importance is not None else UNKNOWN
        kind_name = entry.kind.name if entry.kind is not None else UNKNOWN

        return MemoryEntryDebugView(
            entry_id=entry.entry_id,
            kind=kind_name,
            title=entry.title,
            content=entry.content,
            importance=importance_name,
            session_id=entry.session_id,
            created_at=entry.created_at,
            modified=entry.modified,
            deleted=entry.deleted,
            metadata=entry.metadata or {},
        )
