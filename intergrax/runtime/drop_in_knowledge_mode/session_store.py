# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Unified memory component for the Intergrax Drop-In Knowledge Runtime.

This module defines:
  - ChatSession: a lightweight metadata model for a chat session.
  - SessionStore: a central component responsible for:
        * managing session lifecycle,
        * storing and retrieving conversational history using
          IntergraxConversationalMemory,
        * producing LLM-ready message context for inference.

Design principles:
  - The engine interacts only with SessionStore (not with any internal memory
    components such as conversational memory, profiles, or persistent storage).
  - SessionStore consolidates all memory layers internally.
  - The public API remains simple and stable ("get and use").

Current implementation:
  - Fully in-memory (not yet persistent).
  - Conversational memory is the single source of truth for message history.
  - User profile memory is provided as a simple in-memory map (not yet used
    by the engine for prompt injection).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional
import uuid

from intergrax.llm.messages import (
    AttachmentRef,
    ChatMessage,
)
from intergrax.llm.conversational_memory import IntergraxConversationalMemory


@dataclass
class ChatSession:
    """
    Metadata describing a chat session.

    Important:
      This object does NOT store messages. The single source of truth for
      conversation history is held by the SessionStore using
      IntergraxConversationalMemory or future backends.
    """

    id: str
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    workspace_id: Optional[str] = None

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Optional per-session attachments (not tied directly to a single message)
    attachments: List[AttachmentRef] = field(default_factory=list)

    # Arbitrary metadata (could contain tags, app-specific values, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        """Refresh modification timestamp."""
        self.updated_at = datetime.now(timezone.utc)


class SessionStore:
    """
    The primary memory backbone for the runtime.

    Responsibilities:
      - Create and manage chat sessions.
      - Maintain conversational message history per session.
      - Return an LLM-ready ordered list of messages representing the session context.

    Notes:
      This class intentionally hides implementation details of memory sources
      (e.g., conversation history, user profiles, persistent stores, vector memory)
      from the engine.

    Future expansion:
      - user profile memory (preferences, language style, tone),
      - organization-level memory,
      - long-term semantic memory (facts, embeddings),
      - persistence layer adapters (SQLite, PostgreSQL, Redis, Supabase).
    """

    def __init__(self, *, max_history_messages: int = 200) -> None:
        # Metadata storage (chat sessions registry)
        self._sessions: Dict[str, ChatSession] = {}

        # Internal conversational memory storage (one per session)
        self._conv_memory: Dict[str, IntergraxConversationalMemory] = {}

        # Maximum number of messages to keep before trimming FIFO-style
        self._max_history_messages = max_history_messages

        # Simple in-memory user profile storage:
        #   user_id -> { "key": value, ... }
        # This is the first step towards user profile memory. It is not yet
        # integrated into the engine prompt, but provides a clear place
        # to attach and evolve profile data.
        self._user_profiles: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Return the session metadata if it exists, else None."""
        return self._sessions.get(session_id)

    async def create_session(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ChatSession:
        """
        Create a new chat session and initialize its conversational memory.
        """
        session_id = session_id or str(uuid.uuid4())

        session = ChatSession(
            id=session_id,
            user_id=user_id,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            metadata=dict(metadata) if metadata else {},
        )
        self._sessions[session_id] = session

        # Initialize the backing conversation memory for this session
        self._conv_memory[session_id] = IntergraxConversationalMemory(
            session_id=session_id,
            max_messages=self._max_history_messages,
        )

        return session

    async def save_session(self, session: ChatSession) -> None:
        """
        Persist the session metadata.
        (In-memory version simply updates timestamps and replaces the entry.)
        """
        session.touch()
        self._sessions[session.id] = session

    # ------------------------------------------------------------------
    # Message operations
    # ------------------------------------------------------------------

    async def append_message(self, session_id: str, message: ChatMessage) -> None:
        """
        Append a chat message to the session's conversational history.
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session '{session_id}' does not exist")

        memory = self._conv_memory.get(session_id)

        # Safety fallback (should not occur)
        if memory is None:
            memory = IntergraxConversationalMemory(
                session_id=session_id,
                max_messages=self._max_history_messages,
            )
            self._conv_memory[session_id] = memory

        memory.add_message(message)

        session.touch()
        self._sessions[session_id] = session

    async def list_sessions_for_user(self, user_id: str, limit: int = 50) -> List[ChatSession]:
        """
        Return a list of sessions owned by a user, sorted by recent activity.
        """
        sessions = [s for s in self._sessions.values() if s.user_id == user_id]
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions[:limit]

    # ------------------------------------------------------------------
    # User profile memory (first step)
    # ------------------------------------------------------------------

    async def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Return a shallow copy of the user's profile dictionary.

        This is intentionally generic (Dict[str, Any]) to support:
          - language/tone preferences,
          - domain expertise flags,
          - feature flags,
          - custom per-user settings.

        Engine does not call this directly yet; it will be used as an
        internal building block when injecting profile-based system messages.
        """
        profile = self._user_profiles.get(user_id, {})
        # Return a copy to avoid accidental external mutation.
        return dict(profile)

    async def upsert_user_profile(
        self,
        user_id: str,
        updates: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Merge the provided updates into the user's profile and return
        the resulting profile.

        Example usage:
          await session_store.upsert_user_profile("u1", {
              "preferred_language": "pl",
              "tone": "technical",
              "no_emojis_in_code": True,
          })
        """
        current = self._user_profiles.get(user_id, {})
        merged = dict(current)
        merged.update(dict(updates))
        self._user_profiles[user_id] = merged
        return merged

    # ------------------------------------------------------------------
    # Conversation context builder
    # ------------------------------------------------------------------

    def get_conversation_history(self, session: ChatSession) -> List[ChatMessage]:
        """
        Return an ordered list of ChatMessage objects representing the
        conversation history for the given session.

        Trimming logic (max history size, FIFO) is handled internally by
        the backing conversational memory implementation.
        """
        memory = self._conv_memory.get(session.id)
        if memory is None:
            return []

        # Still uses IntergraxConversationalMemory under the hood,
        # but SessionStore does not talk about LLMs or prompts.
        return memory.get_for_model(native_tools=False)

