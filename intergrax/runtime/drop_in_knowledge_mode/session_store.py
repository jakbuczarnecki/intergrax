# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Session storage and memory integration component for the Intergrax
Drop-In Knowledge Runtime.

This module defines:
  - ChatSession: a lightweight metadata container for a chat session.
  - SessionStore: a unified memory component responsible for:
        * managing session lifecycle,
        * storing and retrieving conversation history via
          IntergraxConversationalMemory,
        * building LLM-ready message context.

Design principles:
  - The engine sees SessionStore as a single "memory black box".
  - SessionStore consolidates all memory layers internally (now only
    conversational memory; later: user profile, org profile, long-term
    memory, preferences, embeddings, etc.).
  - The API exposed to the engine remains simple and stable ("get and use").

Current status:
  - Fully in-memory implementation (not production-ready).
  - Conversational memory is the single source of truth for message history.
  - No persistence backend, no long-term memory yet.

Future extensions will plug into this same SessionStore without changing
the engine architecture.
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
    Lightweight metadata container representing an ongoing chat session.

    Important:
      This object does NOT store the messages. The single source of truth
      for message history is managed by the SessionStore using
      IntergraxConversationalMemory (or future persistent memory backends).
    """

    id: str
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    workspace_id: Optional[str] = None

    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Optional session-level attachments (not tied to specific messages)
    attachments: List[AttachmentRef] = field(default_factory=list)

    # Arbitrary metadata (e.g., runtime labels, task context, custom settings)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        """Update modification timestamp."""
        self.updated_at = datetime.now(timezone.utc)


class SessionStore:
    """
    Unified memory component used by the Drop-In Knowledge Runtime.

    Responsibilities:
      - Create and manage chat sessions.
      - Store and retrieve per-session message history using
        IntergraxConversationalMemory.
      - Provide LLM-ready conversation history with trimming already applied.

    Future responsibility expansion (internal, NOT affecting engine API):
      - User profile memory (preferences, tone, constraints).
      - Organization-level memory (context, knowledge policies).
      - Long-term semantic memory (extracted facts, embeddings, indexing).
      - Memory persistence (SQLite/Postgres/Redis/custom providers).

    Engine should never know *how* memory works — only that:
       session_store.build_conversational_history(session) → List[ChatMessage].
    """

    def __init__(
        self,
        *,
        max_history_messages: int = 200,
    ) -> None:
        # Session metadata storage (no message history here).
        self._sessions: Dict[str, ChatSession] = {}

        # Per-session conversational memory storage.
        self._conv_memory: Dict[str, IntergraxConversationalMemory] = {}

        # Default maximum message count before trimming occurs.
        self._max_history_messages = max_history_messages

        # Placeholder for future memory layers:
        # self._user_profile_memory = ...
        # self._org_profile_memory = ...
        # self._long_term_memory = ...

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Return the session metadata if it exists, otherwise None."""
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
        Create and register a new chat session.
        Also initializes a conversational memory buffer for that session.
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

        # Initialize conversational memory backing this session.
        self._conv_memory[session_id] = IntergraxConversationalMemory(
            session_id=session_id,
            max_messages=self._max_history_messages,
        )

        return session

    async def save_session(self, session: ChatSession) -> None:
        """
        Persist session metadata.
        In-memory implementation only updates the timestamp and replaces the reference.
        """
        session.touch()
        self._sessions[session.id] = session

    # ------------------------------------------------------------------
    # Message and memory operations
    # ------------------------------------------------------------------

    async def append_message(
        self,
        session_id: str,
        message: ChatMessage,
    ) -> None:
        """
        Append a message to the session's memory source of truth.

        - Updates conversational memory
        - Refreshes session timestamp
        """
        session = self._sessions.get(session_id)
        if not session:
            raise KeyError(f"Session '{session_id}' does not exist")

        memory = self._conv_memory.get(session_id)
        if not memory:
            # Safety fallback (should not occur)
            memory = IntergraxConversationalMemory(
                session_id=session_id,
                max_messages=self._max_history_messages,
            )
            self._conv_memory[session_id] = memory

        memory.add_message(message)

        session.touch()
        self._sessions[session_id] = session

    async def list_sessions_for_user(
        self,
        user_id: str,
        limit: int = 50,
    ) -> List[ChatSession]:
        """
        Return user sessions ordered by most recent activity.
        """
        result = [s for s in self._sessions.values() if s.user_id == user_id]
        result.sort(key=lambda s: s.updated_at, reverse=True)
        return result[:limit]

    # ------------------------------------------------------------------
    # Conversation history builder for the engine
    # ------------------------------------------------------------------

    def build_conversational_history(
        self,
        session: ChatSession
    ) -> List[ChatMessage]:
        """
        Return the LLM-ready message history for a session.

        Notes:
        - Trimming is handled by IntergraxConversationalMemory.
        - `max_history_messages` argument is retained for future extensibility
          (e.g. persistent database stores may use it to limit retrieval).
        - Currently `native_tools=False` because we are not yet injecting
          OpenAI Responses native tool call metadata into the memory structure.
        """
        memory = self._conv_memory.get(session.id)
        if not memory:
            return []

        return memory.get_for_model(native_tools=False)
