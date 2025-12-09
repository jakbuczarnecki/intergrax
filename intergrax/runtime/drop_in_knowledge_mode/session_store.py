# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Unified memory component for the Intergrax Drop-In Knowledge Runtime.

This module defines:
  - ChatSession: a lightweight metadata model for a chat session.
  - SessionStore: a central component responsible for:
        * managing session lifecycle,
        * storing and retrieving conversational history,
        * exposing profile and long-term memory bundles for the runtime,
        * producing LLM-ready message context for inference (conversation history).

Design principles:
  - The engine interacts only with SessionStore (not with any internal memory
    components such as conversational memory, profiles, or persistent storage).
  - SessionStore consolidates all memory layers internally.
  - The public API remains simple and stable ("get and use").

Current implementation:
  - Fully in-memory (not yet persistent) for conversational history and sessions.
  - User and organization profiles, as well as long-term memory, are accessed
    via dedicated managers injected into this store.
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
from intergrax.memory.conversational_memory import ConversationalMemory
from intergrax.memory.user_profile_manager import UserProfileManager
from intergrax.runtime.organization.organization_profile_manager import OrganizationProfileManager


@dataclass
class ChatSession:
    """
    Metadata describing a chat session.

    Important:
      This object does NOT store messages. The single source of truth for
      conversation history is held by the SessionStore using ConversationalMemory
      or future backends.
    """

    id: str
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None   # can be used as organization/tenant identifier
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
      - Expose user/organization profile bundles and long-term memory context.
      - Return an LLM-ready ordered list of messages representing the session context.

    Notes:
      This class intentionally hides implementation details of memory sources
      (e.g., conversation history, user profiles, persistent stores, vector memory)
      from the engine.

    Future expansion:
      - persistent storage for sessions and conversational history,
      - deeper integration with long-term semantic memory (facts, embeddings),
      - tighter coupling with ContextBuilder and RAG components.
    """

    def __init__(
        self,
        *,
        max_history_messages: int = 200,
        user_profile_manager: Optional[UserProfileManager] = None,
        organization_profile_manager: Optional[OrganizationProfileManager] = None
    ) -> None:
        # Metadata storage (chat sessions registry)
        self._sessions: Dict[str, ChatSession] = {}

        # Internal conversational memory storage (one per session)
        self._conv_memory: Dict[str, ConversationalMemory] = {}

        # Maximum number of messages to keep before trimming FIFO-style
        self._max_history_messages = max_history_messages

        # High-level managers for other memory types.
        # These are optional to keep the SessionStore usable in minimal setups.
        self._user_profile_manager = user_profile_manager
        self._organization_profile_manager = organization_profile_manager

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
        self._conv_memory[session_id] = ConversationalMemory(
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
            memory = ConversationalMemory(
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
    # Conversation context builder
    # ------------------------------------------------------------------

    def get_conversation_history(self, session: ChatSession, *, native_tools: bool = False) -> List[ChatMessage]:
        """
        Return an ordered list of ChatMessage objects representing the
        conversation history for the given session.

        Trimming logic (max history size, FIFO) is handled internally by
        the backing conversational memory implementation.
        """
        memory = self._conv_memory.get(session.id)
        if memory is None:
            return []

        # Still uses ConversationalMemory under the hood,
        # but SessionStore does not talk about LLMs or prompts.
        return memory.get_for_model(native_tools=native_tools)


    # ------------------------------------------------------------------
    # User profile memory – prompt-level instructions (per session)
    # ------------------------------------------------------------------
    async def get_user_profile_instructions_for_session(
        self,
        session: ChatSession,
    ) -> Optional[str]:
        """
        Return a prompt-ready user profile instruction string for this session.

        Behavior:
          - If a cached value is present in session.metadata["user_profile_instructions"],
            it is returned (after stripping whitespace).
          - Otherwise this method delegates to UserProfileManager, calling
            `get_system_instructions_for_user(user_id)` which returns the
            effective user-level system instructions (already including any
            internal fallbacks), caches the resulting string in metadata,
            and saves the updated session.

        Note:
          - This method no longer uses prompt bundles; it works purely on
            the final system-instructions string exposed by the manager.
        """
        # No associated user or no profile manager → no instructions.
        if not session.user_id:
            return None
        if self._user_profile_manager is None:
            return None

        # 1) Try cached instructions from session metadata.
        cached = session.metadata.get("user_profile_instructions")
        if isinstance(cached, str):
            stripped = cached.strip()
            if stripped:
                return stripped

        # 2) Fallback: resolve from the user profile manager.
        # The manager encapsulates all logic of:
        #   - using profile.system_instructions if set,
        #   - or falling back to a deterministic summary if not.
        instructions = await self._user_profile_manager.get_system_instructions_for_user(
            session.user_id
        )
        if not isinstance(instructions, str):
            return None

        stripped = instructions.strip()
        if not stripped:
            return None

        # 3) Cache in session metadata and persist the session.
        session.metadata["user_profile_instructions"] = stripped
        await self.save_session(session)

        return stripped



    # ------------------------------------------------------------------
    # Organization profile memory – prompt-level instructions (per session)
    # ------------------------------------------------------------------
    async def get_org_profile_instructions_for_session(
        self,
        session: ChatSession
    ) -> Optional[str]:
        """
        Return a prompt-ready organization profile instruction string
        for this session.

        Behavior:
          - If a cached value is present in session.metadata["org_profile_instructions"],
            it is returned (after stripping whitespace).
          - Otherwise this method delegates to OrganizationProfileManager, calling
            `get_system_instructions_for_organization(organization_id, max_summary_length=...)`,
            caches the resulting string in metadata, and saves the updated session.

        Note:
          - This method no longer uses prompt bundles; it works purely on
            the final system-instructions string exposed by the manager.
        """
        # No associated tenant or no organization profile manager → no instructions.
        if not session.tenant_id:
            return None
        if self._organization_profile_manager is None:
            return None

        # 1) Try cached instructions from session metadata.
        cached = session.metadata.get("org_profile_instructions")
        if isinstance(cached, str):
            stripped = cached.strip()
            if stripped:
                return stripped

        # 2) Fallback: resolve from the organization profile manager.
        instructions = await self._organization_profile_manager.get_system_instructions_for_organization(
            organization_id=session.tenant_id
        )
        if not isinstance(instructions, str):
            return None

        stripped = instructions.strip()
        if not stripped:
            return None

        # 3) Cache in session metadata and persist the session.
        session.metadata["org_profile_instructions"] = stripped
        await self.save_session(session)

        return stripped


