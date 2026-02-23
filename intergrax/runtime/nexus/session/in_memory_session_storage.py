# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import uuid
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

from intergrax.llm.messages import ChatMessage
from intergrax.memory.conversational_memory import ConversationalMemory
from intergrax.runtime.nexus.session.chat_session import ChatSession
from intergrax.runtime.nexus.session.session_storage import SessionStorage


class InMemorySessionStorage(SessionStorage):
    """
    In-memory implementation of SessionStorage.

    Responsibilities:
      - Keep ChatSession metadata in an in-process dictionary.
      - Maintain per-session conversation history using ConversationalMemory.
      - Apply a simple FIFO trimming policy via ConversationalMemory's
        max_messages setting.

    This implementation is suitable for:
      - development,
      - tests,
      - single-process / single-node setups.

    It is NOT intended for production use in distributed or long-lived
    environments. For production you should implement a persistent
    SessionStorage based on a database, Redis, or another durable backend.
    """

    def __init__(
        self,
        *,
        max_history_messages: Optional[int] = None,
    ) -> None:
        # Metadata storage (chat sessions registry).
        self._sessions: Dict[Tuple[str, str], ChatSession] = {}

        # Internal conversational memory storage (one per session).
        self._conv_memory: Dict[Tuple[str, str], ConversationalMemory] = {}

        # Maximum number of messages to keep before trimming FIFO-style.
        self._max_history_messages = max_history_messages

    # ------------------------------------------------------------------
    # Session metadata CRUD
    # ------------------------------------------------------------------

    async def get_session(
        self,
        *,
        tenant_id: str,
        session_id: str,
    ) -> Optional[ChatSession]:
        """
        Return the session metadata if it exists for the given tenant,
        else None.
        """
        key = (tenant_id, session_id)
        session = self._sessions.get(key)
        if session is None:
            return None
        return replace(session)


    async def create_session(
        self,
        session_id: Optional[str] = None,
        *,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> ChatSession:
        """
        Create and persist a new ChatSession.

        If session_id is None, a new UUID4-based id is generated.
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        if metadata is None:
            metadata = {}

        session = ChatSession(
            id=session_id,
            user_id=user_id,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            metadata=dict(metadata),
        )

        self._sessions[session_id] = session
        return replace(session)

    async def save_session(self, session: ChatSession) -> None:
        """
        Persist changes to an existing ChatSession scoped by tenant.
        """
        if session.tenant_id is None:
            raise ValueError("Cannot save session without tenant_id")

        key = (session.tenant_id, session.id)
        self._sessions[key] = session


    async def list_sessions_for_user(
        self,
        user_id: str,
        *,
        limit: Optional[int] = None,
    ) -> List[ChatSession]:
        """
        List recent sessions for a given user, ordered by recency.
        """
        sessions = [s for s in self._sessions.values() if s.user_id == user_id]
        sessions.sort(key=lambda s: s.updated_at, reverse=True)

        if limit is not None and limit > 0:
            sessions = sessions[:limit]

        # Return copies so callers cannot mutate internal state directly.
        return [replace(s) for s in sessions]

    # ------------------------------------------------------------------
    # Conversation history operations
    # ------------------------------------------------------------------

    async def append_message(
        self,
        *,
        tenant_id: str,
        session_id: str,
        message: ChatMessage,
    ) -> ChatMessage:
        """
        Append a single message to the conversation history of a session
        scoped to a tenant.
        """
        key = (tenant_id, session_id)

        session = self._sessions.get(key)
        if session is None:
            raise KeyError(
                f"Session '{session_id}' does not exist for tenant '{tenant_id}'"
            )

        memory = self._conv_memory.get(key)

        if memory is None:
            memory = ConversationalMemory(
                session_id=session_id,
                max_messages=self._max_history_messages,
            )
            self._conv_memory[key] = memory

        memory.add_message(message)

        session.touch()
        self._sessions[key] = session

        return message

    async def get_history(
        self,
        *,
        tenant_id: str,
        session_id: str,
        native_tools: bool = False,
    ) -> List[ChatMessage]:
        """
        Return the ordered conversation history for a given session id
        scoped to a tenant.
        """
        key = (tenant_id, session_id)

        memory = self._conv_memory.get(key)
        if memory is None:
            return []

        return memory.get_for_model(native_tools=native_tools)
