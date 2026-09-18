# Reference external session storage (Memory contract only — no Nexus imports).

from __future__ import annotations

import uuid
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

from intergrax.llm.messages import ChatMessage
from intergrax.memory.conversational_memory import ConversationalMemory
from intergrax.memory.contracts.chat_session import ChatSession

FIXTURE_SESSION_STORAGE_MARKER = "external.in_memory_session_storage.v1"


class FixtureExternalSessionStorage:
    """Minimal in-memory SessionStorage for external plugin qualification."""

    fixture_marker: str = FIXTURE_SESSION_STORAGE_MARKER

    def __init__(self, *, max_history_messages: Optional[int] = None) -> None:
        self._sessions: Dict[Tuple[str, str], ChatSession] = {}
        self._conv_memory: Dict[Tuple[str, str], ConversationalMemory] = {}
        self._max_history_messages = max_history_messages

    async def get_session(
        self,
        *,
        tenant_id: str,
        session_id: str,
    ) -> ChatSession | None:
        session = self._sessions.get((tenant_id, session_id))
        if session is None:
            return None
        return replace(session)

    async def create_session(
        self,
        *,
        tenant_id: str,
        session_id: str | None = None,
        user_id: str | None = None,
        workspace_id: str | None = None,
        metadata: dict | None = None,
    ) -> ChatSession:
        if session_id is None:
            session_id = str(uuid.uuid4())
        session = ChatSession(
            id=session_id,
            tenant_id=tenant_id,
            user_id=user_id,
            workspace_id=workspace_id,
            metadata=dict(metadata or {}),
        )
        self._sessions[(tenant_id, session_id)] = session
        return replace(session)

    async def save_session(self, session: ChatSession) -> None:
        self._sessions[(session.tenant_id, session.id)] = session

    async def list_sessions_for_user(
        self,
        *,
        tenant_id: str,
        user_id: str,
        limit: int | None = None,
    ) -> list[ChatSession]:
        sessions = [
            s
            for s in self._sessions.values()
            if s.tenant_id == tenant_id and s.user_id == user_id
        ]
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        if limit is not None and limit > 0:
            sessions = sessions[:limit]
        return [replace(s) for s in sessions]

    async def append_message(
        self,
        *,
        tenant_id: str,
        session_id: str,
        message: ChatMessage,
    ) -> ChatMessage:
        key = (tenant_id, session_id)
        if key not in self._sessions:
            raise KeyError(f"Session '{session_id}' does not exist for tenant '{tenant_id}'")
        memory = self._conv_memory.get(key)
        if memory is None:
            memory = ConversationalMemory(
                session_id=session_id,
                max_messages=self._max_history_messages,
            )
            self._conv_memory[key] = memory
        memory.add_message(message)
        session = self._sessions[key]
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
        memory = self._conv_memory.get((tenant_id, session_id))
        if memory is None:
            return []
        return memory.get_for_model(native_tools=native_tools)
