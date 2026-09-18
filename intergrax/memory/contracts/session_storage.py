# © Artur Czarnecki. All rights reserved.

"""Neutral session storage contract for Memory store plugins (PLUG-02)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.llm.messages import ChatMessage
from intergrax.memory.contracts.chat_session import ChatSession


@runtime_checkable
class SessionStorage(Protocol):
    """Low-level storage for chat sessions and conversation history."""

    async def get_session(
        self,
        *,
        tenant_id: str,
        session_id: str,
    ) -> ChatSession | None:
        ...

    async def create_session(
        self,
        *,
        tenant_id: str,
        session_id: str | None = None,
        user_id: str | None = None,
        workspace_id: str | None = None,
        metadata: dict | None = None,
    ) -> ChatSession:
        ...

    async def save_session(self, session: ChatSession) -> None:
        ...

    async def list_sessions_for_user(
        self,
        *,
        tenant_id: str,
        user_id: str,
        limit: int | None = None,
    ) -> list[ChatSession]:
        ...

    async def append_message(
        self,
        *,
        tenant_id: str,
        session_id: str,
        message: ChatMessage,
    ) -> ChatMessage:
        ...

    async def get_history(
        self,
        *,
        tenant_id: str,
        session_id: str,
        native_tools: bool = False,
    ) -> list[ChatMessage]:
        ...
