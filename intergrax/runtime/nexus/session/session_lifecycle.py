# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Any, Dict, List, Optional

from intergrax.runtime.nexus.session.chat_session import ChatSession
from intergrax.runtime.nexus.session.session_storage import SessionStorage


class SessionLifecycleCoordinator:
    """Session metadata CRUD on top of SessionStorage (no message history)."""

    def __init__(self, storage: SessionStorage) -> None:
        self._storage = storage

    async def get_session(
        self,
        *,
        tenant_id: str,
        session_id: str,
    ) -> Optional[ChatSession]:
        return await self._storage.get_session(tenant_id=tenant_id, session_id=session_id)

    async def create_session(
        self,
        *,
        tenant_id: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChatSession:
        return await self._storage.create_session(
            tenant_id=tenant_id,
            session_id=session_id,
            user_id=user_id,
            workspace_id=workspace_id,
            metadata=metadata,
        )

    async def get_or_create_session(
        self,
        *,
        user_id: str,
        session_id: str,
        tenant_id: str,
        workspace_id: Optional[str] = None,
    ) -> ChatSession:
        session = await self.get_session(tenant_id=tenant_id, session_id=session_id)
        if session is not None:
            if workspace_id is not None and session.workspace_id != workspace_id:
                raise ValueError(
                    "Session workspace mismatch for given session_id. "
                    "Possible cross-workspace collision or access attempt."
                )
            return session
        session = await self.create_session(
            user_id=user_id,
            session_id=session_id,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        await self.save_session(session)
        return session

    async def save_session(self, session: ChatSession) -> None:
        session.touch()
        await self._storage.save_session(session)

    async def list_sessions_for_user(
        self,
        user_id: str,
        *,
        limit: Optional[int] = None,
    ) -> List[ChatSession]:
        return await self._storage.list_sessions_for_user(user_id, limit=limit)
