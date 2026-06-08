# © Artur Czarnecki. All rights reserved.

"""DocumentStore-backed session persistence (Phase MEM-DEPTH-2.1)."""

from __future__ import annotations

import json
import uuid
from dataclasses import replace
from typing import List, Optional

from intergrax.integrations.contracts.document_store import DocumentRecord, DocumentStore
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.session.chat_session import ChatSession
from intergrax.runtime.nexus.session.session_storage import SessionStorage
from intergrax.utils.time_provider import SystemTimeProvider


class DocumentStoreSessionStorage(SessionStorage):
    """
    Persist chat sessions and history via Tier-0 ``DocumentStore``.

    Partition key = tenant_id; row keys:
      - ``session:{session_id}`` — session metadata JSON
      - ``session:{session_id}:history`` — message list JSON
    """

    _SESSION_PREFIX = "session:"

    def __init__(self, document_store: DocumentStore) -> None:
        self._store = document_store

    def _session_key(self, session_id: str) -> str:
        return f"{self._SESSION_PREFIX}{session_id}"

    def _history_key(self, session_id: str) -> str:
        return f"{self._SESSION_PREFIX}{session_id}:history"

    @staticmethod
    def _session_to_dict(session: ChatSession) -> dict[str, object]:
        created = session.created_at.isoformat() if session.created_at else ""
        updated = session.updated_at.isoformat() if session.updated_at else ""
        return {
            "id": session.id,
            "user_id": session.user_id,
            "tenant_id": session.tenant_id,
            "workspace_id": session.workspace_id,
            "metadata": dict(session.metadata or {}),
            "status": session.status.value if hasattr(session.status, "value") else str(session.status),
            "user_turns": session.user_turns,
            "created_at_utc": created,
            "updated_at_utc": updated,
        }

    @staticmethod
    def _session_from_dict(payload: dict[str, object]) -> ChatSession:
        from intergrax.runtime.nexus.session.chat_session import SessionStatus

        status_raw = str(payload.get("status", SessionStatus.OPEN.value))
        try:
            status = SessionStatus(status_raw)
        except ValueError:
            status = SessionStatus.ACTIVE
        from datetime import datetime

        session = ChatSession(
            id=str(payload["id"]),
            user_id=str(payload.get("user_id") or ""),
            tenant_id=str(payload.get("tenant_id") or ""),
            workspace_id=str(payload.get("workspace_id") or ""),
            metadata=dict(payload.get("metadata") or {}),
            status=status,
            user_turns=int(payload.get("user_turns") or 0),
        )
        created_raw = str(payload.get("created_at_utc") or "")
        updated_raw = str(payload.get("updated_at_utc") or "")
        if created_raw:
            session.created_at = datetime.fromisoformat(created_raw)
        if updated_raw:
            session.updated_at = datetime.fromisoformat(updated_raw)
        return session

    @staticmethod
    def _message_to_dict(message: ChatMessage) -> dict[str, object]:
        return {"role": message.role, "content": message.content or ""}

    @staticmethod
    def _message_from_dict(payload: dict[str, object]) -> ChatMessage:
        return ChatMessage(role=str(payload["role"]), content=str(payload.get("content") or ""))

    async def get_session(
        self,
        *,
        tenant_id: str,
        session_id: str,
    ) -> Optional[ChatSession]:
        record = self._store.get(tenant_id, self._session_key(session_id))
        if record is None:
            return None
        raw = record.data.get("session_json")
        if not isinstance(raw, str):
            return None
        payload = json.loads(raw)
        return self._session_from_dict(payload)

    async def create_session(
        self,
        *,
        tenant_id: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> ChatSession:
        sid = session_id or str(uuid.uuid4())
        now = SystemTimeProvider.utc_now().isoformat()
        session = ChatSession(
            id=sid,
            user_id=user_id or "",
            tenant_id=tenant_id,
            workspace_id=workspace_id or "",
            metadata=dict(metadata or {}),
        )
        from datetime import datetime

        session.created_at = datetime.fromisoformat(now)
        session.updated_at = datetime.fromisoformat(now)
        await self.save_session(session)
        self._store.put(
            DocumentRecord(
                partition_key=tenant_id,
                row_key=self._history_key(sid),
                data={"history_json": "[]"},
            )
        )
        return session

    async def save_session(self, session: ChatSession) -> None:
        tenant_id = session.tenant_id or ""
        self._store.put(
            DocumentRecord(
                partition_key=tenant_id,
                row_key=self._session_key(session.id),
                data={"session_json": json.dumps(self._session_to_dict(session))},
            )
        )

    async def list_sessions_for_user(
        self,
        *,
        tenant_id: str,
        user_id: str,
        limit: Optional[int] = None,
    ) -> List[ChatSession]:
        result = self._store.query(tenant_id, row_key_prefix=self._SESSION_PREFIX, limit=limit or 100)
        sessions: List[ChatSession] = []
        for record in result.documents:
            if record.row_key.endswith(":history"):
                continue
            raw = record.data.get("session_json")
            if not isinstance(raw, str):
                continue
            session = self._session_from_dict(json.loads(raw))
            if session.user_id == user_id:
                sessions.append(session)
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        if limit is not None:
            return sessions[:limit]
        return sessions

    async def append_message(
        self,
        *,
        tenant_id: str,
        session_id: str,
        message: ChatMessage,
    ) -> ChatMessage:
        history = await self.get_history(tenant_id=tenant_id, session_id=session_id)
        history.append(message)
        self._store.put(
            DocumentRecord(
                partition_key=tenant_id,
                row_key=self._history_key(session_id),
                data={
                    "history_json": json.dumps([self._message_to_dict(m) for m in history]),
                },
            )
        )
        session = await self.get_session(tenant_id=tenant_id, session_id=session_id)
        if session is not None:
            from datetime import datetime

            session.updated_at = datetime.fromisoformat(SystemTimeProvider.utc_now().isoformat())
            await self.save_session(session)
        return message

    async def get_history(
        self,
        *,
        tenant_id: str,
        session_id: str,
        native_tools: bool = False,
    ) -> List[ChatMessage]:
        _ = native_tools
        record = self._store.get(tenant_id, self._history_key(session_id))
        if record is None:
            return []
        raw = record.data.get("history_json")
        if not isinstance(raw, str):
            return []
        payloads = json.loads(raw)
        return [self._message_from_dict(item) for item in payloads]

    def close(self) -> None:
        self._store.close()
