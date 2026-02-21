# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional, List

import json
from datetime import datetime, timezone
import uuid


from intergrax.runtime.nexus.session.session_storage import SessionStorage
from intergrax.runtime.nexus.session.chat_session import ChatSession, SessionStatus
from intergrax.runtime.nexus.session.session_message_append_result import SessionMessageAppendResult
from intergrax.llm.messages import AttachmentRef, ChatMessage
from intergrax.utils.time_provider import SystemTimeProvider


class SQLiteSessionStorage(SessionStorage):
    """
    SQLite-backed implementation of SessionStorage.

    Fully self-contained.
    No knowledge required from Runtime.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path: str = db_path
        self._connection: sqlite3.Connection = self._create_connection(db_path)
        self._initialize_schema()

    # ------------------------------------------------------------------
    # Internal infrastructure
    # ------------------------------------------------------------------

    def _create_connection(self, db_path: str) -> sqlite3.Connection:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        connection = sqlite3.connect(str(path))
        connection.execute("PRAGMA foreign_keys = ON;")
        return connection

    def _initialize_schema(self) -> None:
        cursor = self._connection.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                metadata_json TEXT,
                status TEXT NOT NULL,
                user_turns INTEGER NOT NULL,
                created_at_utc TEXT NOT NULL,
                updated_at_utc TEXT NOT NULL
            );
            """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_sessions_user_lookup
            ON sessions (tenant_id, workspace_id, user_id);
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS session_messages (
                entry_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,

                role TEXT NOT NULL,
                content TEXT NOT NULL,

                deleted INTEGER NOT NULL,
                modified INTEGER NOT NULL,

                created_at TEXT NOT NULL,

                tool_call_id TEXT,
                name TEXT,

                tool_calls_json TEXT,
                attachments_json TEXT,
                metadata_json TEXT,

                FOREIGN KEY (session_id)
                    REFERENCES sessions(session_id)
                    ON DELETE CASCADE
            );
            """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_session_messages_session
            ON session_messages (session_id);
            """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_session_messages_created_at
            ON session_messages (created_at);
            """
        )

        self._connection.commit()

    # ------------------------------------------------------------------
    # SessionStorage contract (not implemented yet)
    # ------------------------------------------------------------------

    async def create_session(
        self,
        *,
        session_id: Optional[str],
        tenant_id: str,
        workspace_id: str,
        user_id: str,
        metadata: Optional[dict]
    ) -> ChatSession:

        if session_id is None:
            session_id = str(uuid.uuid4())

        # ChatSession expects:
        # - id: str
        # - user_id/tenant_id/workspace_id: Optional[str] (we pass str)
        # - metadata: Dict[str, Any] (never None)
        session = ChatSession(
            id=session_id,
            user_id=user_id,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            metadata=metadata or {},
        )

        cursor = self._connection.cursor()
        cursor.execute(
            """
            INSERT INTO sessions (
                session_id,
                tenant_id,
                workspace_id,
                user_id,
                metadata_json,
                status,
                user_turns,
                created_at_utc,
                updated_at_utc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session.id,
                tenant_id,
                workspace_id,
                user_id,
                json.dumps(session.metadata) if session.metadata else None,
                session.status.value,
                session.user_turns,
                session.created_at.isoformat(),
                session.updated_at.isoformat(),
            ),
        )

        self._connection.commit()
        return session

    async def get_session(self, session_id: str) -> Optional[ChatSession]:

        cursor = self._connection.cursor()
        cursor.execute(
            """
            SELECT
                session_id,
                tenant_id,
                workspace_id,
                user_id,
                metadata_json,
                status,
                user_turns,
                created_at_utc,
                updated_at_utc
            FROM sessions
            WHERE session_id = ?
            """,
            (session_id,),
        )

        row = cursor.fetchone()
        if row is None:
            return None
        
        (
            db_session_id,
            tenant_id,
            workspace_id,
            user_id,
            metadata_json,
            status_value,
            user_turns,
            created_at_utc,
            updated_at_utc,
        ) = row

        metadata = json.loads(metadata_json) if metadata_json else {}

        session = ChatSession(
            id=db_session_id,
            user_id=user_id,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            metadata=metadata,
            status=SessionStatus(status_value),
            user_turns=user_turns,
            created_at=datetime.fromisoformat(created_at_utc),
            updated_at=datetime.fromisoformat(updated_at_utc),
        )

        return session

    async def save_session(self, session: ChatSession) -> None:
        cursor = self._connection.cursor()

        cursor.execute(
            """
            UPDATE sessions
            SET
                tenant_id = ?,
                workspace_id = ?,
                user_id = ?,
                metadata_json = ?,
                status = ?,
                user_turns = ?,
                created_at_utc = ?,
                updated_at_utc = ?
            WHERE session_id = ?
            """,
            (
                session.tenant_id,
                session.workspace_id,
                session.user_id,
                json.dumps(session.metadata) if session.metadata else None,
                session.status.value,
                session.user_turns,
                session.created_at.isoformat(),
                session.updated_at.isoformat(),
                session.id,
            ),
        )

        self._connection.commit()

    async def list_sessions_for_user(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        user_id: str,
        limit: int
    ) -> List[ChatSession]:
        
        cursor = self._connection.cursor()
        cursor.execute(
            """
            SELECT
                session_id,
                tenant_id,
                workspace_id,
                user_id,
                metadata_json,
                status,
                user_turns,
                created_at_utc,
                updated_at_utc
            FROM sessions
            WHERE tenant_id = ?
            AND workspace_id = ?
            AND user_id = ?
            ORDER BY updated_at_utc DESC
            LIMIT ?
            """,
            (tenant_id, workspace_id, user_id, limit),
        )

        rows = cursor.fetchall()
        sessions: List[ChatSession] = []

        for row in rows:
            (
                db_session_id,
                db_tenant_id,
                db_workspace_id,
                db_user_id,
                metadata_json,
                status_value,
                user_turns,
                created_at_utc,
                updated_at_utc,
            ) = row

            metadata = json.loads(metadata_json) if metadata_json else {}

            session = ChatSession(
                id=db_session_id,
                user_id=db_user_id,
                tenant_id=db_tenant_id,
                workspace_id=db_workspace_id,
                metadata=metadata,
                status=SessionStatus(status_value),
                user_turns=user_turns,
                created_at=datetime.fromisoformat(created_at_utc),
                updated_at=datetime.fromisoformat(updated_at_utc),
            )

            sessions.append(session)

        return sessions

    async def append_message(
        self,
        *,
        session_id: str,
        message: ChatMessage
    ) -> SessionMessageAppendResult:        

        session = await self.get_session(session_id)
        if session is None:
            raise ValueError(f"Session not found: {session_id}")

        cursor = self._connection.cursor()

        cursor.execute(
            """
            INSERT INTO session_messages (
                entry_id,
                session_id,
                role,
                content,
                deleted,
                modified,
                created_at,
                tool_call_id,
                name,
                tool_calls_json,
                attachments_json,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message.entry_id,
                session_id,
                message.role,
                message.content,
                1 if message.deleted else 0,
                1 if message.modified else 0,
                message.created_at,
                message.tool_call_id,
                message.name,
                json.dumps(message.tool_calls) if message.tool_calls else None,
                json.dumps(
                    [
                        {
                            "id": a.id,
                            "type": a.type,
                            "uri": a.uri,
                            "metadata": a.metadata,
                        }
                        for a in message.attachments
                    ]
                ) if message.attachments else None,
                json.dumps(message.metadata) if message.metadata else None,
            ),
        )

        cursor.execute(
            """
            UPDATE sessions
            SET updated_at_utc = ?
            WHERE session_id = ?
            """,
            (
                message.created_at,
                session_id,
            ),
        )

        self._connection.commit()

        return SessionMessageAppendResult(
            message=message,
            consolidation_diag=None,
        )

    async def get_history(
        self,
        *,
        session_id: str,
        native_tools: bool
    ) -> List[ChatMessage]:
        
        cursor = self._connection.cursor()

        cursor.execute(
            """
            SELECT
                entry_id,
                role,
                content,
                deleted,
                modified,
                created_at,
                tool_call_id,
                name,
                tool_calls_json,
                attachments_json,
                metadata_json
            FROM session_messages
            WHERE session_id = ?
            ORDER BY created_at ASC
            """,
            (session_id,),
        )

        rows = cursor.fetchall()
        messages: List[ChatMessage] = []

        for row in rows:
            (
                entry_id,
                role,
                content,
                deleted,
                modified,
                created_at,
                tool_call_id,
                name,
                tool_calls_json,
                attachments_json,
                metadata_json,
            ) = row

            tool_calls = json.loads(tool_calls_json) if tool_calls_json else None
            attachments_raw = json.loads(attachments_json) if attachments_json else []
            metadata = json.loads(metadata_json) if metadata_json else {}

            attachments = [
                AttachmentRef(
                    id=a["id"],
                    type=a["type"],
                    uri=a["uri"],
                    metadata=a.get("metadata", {}),
                )
                for a in attachments_raw
            ]

            message = ChatMessage(
                role=role,
                content=content,
                entry_id=entry_id,
                deleted=bool(deleted),
                modified=bool(modified),
                created_at=created_at,
                tool_call_id=tool_call_id,
                name=name,
                tool_calls=tool_calls,
                attachments=attachments,
                metadata=metadata,
            )

            messages.append(message)

        if not native_tools:
            messages = [
                m for m in messages
                if m.role != "tool"
            ]

        return messages