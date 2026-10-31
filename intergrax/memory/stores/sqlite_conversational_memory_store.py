# © Artur Czarnecki. All rights reserved.

"""SQLite-backed conversational memory store (Phase MEM-ST.4)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from intergrax.llm.messages import ChatMessage, MessageRole
from intergrax.memory.conversational_memory import ConversationalMemory
from intergrax.memory.conversational_memory_persistence import (
    ConversationalMemoryMessageWire,
    ConversationalMemoryPersistedPayload,
    parse_conversational_memory_persisted,
)
from intergrax.memory.conversational_store import ConversationalMemoryStore


class SQLiteConversationalMemoryStore(ConversationalMemoryStore):
    """Persist ``ConversationalMemory`` aggregates per tenant/session."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._connection = self._open(db_path)
        self._init_schema()

    def _open(self, db_path: str) -> sqlite3.Connection:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(path))
        connection.execute("PRAGMA foreign_keys = ON;")
        return connection

    def _init_schema(self) -> None:
        cursor = self._connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS conversational_memory (
                tenant_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (tenant_id, session_id)
            );
            """
        )
        self._connection.commit()

    async def load_memory(
        self,
        *,
        tenant_id: str,
        session_id: str,
        max_messages: int | None = None,
    ) -> ConversationalMemory:
        cursor = self._connection.cursor()
        cursor.execute(
            """
            SELECT payload_json FROM conversational_memory
            WHERE tenant_id = ? AND session_id = ?
            """,
            (tenant_id, session_id),
        )
        row = cursor.fetchone()
        if row is None:
            return ConversationalMemory(session_id=session_id, max_messages=max_messages or 100)
        raw = json.loads(str(row[0]))
        if not isinstance(raw, dict):
            raise ValueError("corrupt conversational memory payload")
        envelope = parse_conversational_memory_persisted(raw)
        messages = [ChatMessage(role=MessageRole(item.role), content=item.content) for item in envelope.messages]
        limit = max_messages or envelope.max_messages
        memory = ConversationalMemory(session_id=session_id, max_messages=limit)
        for message in messages:
            memory.add_message(message)
        return memory

    async def save_memory(
        self,
        *,
        tenant_id: str,
        memory: ConversationalMemory,
    ) -> None:
        limit = memory._max_messages if memory._max_messages is not None else 100
        envelope = ConversationalMemoryPersistedPayload(
            max_messages=limit,
            messages=tuple(
                ConversationalMemoryMessageWire(
                    role=str(message.role),
                    content=message.content,
                )
                for message in memory.get_all()
            ),
        )
        payload = envelope.model_dump(mode="json")
        cursor = self._connection.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO conversational_memory (tenant_id, session_id, payload_json)
            VALUES (?, ?, ?)
            """,
            (tenant_id, memory.session_id, json.dumps(payload)),
        )
        self._connection.commit()

    async def append_message(
        self,
        *,
        tenant_id: str,
        session_id: str,
        message: ChatMessage,
    ) -> None:
        memory = await self.load_memory(tenant_id=tenant_id, session_id=session_id)
        memory.add_message(message)
        await self.save_memory(tenant_id=tenant_id, memory=memory)

    async def clear_memory(self, *, tenant_id: str, session_id: str) -> None:
        cursor = self._connection.cursor()
        cursor.execute(
            """
            DELETE FROM conversational_memory
            WHERE tenant_id = ? AND session_id = ?
            """,
            (tenant_id, session_id),
        )
        self._connection.commit()
