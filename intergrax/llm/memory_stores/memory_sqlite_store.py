# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from typing import List, Optional

from intergrax.llm.conversational_memory import IntergraxConversationalMemory
from intergrax.llm.messages import ChatMessage, MessageRole


class SqliteMemoryStore:
    """
    SQLite-based persistence provider for IntergraxConversationalMemory.

    Responsibilities:
      - create the messages table if it does not exist,
      - persist all messages for a given session_id,
      - load messages for a given session_id,
      - optionally append a single message.

    Schema:
      CREATE TABLE IF NOT EXISTS messages (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          session_id TEXT NOT NULL,
          role TEXT NOT NULL,
          content TEXT NOT NULL,
          created_at TEXT NOT NULL
      );
      CREATE INDEX IF NOT EXISTS idx_messages_session_id
          ON messages(session_id);
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_session_id "
                "ON messages(session_id)"
            )
        finally:
            conn.commit()
            conn.close()

    # ------------------------------------------------------------------
    # Save / load full session
    # ------------------------------------------------------------------

    def save_session(
        self,
        memory: IntergraxConversationalMemory,
        *,
        replace_session: bool = True,
    ) -> None:
        """
        Persist all messages from the given memory instance into SQLite.

        If replace_session=True:
          - deletes all existing records for memory.session_id, then inserts fresh ones.
        If replace_session=False:
          - simply inserts current messages, without deleting old ones.
        """
        conn = sqlite3.connect(self._db_path)
        try:
            if replace_session:
                conn.execute(
                    "DELETE FROM messages WHERE session_id = ?",
                    (memory.session_id,),
                )

            messages: List[ChatMessage] = memory.get_all()
            rows = [
                (memory.session_id, m.role, m.content, m.created_at)
                for m in messages
            ]

            conn.executemany(
                "INSERT INTO messages(session_id, role, content, created_at) "
                "VALUES (?,?,?,?)",
                rows,
            )
        finally:
            conn.commit()
            conn.close()

    def load_session(
        self,
        memory: IntergraxConversationalMemory,
        *,
        session_id: Optional[str] = None,
        replace: bool = True,
    ) -> None:
        """
        Load messages for the given session_id into the provided memory.

        If session_id is None:
          - uses memory.session_id.
        If replace=True:
          - clears current memory before appending loaded messages.
        """
        sess_id = session_id or memory.session_id

        conn = sqlite3.connect(self._db_path)
        try:
            cursor = conn.execute(
                "SELECT role, content, created_at "
                "FROM messages WHERE session_id = ? ORDER BY id ASC",
                (sess_id,),
            )
            rows = cursor.fetchall()
        finally:
            conn.close()

        if not rows:
            return

        loaded: List[ChatMessage] = [
            ChatMessage(role=row[0], content=row[1], created_at=row[2])
            for row in rows
        ]

        if replace:
            memory.clear()
        memory.extend(loaded)

    # ------------------------------------------------------------------
    # Single-message append helper
    # ------------------------------------------------------------------

    def append_message(
        self,
        memory: IntergraxConversationalMemory,
        role: MessageRole,
        content: str,
    ) -> None:
        """
        Convenience method:
         - inserts a single message into SQLite for the given session_id,
         - also appends the same message to the in-memory history.
        """
        now = datetime.utcnow().isoformat()
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(
                "INSERT INTO messages(session_id, role, content, created_at) "
                "VALUES (?,?,?,?)",
                (memory.session_id, role, str(content), now),
            )
        finally:
            conn.commit()
            conn.close()

        # Mirror the change in RAM
        memory.add_message(role=role, content=content)
