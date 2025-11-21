# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Any, Dict, Literal, List, Optional, Sequence


MessageRole = Literal["system", "user", "assistant", "tool"]


@dataclass
class AttachmentRef:
    """
    Lightweight reference to an attachment associated with a message or session.

    The actual binary content is stored elsewhere (filesystem, object storage,
    database BLOB, etc.). Here we only keep stable identifiers and metadata.
    """

    id: str
    type: str  # e.g. "pdf", "docx", "image", "audio", "video", "code", "json"
    uri: str   # e.g. "file://...", "s3://...", "db://attachments/<id>"
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# ChatMessage – extended with tool_calls and to_dict() method
# ============================================================
@dataclass
class ChatMessage:
    """
    Universal chat message compatible with the OpenAI Responses API.
    Supports fields:
      - tool_call_id  → for single tool calls (from field 'id'),
      - tool_calls    → list of calls (for assistant.tool_calls).
    """
    role: MessageRole
    content: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[dict]] = None
    attachments: List[AttachmentRef] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """
        Converts the object to a dict compatible with OpenAI Responses API / ChatCompletions.
        """
        msg = {
            "role": self.role,
            "content": self.content,
        }
        if self.name:
            msg["name"] = self.name
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        return msg

    def __repr__(self):
        extras = []
        if self.name:
            extras.append(f"name={self.name}")
        if self.tool_call_id:
            extras.append(f"tool_call_id={self.tool_call_id}")
        if self.tool_calls:
            extras.append(f"tool_calls={len(self.tool_calls)}")
        extras_str = ", ".join(extras)
        return f"<ChatMessage role={self.role} {extras_str}>"


def append_chat_messages(
        existing: Optional[List[ChatMessage]],
        new: List[ChatMessage]
) -> List[ChatMessage]:
    """
    Custom reducer for LangGraph state.

    LangGraph calls this function when merging state updates:
      - `existing`: the current list of messages in the state (may be None)
      - `new`: the list of messages provided by a node update

    We simply append the new messages to the existing ones.
    """
    if existing is None:
        return list(new)
    return [*existing, *new]


# ============================================================
# IntergraxConversationalMemory – expanded with get_for_model()
# ============================================================
class IntergraxConversationalMemory:
    """
    Universal conversation memory:
      - works independently of LLMs/adapters,
      - keeps messages in RAM,
      - has separate save/load methods to FILES (JSON/NDJSON) and SQLite,
      - no subclasses/adapters required.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        *,
        max_messages: Optional[int] = None,
    ):
        self.session_id: str = session_id or str(uuid.uuid4())
        self._messages: List[ChatMessage] = []
        self._max_messages = max_messages
        self._lock = threading.RLock()

    # --------- Basic in-memory operations ---------

    def add_message(self, role: MessageRole, content: str) -> None:
        with self._lock:
            self._messages.append(ChatMessage(role=role, content=str(content)))
            if self._max_messages is not None and len(self._messages) > self._max_messages:
                overflow = len(self._messages) - self._max_messages
                del self._messages[0:overflow]

    def extend(self, messages: Sequence[ChatMessage]) -> None:
        with self._lock:
            for m in messages:
                self._messages.append(m)
            if self._max_messages is not None and len(self._messages) > self._max_messages:
                overflow = len(self._messages) - self._max_messages
                del self._messages[0:overflow]

    def get_all(self) -> List[ChatMessage]:
        with self._lock:
            return list(self._messages)

    # --------- New filtering method for OpenAI / Ollama ---------
    def get_for_model(self, native_tools: bool = False) -> List[ChatMessage]:
        """
        Returns messages prepared for sending to the model.
        If native_tools=True (e.g., OpenAI), removes older 'tool' messages
        that cannot appear in history (API requires assistant.tool_calls -> tool pairing).
        For planners (e.g., Ollama) returns the full history.
        """
        with self._lock:
            if native_tools:
                return [m for m in self._messages if m.role in ("system", "user", "assistant")]
            return list(self._messages)

    def get_recent(self, n: int) -> List[ChatMessage]:
        with self._lock:
            return self._messages[-n:] if n > 0 else []

    def clear(self) -> None:
        with self._lock:
            self._messages.clear()

    # --------- Save/Load TO FILES ---------

    def save_to_json(self, file_path: str, *, pretty: bool = True) -> None:
        """
        Save the entire history to a single JSON file.
        """
        with self._lock:
            payload = {
                "session_id": self.session_id,
                "messages": [asdict(m) for m in self._messages],
            }
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            if pretty:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            else:
                json.dump(payload, f, ensure_ascii=False)

    def load_from_json(self, file_path: str, *, replace: bool = True) -> None:
        """
        Load history from a JSON file (same format as in save_to_json).
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        messages = [ChatMessage(**m) for m in data.get("messages", [])]
        with self._lock:
            if replace:
                self._messages = messages
            else:
                self._messages.extend(messages)

    def append_to_ndjson(self, file_path: str) -> None:
        """
        Append history to an NDJSON file (one record = one message).
        Each line has keys: session_id, role, content, created_at.
        """
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        with self._lock, open(file_path, "a", encoding="utf-8") as f:
            for m in self._messages:
                rec = {
                    "session_id": self.session_id,
                    "role": m.role,
                    "content": m.content,
                    "created_at": m.created_at,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def load_from_ndjson(self, file_path: str, *, session_id: Optional[str] = None, replace: bool = True) -> None:
        """
        Loads lines from NDJSON. If you provide session_id – filters by it.
        """
        messages: List[ChatMessage] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if session_id is None or rec.get("session_id") == session_id:
                    messages.append(
                        ChatMessage(
                            role=rec.get("role", "user"),
                            content=rec.get("content", ""),
                            created_at=rec.get("created_at") or datetime.utcnow().isoformat(),
                        )
                    )
        with self._lock:
            if replace:
                self._messages = messages
            else:
                self._messages.extend(messages)

    # --------- Save/Load TO SQLite ---------

    @staticmethod
    def _sqlite_ensure_schema(db_path: str) -> None:
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        conn = sqlite3.connect(db_path)
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
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id)")
        finally:
            conn.commit()
            conn.close()

    def save_to_sqlite(self, db_path: str, *, replace_session: bool = False) -> None:
        """
        Saves the current memory to SQLite.
        If replace_session=True – deletes previously saved messages for this session and writes fresh.
        """
        self._sqlite_ensure_schema(db_path)
        conn = sqlite3.connect(db_path)
        try:
            if replace_session:
                conn.execute("DELETE FROM messages WHERE session_id = ?", (self.session_id,))
            with self._lock:
                rows = [
                    (self.session_id, m.role, m.content, m.created_at)
                    for m in self._messages
                ]
            conn.executemany(
                "INSERT INTO messages(session_id, role, content, created_at) VALUES (?,?,?,?)",
                rows,
            )
        finally:
            conn.commit()
            conn.close()

    def load_from_sqlite(
        self,
        db_path: str,
        *,
        session_id: Optional[str] = None,
        limit: Optional[int] = None,
        replace: bool = True,
        order: str = "ASC",  # "ASC" from oldest, "DESC" from newest
    ) -> None:
        """
        Loads messages from a SQLite database (for the current or given session_id).
        """
        self._sqlite_ensure_schema(db_path)
        sess = session_id or self.session_id
        order = "DESC" if str(order).upper() == "DESC" else "ASC"

        conn = sqlite3.connect(db_path)
        try:
            if limit is None:
                cur = conn.execute(
                    f"SELECT role, content, created_at FROM messages WHERE session_id=? ORDER BY id {order}",
                    (sess,),
                )
            else:
                cur = conn.execute(
                    f"SELECT role, content, created_at FROM messages WHERE session_id=? ORDER BY id {order} LIMIT ?",
                    (sess, limit),
                )
            rows = cur.fetchall()
        finally:
            conn.close()

        msgs = [ChatMessage(role=r[0], content=r[1], created_at=r[2]) for r in rows]
        if order == "DESC":
            msgs.reverse()  # in memory we want oldest to newest

        with self._lock:
            if replace:
                self._messages = msgs
            else:
                self._messages.extend(msgs)

    def append_message_to_sqlite(self, db_path: str, role: MessageRole, content: str) -> None:
        """
        Convenience version: append a SINGLE message directly to SQLite (without losing RAM).
        """
        self._sqlite_ensure_schema(db_path)
        now = datetime.utcnow().isoformat()
        conn = sqlite3.connect(db_path)
        try:
            conn.execute(
                "INSERT INTO messages(session_id, role, content, created_at) VALUES (?,?,?,?)",
                (self.session_id, role, str(content), now),
            )
        finally:
            conn.commit()
            conn.close()
        # Also append to RAM:
        self.add_message(role, content)
