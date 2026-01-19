# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import sqlite3
from datetime import datetime
from typing import List, Dict, Optional, Tuple

DB_NAME = "rag_app.db"

# --------- Low-level helpers ---------

def get_db_connection():
    conn = sqlite3.connect(DB_NAME, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row    
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

# --------- Schema creation / migration ---------

def create_schema():
    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('user','assistant','system','tool')),
            content TEXT NOT NULL,
            model TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_session_created ON messages(session_id, created_at)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_session_role ON messages(session_id, role)")
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS document_store (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,                -- nazwa widoczna w UI
            original_filename TEXT,                -- oryginalna nazwa uploadu
            size_bytes INTEGER,
            mime_type TEXT,
            content_hash TEXT,                     -- np. sha256 do dedupu
            upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_docs_uploaded ON document_store(upload_timestamp DESC)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_docs_hash ON document_store(content_hash)")

    conn.commit()
    conn.close()

def migrate_from_legacy_application_logs():    
    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='application_logs'")
    if cur.fetchone() is None:
        conn.close()
        return
    
    rows = cur.execute("""
        SELECT session_id, user_query, gpt_response, model, created_at
        FROM application_logs
        ORDER BY created_at
    """).fetchall()
    
    for r in rows:
        sid = r["session_id"]
        if not sid:
            continue
        cur.execute("INSERT OR IGNORE INTO sessions(session_id) VALUES (?)", (sid,))

        # user
        if r["user_query"]:
            cur.execute("""
                INSERT INTO messages(session_id, role, content, model, created_at)
                VALUES (?, 'user', ?, ?, ?)
            """, (sid, r["user_query"], r["model"], r["created_at"]))

        # assistant
        if r["gpt_response"]:
            cur.execute("""
                INSERT INTO messages(session_id, role, content, model, created_at)
                VALUES (?, 'assistant', ?, ?, ?)
            """, (sid, r["gpt_response"], r["model"], r["created_at"]))

    conn.commit()
    conn.close()

# --------- Public API: messages ---------

def ensure_session(session_id: str):
    conn = get_db_connection()
    conn.execute("INSERT OR IGNORE INTO sessions(session_id) VALUES (?)", (session_id,))
    conn.commit()
    conn.close()

def insert_message(session_id: str, role: str, content: str, model: Optional[str] = None):
    """
    role ∈ {'user','assistant','system','tool'}
    """
    if role not in ("user","assistant","system","tool"):
        raise ValueError("Invalid role")
    conn = get_db_connection()
    conn.execute("INSERT OR IGNORE INTO sessions(session_id) VALUES (?)", (session_id,))
    conn.execute(
        "INSERT INTO messages (session_id, role, content, model) VALUES (?,?,?,?)",
        (session_id, role, content, model),
    )
    conn.commit()
    conn.close()

def get_messages(session_id: str, limit: Optional[int] = None) -> List[Dict]:
    conn = get_db_connection()
    cur = conn.cursor()
    # if limit:
    #     cur.execute(
    #         "SELECT * FROM messages WHERE session_id=? ORDER BY created_at LIMIT ?",
    #         (session_id, limit),
    #     )
    # else:
    #     cur.execute(
    #         "SELECT * FROM messages WHERE session_id=? ORDER BY created_at",
    #         (session_id,),
    #     )

    if limit:
        cur.execute(
            "SELECT * FROM messages ORDER BY created_at LIMIT ?",
            (limit),
        )
    else:
        cur.execute(
            "SELECT * FROM messages ORDER BY created_at",
        )

    out = [dict(r) for r in cur.fetchall()]
    conn.close()
    return out

def get_history_pairs(session_id: str, turns: int = 1) -> List[Tuple[str,str]]:
    msgs = get_messages(session_id)
    pairs: List[Tuple[str,str]] = []
    buf_user: Optional[str] = None
    for m in msgs:
        if m["role"] == "user":
            buf_user = m["content"]
        elif m["role"] == "assistant" and buf_user is not None:
            pairs.append((buf_user, m["content"]))
            buf_user = None
    # zwróć tylko końcowe N par
    if turns and turns > 0:
        pairs = pairs[-turns:]
    return pairs

# --------- Public API: documents ---------

def insert_document_record(
    filename: str,
    *,
    original_filename: Optional[str] = None,
    size_bytes: Optional[int] = None,
    mime_type: Optional[str] = None,
    content_hash: Optional[str] = None,
) -> int:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO document_store (filename, original_filename, size_bytes, mime_type, content_hash)
        VALUES (?, ?, ?, ?, ?)
    """, (filename, original_filename, size_bytes, mime_type, content_hash))
    file_id = cur.lastrowid
    conn.commit()
    conn.close()
    return file_id

def delete_document_record(file_id: int) -> bool:
    conn = get_db_connection()
    conn.execute("DELETE FROM document_store WHERE id = ?", (file_id,))
    conn.commit()
    conn.close()
    return True

def get_all_documents() -> List[Dict]:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM document_store ORDER BY upload_timestamp DESC")
    docs = [dict(r) for r in cur.fetchall()]
    conn.close()
    return docs


def create_application_logs():
    create_schema()

def insert_application_logs(session_id, user_query, gpt_response, model):
    ensure_session(session_id)
    insert_message(session_id, "user", user_query, model=None)
    insert_message(session_id, "assistant", gpt_response, model=model)

def get_chat_history(session_id):
    msgs = get_messages(session_id)
    out = []
    for m in msgs:
        if m["role"] in ("user","assistant"):
            out.append({"role": m["role"], "content": m["content"]})
    return out

# --------- Init ---------

create_schema()
migrate_from_legacy_application_logs()
