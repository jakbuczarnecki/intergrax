# © Artur Czarnecki. All rights reserved.

"""Qualification-only Class B HTTP provider idempotency oracle (R4 F8 BLOCKER D)."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Literal

_ORACLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS class_b_effects (
    idempotency_key TEXT PRIMARY KEY,
    body_digest TEXT NOT NULL,
    body_text TEXT NOT NULL,
    effect_count INTEGER NOT NULL,
    response_json TEXT NOT NULL
);
"""


class ClassBHttpProviderOracleConflictError(ValueError):
    """Same idempotency key with a different payload."""


class ClassBHttpProviderOracleRetentionExceededError(RuntimeError):
    """Deduplication retention envelope cannot guarantee replay."""


@dataclass(frozen=True, slots=True)
class ClassBHttpProviderOracleDisposition:
    http_status: int
    payload: dict[str, Any] | None
    effect_count: int
    replay: bool


class ClassBHttpProviderOracle:
    """
    Authoritative Class B provider semantics for qualification HTTP fixtures.

    Durable rows survive process restart when ``db_path`` is set.
    """

    def __init__(
        self,
        *,
        db_path: Path | None = None,
        max_retained_keys: int = 10_000,
    ) -> None:
        self._lock = threading.Lock()
        self._max_retained_keys = max(1, max_retained_keys)
        self._memory: dict[str, dict[str, Any]] = {}
        self._db_path = db_path
        if db_path is not None:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(db_path) as conn:
                conn.executescript(_ORACLE_SCHEMA)

    def clear(self) -> None:
        with self._lock:
            self._memory.clear()
            if self._db_path is not None:
                with sqlite3.connect(self._db_path) as conn:
                    conn.execute("DELETE FROM class_b_effects")

    def reload_from_durable_store(self) -> None:
        """Rehydrate in-memory view from SQLite (restart simulation)."""
        if self._db_path is None:
            return
        with self._lock:
            self._memory.clear()
            with sqlite3.connect(self._db_path) as conn:
                rows = conn.execute(
                    "SELECT idempotency_key, body_digest, body_text, effect_count, response_json "
                    "FROM class_b_effects",
                ).fetchall()
            for key, digest, body_text, count, response_json in rows:
                self._memory[key] = {
                    "body_digest": digest,
                    "body_text": body_text,
                    "effect_count": int(count),
                    "response_json": response_json,
                }

    def effect_count(self, idempotency_key: str) -> int:
        key = idempotency_key.strip()
        with self._lock:
            row = self._memory.get(key)
            if row is None and self._db_path is not None:
                self._load_row(key)
                row = self._memory.get(key)
            if row is None:
                return 0
            return int(row["effect_count"])

    def apply_post(
        self,
        *,
        idempotency_key: str,
        body: bytes,
    ) -> ClassBHttpProviderOracleDisposition:
        key = idempotency_key.strip()
        if not key:
            return ClassBHttpProviderOracleDisposition(
                http_status=400,
                payload=None,
                effect_count=0,
                replay=False,
            )
        digest = hashlib.sha256(body).hexdigest()
        body_text = body.decode("utf-8", errors="replace")
        with self._lock:
            existing = self._memory.get(key)
            if existing is None and self._db_path is not None:
                self._load_row(key)
                existing = self._memory.get(key)
            if existing is not None:
                if existing["body_digest"] != digest:
                    raise ClassBHttpProviderOracleConflictError(
                        "idempotency key reused with different payload",
                    )
                payload = json.loads(existing["response_json"])
                return ClassBHttpProviderOracleDisposition(
                    http_status=200,
                    payload=payload,
                    effect_count=int(existing["effect_count"]),
                    replay=True,
                )
            if len(self._memory) >= self._max_retained_keys:
                raise ClassBHttpProviderOracleRetentionExceededError(
                    "class_b_provider_deduplication_retention_exceeded",
                )
            payload = {
                "status": "applied",
                "body": body_text,
                "count": 1,
            }
            row = {
                "body_digest": digest,
                "body_text": body_text,
                "effect_count": 1,
                "response_json": json.dumps(payload),
            }
            self._memory[key] = row
            if self._db_path is not None:
                with sqlite3.connect(self._db_path) as conn:
                    conn.execute(
                        "INSERT INTO class_b_effects "
                        "(idempotency_key, body_digest, body_text, effect_count, response_json) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (key, digest, body_text, 1, row["response_json"]),
                    )
            return ClassBHttpProviderOracleDisposition(
                http_status=200,
                payload=payload,
                effect_count=1,
                replay=False,
            )

    def _load_row(self, key: str) -> None:
        assert self._db_path is not None
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT body_digest, body_text, effect_count, response_json "
                "FROM class_b_effects WHERE idempotency_key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return
        digest, body_text, count, response_json = row
        self._memory[key] = {
            "body_digest": digest,
            "body_text": body_text,
            "effect_count": int(count),
            "response_json": response_json,
        }


def class_b_http_handler_for_oracle(
    oracle: ClassBHttpProviderOracle,
    *,
    mode: Literal["ok", "lost_ack_after_commit"] = "ok",
) -> type[BaseHTTPRequestHandler]:
    """Build a threaded HTTP handler backed by the shared oracle."""

    class _Handler(BaseHTTPRequestHandler):
        _lost_ack_committed: threading.Event | None = None
        _lost_ack_release: threading.Event | None = None
        _redrive_ok = False

        def log_message(self, format: str, *args: object) -> None:
            return

        def do_POST(self) -> None:
            import socket

            key = self.headers.get("Idempotency-Key", "").strip()
            length = int(self.headers.get("Content-Length", "0") or "0")
            body = self.rfile.read(length) if length else b""
            try:
                disposition = oracle.apply_post(idempotency_key=key, body=body)
            except ClassBHttpProviderOracleConflictError:
                self.send_response(409)
                self.end_headers()
                return
            except ClassBHttpProviderOracleRetentionExceededError:
                self.send_response(503)
                self.end_headers()
                return
            if disposition.http_status != 200:
                self.send_response(disposition.http_status)
                self.end_headers()
                return
            if mode == "lost_ack_after_commit" and not type(self)._redrive_ok:
                committed = type(self)._lost_ack_committed
                release = type(self)._lost_ack_release
                if committed is not None:
                    committed.set()
                if release is not None:
                    release.wait(timeout=15.0)
                self.request.shutdown(socket.SHUT_RDWR)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            assert disposition.payload is not None
            self.wfile.write(json.dumps(disposition.payload).encode("utf-8"))

    return _Handler
