# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""SQLite durable decision event append store (W3-C)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from intergrax.contracts.decision_event_append import (
    DecisionEvent,
    DuplicateDecisionEventError,
    StaleDecisionEventAppendError,
)
from intergrax.contracts.decision_finalization import DecisionFinalizationKey
from intergrax.contracts.decision_identity import validate_decision_id
from intergrax.knowledge.contracts.validation import validate_json_value
from intergrax.runtime.execution.decision_event_payload_codec import (
    DecisionEventPayloadCodecRegistry,
)
from intergrax.runtime.execution.decision_event_record import DecisionEventRecord


def _stream_key_row(key: DecisionFinalizationKey) -> tuple[str, str, str, str]:
    return (
        key.tenant_id,
        str(key.decision_id),
        key.scope.namespace,
        key.scope.subject,
    )


class SQLiteDecisionEventAppendPersistence:
    """Durable compare-and-append decision event stream per finalization key."""

    __slots__ = ("_db_path", "_payload_codecs")

    def __init__(
        self,
        *,
        db_path: Path,
        payload_codecs: DecisionEventPayloadCodecRegistry,
    ) -> None:
        self._db_path = db_path
        self._payload_codecs = payload_codecs
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=FULL;")
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS decision_event_stream_head (
                    tenant_id TEXT NOT NULL,
                    decision_id TEXT NOT NULL,
                    scope_namespace TEXT NOT NULL,
                    scope_subject TEXT NOT NULL,
                    last_sequence INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (
                        tenant_id,
                        decision_id,
                        scope_namespace,
                        scope_subject
                    )
                );
                """,
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS decision_events (
                    tenant_id TEXT NOT NULL,
                    decision_id TEXT NOT NULL,
                    scope_namespace TEXT NOT NULL,
                    scope_subject TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    event_sequence INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    occurred_at_utc TEXT NOT NULL,
                    payload_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (
                        tenant_id,
                        decision_id,
                        scope_namespace,
                        scope_subject,
                        event_id
                    ),
                    UNIQUE (
                        tenant_id,
                        decision_id,
                        scope_namespace,
                        scope_subject,
                        event_sequence
                    )
                );
                """,
            )

    def last_sequence(self, *, key: DecisionFinalizationKey) -> int:
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT last_sequence
                FROM decision_event_stream_head
                WHERE tenant_id = ? AND decision_id = ?
                  AND scope_namespace = ? AND scope_subject = ?
                """,
                _stream_key_row(key),
            ).fetchone()
        if row is None:
            return 0
        return int(row["last_sequence"])

    def append(
        self,
        *,
        key: DecisionFinalizationKey,
        event: DecisionEvent,
        expected_last_sequence: int,
    ) -> DecisionEvent:
        if str(event.decision_id) != str(key.decision_id):
            raise ValueError("decision event decision_id does not match finalization key")
        payload_type, payload_wire = self._payload_codecs.encode(event.payload)
        payload_json = json.dumps(payload_wire, separators=(",", ":"), sort_keys=True)
        key_row = _stream_key_row(key)

        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                """
                SELECT event_sequence, event_type, occurred_at_utc,
                       payload_type, payload_json
                FROM decision_events
                WHERE tenant_id = ? AND decision_id = ?
                  AND scope_namespace = ? AND scope_subject = ?
                  AND event_id = ?
                """,
                (*key_row, event.event_id),
            ).fetchone()
            if existing is not None:
                if (
                    existing["event_type"] == event.event_type
                    and existing["occurred_at_utc"] == event.occurred_at_utc
                    and existing["payload_type"] == payload_type
                    and existing["payload_json"] == payload_json
                ):
                    replay = self._row_to_record(
                        key=key,
                        event_id=event.event_id,
                        row=existing,
                    )
                    conn.rollback()
                    return replay
                conn.rollback()
                raise DuplicateDecisionEventError(
                    f"event_id={event.event_id!r} already exists with different payload",
                )

            head = conn.execute(
                """
                SELECT last_sequence
                FROM decision_event_stream_head
                WHERE tenant_id = ? AND decision_id = ?
                  AND scope_namespace = ? AND scope_subject = ?
                """,
                key_row,
            ).fetchone()
            current_last = int(head["last_sequence"]) if head is not None else 0
            if current_last != expected_last_sequence:
                conn.rollback()
                raise StaleDecisionEventAppendError(
                    f"expected last_sequence={expected_last_sequence}, actual={current_last}",
                )

            next_sequence = expected_last_sequence + 1
            if head is None:
                conn.execute(
                    """
                    INSERT INTO decision_event_stream_head (
                        tenant_id, decision_id, scope_namespace, scope_subject,
                        last_sequence
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (*key_row, next_sequence),
                )
            else:
                updated = conn.execute(
                    """
                    UPDATE decision_event_stream_head
                    SET last_sequence = ?
                    WHERE tenant_id = ? AND decision_id = ?
                      AND scope_namespace = ? AND scope_subject = ?
                      AND last_sequence = ?
                    """,
                    (next_sequence, *key_row, expected_last_sequence),
                )
                if updated.rowcount != 1:
                    conn.rollback()
                    raise StaleDecisionEventAppendError(
                        "concurrent decision event append lost stream head CAS",
                    )

            try:
                conn.execute(
                    """
                    INSERT INTO decision_events (
                        tenant_id, decision_id, scope_namespace, scope_subject,
                        event_id, event_sequence, event_type, occurred_at_utc,
                        payload_type, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        *key_row,
                        event.event_id,
                        next_sequence,
                        event.event_type,
                        event.occurred_at_utc,
                        payload_type,
                        payload_json,
                    ),
                )
            except sqlite3.IntegrityError:
                conn.rollback()
                raise StaleDecisionEventAppendError(
                    "concurrent decision event append lost sequence uniqueness",
                )
            conn.commit()

        return DecisionEventRecord(
            event_id=event.event_id,
            decision_id=validate_decision_id(str(key.decision_id)),
            event_sequence=next_sequence,
            event_type=event.event_type,
            occurred_at_utc=event.occurred_at_utc,
            payload=event.payload,
        )

    def _row_to_record(
        self,
        *,
        key: DecisionFinalizationKey,
        event_id: str,
        row: sqlite3.Row,
    ) -> DecisionEventRecord:
        wire = validate_json_value(
            json.loads(row["payload_json"]),
            field_name="decision_event.payload_json",
        )
        payload = self._payload_codecs.decode(
            payload_type=row["payload_type"],
            payload=wire,
        )
        return DecisionEventRecord(
            event_id=event_id,
            decision_id=validate_decision_id(str(key.decision_id)),
            event_sequence=int(row["event_sequence"]),
            event_type=row["event_type"],
            occurred_at_utc=row["occurred_at_utc"],
            payload=payload,
        )
