# © Artur Czarnecki. All rights reserved.

"""MP-6D — durable, atomic Collaborative Activity append store (provider adapters)."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable
from datetime import datetime
from typing import Any

from intergrax.collaborative_work.postgresql_repository import PostgreSQLCollaborativeWorkStore
from intergrax.collaborative_work.serialization import (
    collaborative_activity_from_json,
    collaborative_activity_to_json,
)
from intergrax.collaborative_work.sqlite_repository import SQLiteCollaborativeWorkStore
from intergrax.contracts.collaborative_activity import (
    SCHEMA_COLLABORATIVE_ACTIVITY_V1,
    ActivityIdempotencyKey,
    CollaborativeActivity,
    CollaborativeActivityAppendIntent,
    mint_collaborative_activity_id,
)
from intergrax.integrations.providers.relational_store.postgresql.session import (
    is_postgresql_unique_violation,
)
from intergrax.utils.time_provider import SystemTimeProvider

_MAX_APPEND_RETRIES = 3


class CollaborativeActivityAppendPersistenceError(RuntimeError):
    """Append store failed to persist or reconstruct a collaborative activity."""


class CollaborativeActivityAppendIntegrityError(CollaborativeActivityAppendPersistenceError):
    """Persisted collaborative activity data violates canonical contract invariants."""


UtcNow = Callable[[], datetime]


def _idempotency_params(key: ActivityIdempotencyKey) -> tuple[str, str, str, str, str]:
    return (
        key.tenant_id.strip(),
        key.workspace_id.strip(),
        key.source.qualified_id,
        key.source_stable_id.strip(),
        key.activity_type.qualified_id,
    )


def materialize_collaborative_activity_from_intent(
    intent: CollaborativeActivityAppendIntent,
    *,
    append_position: int,
    recorded_at: datetime,
) -> CollaborativeActivity:
    publication = intent.publication
    activity_id = mint_collaborative_activity_id(idempotency_key=publication.idempotency_key)
    return CollaborativeActivity(
        activity_id=activity_id,
        idempotency_key=publication.idempotency_key,
        activity_type=publication.activity_type,
        actor=publication.actor,
        scope=publication.scope,
        target=publication.target,
        outcome=publication.outcome,
        occurred_at=publication.occurred_at,
        recorded_at=recorded_at,
        append_position=append_position,
        provenance_refs=publication.provenance_refs,
        correlation=publication.correlation,
        caused_by_activity_id=publication.caused_by_activity_id,
        durability_class=intent.effective_durability_class,
    )


def _decode_activity_row(record_json: str) -> CollaborativeActivity:
    try:
        return collaborative_activity_from_json(record_json)
    except Exception as exc:
        raise CollaborativeActivityAppendIntegrityError(
            "collaborative activity record_json is corrupt or incompatible"
        ) from exc


class SQLiteCollaborativeActivityAppendStore:
    """SQLite-backed atomic append store — qualified for single-process / file durability."""

    def __init__(
        self,
        store: SQLiteCollaborativeWorkStore,
        *,
        utc_now: UtcNow | None = None,
    ) -> None:
        self._store = store
        self._utc_now = utc_now or SystemTimeProvider.utc_now

    def append_idempotent(self, intent: CollaborativeActivityAppendIntent) -> CollaborativeActivity:
        for attempt in range(_MAX_APPEND_RETRIES):
            try:
                return self._append_once(intent)
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or attempt + 1 >= _MAX_APPEND_RETRIES:
                    raise CollaborativeActivityAppendPersistenceError(
                        "sqlite append store operational failure"
                    ) from exc
        raise CollaborativeActivityAppendPersistenceError("sqlite append store retry budget exhausted")

    def get_by_idempotency_key(self, key: ActivityIdempotencyKey) -> CollaborativeActivity | None:
        with self._store._lock:
            self._store._ensure_open()
            return self._load_by_idempotency_key(self._store.transaction(), key)

    def _append_once(self, intent: CollaborativeActivityAppendIntent) -> CollaborativeActivity:
        key = intent.publication.idempotency_key
        with self._store._lock:
            self._store._ensure_open()
            conn = self._store.transaction()
            conn.execute("BEGIN IMMEDIATE")
            try:
                existing = self._load_by_idempotency_key(conn, key)
                if existing is not None:
                    conn.commit()
                    return existing

                append_position = self._allocate_append_position(conn, key.tenant_id, key.workspace_id)
                recorded_at = self._utc_now()
                activity = materialize_collaborative_activity_from_intent(
                    intent,
                    append_position=append_position,
                    recorded_at=recorded_at,
                )
                self._insert_activity(conn, activity)
                conn.commit()
                return activity
            except sqlite3.IntegrityError as exc:
                conn.rollback()
                replay = self.get_by_idempotency_key(key)
                if replay is not None:
                    return replay
                raise CollaborativeActivityAppendIntegrityError(
                    "collaborative activity idempotency or identity constraint violation"
                ) from exc
            except Exception:
                conn.rollback()
                raise

    def _allocate_append_position(
        self,
        conn: sqlite3.Connection,
        tenant_id: str,
        workspace_id: str,
    ) -> int:
        row = conn.execute(
            """
            INSERT INTO collaborative_activity_workspace_sequence (
                tenant_id, workspace_id, next_append_position
            ) VALUES (?, ?, 2)
            ON CONFLICT(tenant_id, workspace_id) DO UPDATE SET
                next_append_position = next_append_position + 1
            RETURNING next_append_position - 1 AS allocated_position
            """,
            (tenant_id.strip(), workspace_id.strip()),
        ).fetchone()
        if row is None:
            raise CollaborativeActivityAppendPersistenceError(
                "append position allocation returned no row"
            )
        return int(row["allocated_position"])

    def _insert_activity(self, conn: sqlite3.Connection, activity: CollaborativeActivity) -> None:
        key = activity.idempotency_key
        conn.execute(
            """
            INSERT INTO collaborative_activities (
                tenant_id,
                workspace_id,
                activity_id,
                source_qualified_id,
                source_stable_id,
                activity_type_qualified_id,
                append_position,
                recorded_at,
                record_json,
                schema_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                key.tenant_id.strip(),
                key.workspace_id.strip(),
                activity.activity_id.strip(),
                key.source.qualified_id,
                key.source_stable_id.strip(),
                key.activity_type.qualified_id,
                activity.append_position,
                activity.recorded_at.isoformat(),
                collaborative_activity_to_json(activity),
                SCHEMA_COLLABORATIVE_ACTIVITY_V1,
            ),
        )

    def _load_by_idempotency_key(
        self,
        conn: sqlite3.Connection,
        key: ActivityIdempotencyKey,
    ) -> CollaborativeActivity | None:
        row = conn.execute(
            """
            SELECT record_json
            FROM collaborative_activities
            WHERE tenant_id = ? AND workspace_id = ?
              AND source_qualified_id = ? AND source_stable_id = ?
              AND activity_type_qualified_id = ?
            """,
            _idempotency_params(key),
        ).fetchone()
        if row is None:
            return None
        return _decode_activity_row(row["record_json"])


class PostgreSQLCollaborativeActivityAppendStore:
    """PostgreSQL-backed atomic append store — production relational provider."""

    def __init__(
        self,
        store: PostgreSQLCollaborativeWorkStore,
        *,
        utc_now: UtcNow | None = None,
    ) -> None:
        self._store = store
        self._utc_now = utc_now or SystemTimeProvider.utc_now
        self._lock = threading.RLock()

    def append_idempotent(self, intent: CollaborativeActivityAppendIntent) -> CollaborativeActivity:
        key = intent.publication.idempotency_key
        last_error: BaseException | None = None
        for _ in range(_MAX_APPEND_RETRIES):
            try:
                return self._append_once(intent)
            except Exception as exc:
                if not is_postgresql_unique_violation(exc):
                    if isinstance(exc, CollaborativeActivityAppendPersistenceError):
                        raise
                    raise CollaborativeActivityAppendPersistenceError(
                        "postgresql append store failed"
                    ) from exc
                replay = self.get_by_idempotency_key(key)
                if replay is not None:
                    return replay
                last_error = exc
        raise CollaborativeActivityAppendIntegrityError(
            "collaborative activity idempotency or identity constraint violation"
        ) from last_error

    def get_by_idempotency_key(self, key: ActivityIdempotencyKey) -> CollaborativeActivity | None:
        with self._lock:
            self._store._ensure_open()
            with self._store.transaction() as session:
                return self._load_by_idempotency_key(session, key)

    def _append_once(self, intent: CollaborativeActivityAppendIntent) -> CollaborativeActivity:
        key = intent.publication.idempotency_key
        with self._lock:
            self._store._ensure_open()
            with self._store.transaction() as session:
                existing = self._load_by_idempotency_key(session, key)
                if existing is not None:
                    return existing

                append_position = self._allocate_append_position(
                    session,
                    key.tenant_id,
                    key.workspace_id,
                )
                recorded_at = self._utc_now()
                activity = materialize_collaborative_activity_from_intent(
                    intent,
                    append_position=append_position,
                    recorded_at=recorded_at,
                )
                try:
                    self._insert_activity(session, activity)
                except Exception as exc:
                    if is_postgresql_unique_violation(exc):
                        session.rollback()
                        with self._store.transaction() as replay_session:
                            replay = self._load_by_idempotency_key(replay_session, key)
                            if replay is not None:
                                return replay
                    raise
                return activity

    def _allocate_append_position(self, session: Any, tenant_id: str, workspace_id: str) -> int:
        row = session.execute(
            """
            INSERT INTO collaborative_activity_workspace_sequence (
                tenant_id, workspace_id, next_append_position
            ) VALUES (%s, %s, 2)
            ON CONFLICT (tenant_id, workspace_id) DO UPDATE SET
                next_append_position = collaborative_activity_workspace_sequence.next_append_position + 1
            RETURNING next_append_position - 1 AS allocated_position
            """,
            (tenant_id.strip(), workspace_id.strip()),
        ).fetchone()
        if row is None:
            raise CollaborativeActivityAppendPersistenceError(
                "append position allocation returned no row"
            )
        return int(row["allocated_position"])

    def _insert_activity(self, session: Any, activity: CollaborativeActivity) -> None:
        key = activity.idempotency_key
        session.execute(
            """
            INSERT INTO collaborative_activities (
                tenant_id,
                workspace_id,
                activity_id,
                source_qualified_id,
                source_stable_id,
                activity_type_qualified_id,
                append_position,
                recorded_at,
                record_json,
                schema_version
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                key.tenant_id.strip(),
                key.workspace_id.strip(),
                activity.activity_id.strip(),
                key.source.qualified_id,
                key.source_stable_id.strip(),
                key.activity_type.qualified_id,
                activity.append_position,
                activity.recorded_at.isoformat(),
                collaborative_activity_to_json(activity),
                SCHEMA_COLLABORATIVE_ACTIVITY_V1,
            ),
        )

    def _load_by_idempotency_key(
        self,
        session: Any,
        key: ActivityIdempotencyKey,
    ) -> CollaborativeActivity | None:
        row = session.execute(
            """
            SELECT record_json
            FROM collaborative_activities
            WHERE tenant_id = %s AND workspace_id = %s
              AND source_qualified_id = %s AND source_stable_id = %s
              AND activity_type_qualified_id = %s
            """,
            _idempotency_params(key),
        ).fetchone()
        if row is None:
            return None
        return _decode_activity_row(row["record_json"])


__all__ = [
    "CollaborativeActivityAppendIntegrityError",
    "CollaborativeActivityAppendPersistenceError",
    "PostgreSQLCollaborativeActivityAppendStore",
    "SQLiteCollaborativeActivityAppendStore",
    "materialize_collaborative_activity_from_intent",
]
