# © Artur Czarnecki. All rights reserved.

"""Active profile version pointer store (Phase W-ADAPT-4.4–4.5)."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from intergrax.runtime.adaptive.contracts import ProfileArtifactType


class ProfileActivePointerConflictError(Exception):
    """Concurrent or stale active profile pointer mutation."""


class ProfileActivePointer(BaseModel):
    """Active version pointer with rollback reference."""

    model_config = ConfigDict(extra="forbid")

    tenant_id: str
    task_class: str
    artifact_type: ProfileArtifactType
    active_version_id: str
    previous_version_id: str | None = None


class ProfileActivePointerStore(Protocol):
    """Tracks active profile version pointers per tenant/task/artifact."""

    def get_pointer(
        self,
        *,
        tenant_id: str,
        task_class: str,
        artifact_type: ProfileArtifactType,
    ) -> ProfileActivePointer | None: ...

    def swap_active(
        self,
        *,
        tenant_id: str,
        task_class: str,
        artifact_type: ProfileArtifactType,
        new_active_version_id: str,
        expected_active_version_id: str | None,
    ) -> ProfileActivePointer: ...

    def clear(self) -> None: ...


def default_adaptive_profile_db_path(repo_root: Path | None = None) -> Path:
    from intergrax.runtime.adaptive.profile_mutation_store import (
        default_adaptive_profile_db_path as _default_path,
    )

    return _default_path(repo_root)


class InMemoryProfileActivePointerStore:
    """In-process active pointer store for unit tests."""

    def __init__(self) -> None:
        self._pointers: dict[tuple[str, str, ProfileArtifactType], ProfileActivePointer] = {}

    def _key(
        self,
        *,
        tenant_id: str,
        task_class: str,
        artifact_type: ProfileArtifactType,
    ) -> tuple[str, str, ProfileArtifactType]:
        return (tenant_id, task_class, artifact_type)

    def get_pointer(
        self,
        *,
        tenant_id: str,
        task_class: str,
        artifact_type: ProfileArtifactType,
    ) -> ProfileActivePointer | None:
        return self._pointers.get(self._key(
            tenant_id=tenant_id,
            task_class=task_class,
            artifact_type=artifact_type,
        ))

    def swap_active(
        self,
        *,
        tenant_id: str,
        task_class: str,
        artifact_type: ProfileArtifactType,
        new_active_version_id: str,
        expected_active_version_id: str | None,
    ) -> ProfileActivePointer:
        key = self._key(
            tenant_id=tenant_id,
            task_class=task_class,
            artifact_type=artifact_type,
        )
        existing = self._pointers.get(key)
        actual_active = existing.active_version_id if existing else None
        if actual_active != expected_active_version_id:
            raise ProfileActivePointerConflictError(
                "active profile pointer changed before swap"
            )
        pointer = ProfileActivePointer(
            tenant_id=tenant_id,
            task_class=task_class,
            artifact_type=artifact_type,
            active_version_id=new_active_version_id,
            previous_version_id=existing.active_version_id if existing else None,
        )
        self._pointers[key] = pointer
        return pointer

    def clear(self) -> None:
        self._pointers.clear()


class SQLiteProfileActivePointerStore:
    """SQLite-backed active pointer store."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or default_adaptive_profile_db_path()
        self._ensure_schema()

    def _connection(self) -> sqlite3.Connection:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS profile_active_pointers (
                    tenant_id TEXT NOT NULL,
                    task_class TEXT NOT NULL,
                    artifact_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, task_class, artifact_type)
                );
                """
            )

    def get_pointer(
        self,
        *,
        tenant_id: str,
        task_class: str,
        artifact_type: ProfileArtifactType,
    ) -> ProfileActivePointer | None:
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT payload_json FROM profile_active_pointers
                WHERE tenant_id = ? AND task_class = ? AND artifact_type = ?
                """,
                (tenant_id, task_class, artifact_type.value),
            ).fetchone()
        if row is None:
            return None
        return ProfileActivePointer.model_validate_json(row["payload_json"])

    def swap_active(
        self,
        *,
        tenant_id: str,
        task_class: str,
        artifact_type: ProfileArtifactType,
        new_active_version_id: str,
        expected_active_version_id: str | None,
    ) -> ProfileActivePointer:
        pointer = ProfileActivePointer(
            tenant_id=tenant_id,
            task_class=task_class,
            artifact_type=artifact_type,
            active_version_id=new_active_version_id,
            previous_version_id=expected_active_version_id,
        )
        payload = pointer.model_dump_json()
        with self._connection() as conn:
            if expected_active_version_id is None:
                inserted = conn.execute(
                    """
                    INSERT INTO profile_active_pointers (
                        tenant_id, task_class, artifact_type, payload_json
                    )
                    SELECT ?, ?, ?, ?
                    WHERE NOT EXISTS (
                        SELECT 1 FROM profile_active_pointers
                        WHERE tenant_id = ? AND task_class = ? AND artifact_type = ?
                    )
                    """,
                    (
                        tenant_id,
                        task_class,
                        artifact_type.value,
                        payload,
                        tenant_id,
                        task_class,
                        artifact_type.value,
                    ),
                )
                if inserted.rowcount == 0:
                    raise ProfileActivePointerConflictError(
                        "active profile pointer already exists"
                    )
                return pointer

            updated = conn.execute(
                """
                UPDATE profile_active_pointers
                SET payload_json = ?
                WHERE tenant_id = ? AND task_class = ? AND artifact_type = ?
                  AND json_extract(payload_json, '$.active_version_id') = ?
                """,
                (
                    payload,
                    tenant_id,
                    task_class,
                    artifact_type.value,
                    expected_active_version_id,
                ),
            )
            if updated.rowcount == 0:
                raise ProfileActivePointerConflictError(
                    "active profile pointer changed before swap"
                )
        return pointer

    def clear(self) -> None:
        with self._connection() as conn:
            conn.execute("DELETE FROM profile_active_pointers")
