# © Artur Czarnecki. All rights reserved.

"""Active profile version pointer store (Phase W-ADAPT-4.4–4.5)."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from intergrax.runtime.adaptive.contracts import ProfileArtifactType


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
    ) -> ProfileActivePointer: ...

    def clear(self) -> None: ...


def default_profile_pointer_store_path(repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parents[3]
    return root / "build" / "adaptive_harness" / "profile_pointers.db"


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
    ) -> ProfileActivePointer:
        key = self._key(
            tenant_id=tenant_id,
            task_class=task_class,
            artifact_type=artifact_type,
        )
        existing = self._pointers.get(key)
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
        self._db_path = db_path or default_profile_pointer_store_path()
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
    ) -> ProfileActivePointer:
        existing = self.get_pointer(
            tenant_id=tenant_id,
            task_class=task_class,
            artifact_type=artifact_type,
        )
        pointer = ProfileActivePointer(
            tenant_id=tenant_id,
            task_class=task_class,
            artifact_type=artifact_type,
            active_version_id=new_active_version_id,
            previous_version_id=existing.active_version_id if existing else None,
        )
        payload = pointer.model_dump_json()
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO profile_active_pointers (
                    tenant_id, task_class, artifact_type, payload_json
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(tenant_id, task_class, artifact_type)
                DO UPDATE SET payload_json = excluded.payload_json
                """,
                (tenant_id, task_class, artifact_type.value, payload),
            )
        return pointer

    def clear(self) -> None:
        with self._connection() as conn:
            conn.execute("DELETE FROM profile_active_pointers")
