# © Artur Czarnecki. All rights reserved.

"""Profile version persistence for adaptive harness (Phase W-ADAPT-3.1)."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Protocol

from intergrax.runtime.adaptive.contracts import (
    ProfileArtifactType,
    ProfileVersionDraft,
    ProfileVersionRecord,
    ProfileVersionStatus,
)


class ProfileVersionStore(Protocol):
    """CRUD store for immutable profile version records."""

    def create_from_draft(
        self,
        draft: ProfileVersionDraft,
        *,
        tenant_id: str,
        task_class: str = "",
    ) -> ProfileVersionRecord: ...

    def get(self, version_id: str) -> ProfileVersionRecord | None: ...

    def save_status(self, record: ProfileVersionRecord) -> ProfileVersionRecord: ...

    def list_versions(
        self,
        *,
        tenant_id: str | None = None,
        task_class: str | None = None,
        artifact_type: ProfileArtifactType | None = None,
        status: ProfileVersionStatus | None = None,
        limit: int = 100,
    ) -> list[ProfileVersionRecord]: ...

    def clear(self) -> None: ...


def default_adaptive_profile_db_path(repo_root: Path | None = None) -> Path:
    from intergrax.runtime.adaptive.profile_mutation_store import (
        default_adaptive_profile_db_path as _default_path,
    )

    return _default_path(repo_root)


def _draft_to_record(
    draft: ProfileVersionDraft,
    *,
    tenant_id: str,
    task_class: str,
) -> ProfileVersionRecord:
    return ProfileVersionRecord(
        version_id=draft.version_id,
        tenant_id=tenant_id,
        task_class=task_class,
        artifact_type=draft.artifact_type,
        artifact_payload=dict(draft.artifact_payload),
        parent_version_id=draft.parent_version_id,
        created_by=draft.created_by,
        status=draft.status,
    )


class InMemoryProfileVersionStore:
    """In-process profile version store for unit tests."""

    def __init__(self) -> None:
        self._records: dict[str, ProfileVersionRecord] = {}

    def create_from_draft(
        self,
        draft: ProfileVersionDraft,
        *,
        tenant_id: str,
        task_class: str = "",
    ) -> ProfileVersionRecord:
        if draft.version_id in self._records:
            raise ValueError(f"Profile version already exists: {draft.version_id}")
        record = _draft_to_record(draft, tenant_id=tenant_id, task_class=task_class)
        self._records[record.version_id] = record
        return record

    def get(self, version_id: str) -> ProfileVersionRecord | None:
        return self._records.get(version_id)

    def save_status(self, record: ProfileVersionRecord) -> ProfileVersionRecord:
        existing = self._records.get(record.version_id)
        if existing is None:
            raise ValueError(f"Unknown profile version: {record.version_id}")
        if existing.artifact_payload != record.artifact_payload:
            raise ValueError("Profile version payloads are immutable")
        self._records[record.version_id] = record
        return record

    def list_versions(
        self,
        *,
        tenant_id: str | None = None,
        task_class: str | None = None,
        artifact_type: ProfileArtifactType | None = None,
        status: ProfileVersionStatus | None = None,
        limit: int = 100,
    ) -> list[ProfileVersionRecord]:
        items = list(self._records.values())
        if tenant_id is not None:
            items = [item for item in items if item.tenant_id == tenant_id]
        if task_class is not None:
            items = [item for item in items if item.task_class == task_class]
        if artifact_type is not None:
            items = [item for item in items if item.artifact_type == artifact_type]
        if status is not None:
            items = [item for item in items if item.status == status]
        items.sort(key=lambda item: item.created_at)
        return items[-limit:]

    def clear(self) -> None:
        self._records.clear()


class SQLiteProfileVersionStore:
    """SQLite-backed profile version store under ``build/adaptive_harness/``."""

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
                CREATE TABLE IF NOT EXISTS profile_versions (
                    version_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    task_class TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                """
            )

    def create_from_draft(
        self,
        draft: ProfileVersionDraft,
        *,
        tenant_id: str,
        task_class: str = "",
    ) -> ProfileVersionRecord:
        record = _draft_to_record(draft, tenant_id=tenant_id, task_class=task_class)
        payload = record.model_dump_json()
        with self._connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO profile_versions (
                        version_id, tenant_id, task_class, status, payload_json
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        record.version_id,
                        record.tenant_id,
                        record.task_class,
                        record.status.value,
                        payload,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"Profile version already exists: {record.version_id}") from exc
        return record

    def get(self, version_id: str) -> ProfileVersionRecord | None:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT payload_json FROM profile_versions WHERE version_id = ?",
                (version_id,),
            ).fetchone()
        if row is None:
            return None
        return ProfileVersionRecord.model_validate_json(row["payload_json"])

    def save_status(self, record: ProfileVersionRecord) -> ProfileVersionRecord:
        existing = self.get(record.version_id)
        if existing is None:
            raise ValueError(f"Unknown profile version: {record.version_id}")
        if existing.artifact_payload != record.artifact_payload:
            raise ValueError("Profile version payloads are immutable")
        payload = record.model_dump_json()
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE profile_versions
                SET status = ?, payload_json = ?
                WHERE version_id = ?
                """,
                (record.status.value, payload, record.version_id),
            )
        return record

    def list_versions(
        self,
        *,
        tenant_id: str | None = None,
        task_class: str | None = None,
        artifact_type: ProfileArtifactType | None = None,
        status: ProfileVersionStatus | None = None,
        limit: int = 100,
    ) -> list[ProfileVersionRecord]:
        query = "SELECT payload_json FROM profile_versions WHERE 1=1"
        params: list[str] = []
        if tenant_id is not None:
            query += " AND tenant_id = ?"
            params.append(tenant_id)
        if task_class is not None:
            query += " AND task_class = ?"
            params.append(task_class)
        if status is not None:
            query += " AND status = ?"
            params.append(status.value)
        query += " ORDER BY version_id ASC"
        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()
        records = [ProfileVersionRecord.model_validate_json(row["payload_json"]) for row in rows]
        if artifact_type is not None:
            records = [record for record in records if record.artifact_type == artifact_type]
        return records[-max(1, limit) :]

    def clear(self) -> None:
        with self._connection() as conn:
            conn.execute("DELETE FROM profile_versions")
