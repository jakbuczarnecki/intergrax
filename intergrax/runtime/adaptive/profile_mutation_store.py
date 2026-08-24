# © Artur Czarnecki. All rights reserved.

"""Atomic AHI profile apply/rollback persistence (CLA-CPM-AHI Option A+)."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Protocol

from intergrax.runtime.adaptive.contracts import (
    ProfileArtifactType,
    ProfileVersionRecord,
    ProfileVersionStatus,
)
from intergrax.runtime.adaptive.profile_lifecycle import (
    ProfileLifecycleTransitionError,
    validate_profile_transition,
)
from intergrax.runtime.adaptive.profile_pointer_store import (
    ProfileActivePointer,
    ProfileActivePointerConflictError,
    InMemoryProfileActivePointerStore,
)
from intergrax.runtime.adaptive.profile_version_store import InMemoryProfileVersionStore


def default_adaptive_profile_db_path(repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parents[3]
    return root / "build" / "adaptive_harness" / "adaptive_harness.db"


class AdaptiveProfileMutationStore(Protocol):
    """Domain-local atomic mutation contract for profile lifecycle + active pointer."""

    def commit_apply(
        self,
        *,
        tenant_id: str,
        task_class: str,
        artifact_type: ProfileArtifactType,
        target_version_id: str,
        expected_active_version_id: str | None,
    ) -> ProfileActivePointer: ...

    def commit_rollback(
        self,
        *,
        tenant_id: str,
        task_class: str,
        artifact_type: ProfileArtifactType,
        expected_active_version_id: str,
    ) -> ProfileActivePointer: ...


def _validate_scoped_version(
    record: ProfileVersionRecord,
    *,
    tenant_id: str,
    task_class: str,
    artifact_type: ProfileArtifactType,
) -> None:
    if record.tenant_id != tenant_id:
        raise ValueError(
            "Profile version tenant mismatch: "
            f"expected {tenant_id!r}, got {record.tenant_id!r}"
        )
    if record.task_class != task_class:
        raise ValueError(
            "Profile version task_class mismatch: "
            f"expected {task_class!r}, got {record.task_class!r}"
        )
    if record.artifact_type != artifact_type:
        raise ValueError(
            "Profile version artifact_type mismatch: "
            f"expected {artifact_type.value!r}, got {record.artifact_type.value!r}"
        )


def _plan_apply_status_transitions(
    target_status: ProfileVersionStatus,
) -> list[ProfileVersionStatus]:
    if target_status == ProfileVersionStatus.SHADOW:
        validate_profile_transition(
            current=ProfileVersionStatus.SHADOW,
            target=ProfileVersionStatus.CANARY,
        )
        validate_profile_transition(
            current=ProfileVersionStatus.CANARY,
            target=ProfileVersionStatus.ACTIVE,
        )
        return [ProfileVersionStatus.CANARY, ProfileVersionStatus.ACTIVE]
    if target_status == ProfileVersionStatus.CANARY:
        validate_profile_transition(
            current=ProfileVersionStatus.CANARY,
            target=ProfileVersionStatus.ACTIVE,
        )
        return [ProfileVersionStatus.ACTIVE]
    validate_profile_transition(
        current=target_status,
        target=ProfileVersionStatus.ACTIVE,
    )
    raise ValueError(
        f"Apply requires shadow or canary status, got {target_status.value}"
    )


class InMemoryAdaptiveProfileMutationStore:
    """In-memory atomic mutation store for unit tests."""

    def __init__(
        self,
        *,
        version_store: InMemoryProfileVersionStore,
        pointer_store: InMemoryProfileActivePointerStore,
    ) -> None:
        self._version_store = version_store
        self._pointer_store = pointer_store

    def commit_apply(
        self,
        *,
        tenant_id: str,
        task_class: str,
        artifact_type: ProfileArtifactType,
        target_version_id: str,
        expected_active_version_id: str | None,
    ) -> ProfileActivePointer:
        pointer = self._pointer_store.get_pointer(
            tenant_id=tenant_id,
            task_class=task_class,
            artifact_type=artifact_type,
        )
        actual_active = pointer.active_version_id if pointer is not None else None
        if actual_active != expected_active_version_id:
            raise ProfileActivePointerConflictError(
                "active profile pointer changed before mutation"
            )

        target = self._version_store.get(target_version_id)
        if target is None:
            raise ValueError(f"Unknown profile version: {target_version_id}")
        _validate_scoped_version(
            target,
            tenant_id=tenant_id,
            task_class=task_class,
            artifact_type=artifact_type,
        )
        transitions = _plan_apply_status_transitions(target.status)

        retire_version_id: str | None = None
        if actual_active is not None and actual_active != target_version_id:
            previous = self._version_store.get(actual_active)
            if previous is not None and previous.status == ProfileVersionStatus.ACTIVE:
                validate_profile_transition(
                    current=ProfileVersionStatus.ACTIVE,
                    target=ProfileVersionStatus.RETIRED,
                )
                retire_version_id = previous.version_id

        current = target
        for next_status in transitions:
            validate_profile_transition(current=current.status, target=next_status)
            current = self._version_store.save_status(
                current.model_copy(update={"status": next_status})
            )

        if retire_version_id is not None:
            previous = self._version_store.get(retire_version_id)
            if previous is not None:
                self._version_store.save_status(
                    previous.model_copy(update={"status": ProfileVersionStatus.RETIRED})
                )

        return self._pointer_store.swap_active(
            tenant_id=tenant_id,
            task_class=task_class,
            artifact_type=artifact_type,
            new_active_version_id=current.version_id,
            expected_active_version_id=expected_active_version_id,
        )

    def commit_rollback(
        self,
        *,
        tenant_id: str,
        task_class: str,
        artifact_type: ProfileArtifactType,
        expected_active_version_id: str,
    ) -> ProfileActivePointer:
        pointer = self._pointer_store.get_pointer(
            tenant_id=tenant_id,
            task_class=task_class,
            artifact_type=artifact_type,
        )
        if pointer is None or pointer.previous_version_id is None:
            raise ValueError("No rollback pointer available for active profile version")
        if pointer.active_version_id != expected_active_version_id:
            raise ProfileActivePointerConflictError(
                "active profile pointer changed before rollback"
            )

        current = self._version_store.get(pointer.active_version_id)
        previous = self._version_store.get(pointer.previous_version_id)
        if current is None or previous is None:
            raise ValueError("Rollback requires both current and previous profile versions")

        if current.status == ProfileVersionStatus.ACTIVE:
            validate_profile_transition(
                current=ProfileVersionStatus.ACTIVE,
                target=ProfileVersionStatus.DRAFT,
            )
        validate_profile_transition(
            current=previous.status,
            target=ProfileVersionStatus.ACTIVE,
        )

        if current.status == ProfileVersionStatus.ACTIVE:
            self._version_store.save_status(
                current.model_copy(update={"status": ProfileVersionStatus.DRAFT})
            )
        restored = self._version_store.save_status(
            previous.model_copy(update={"status": ProfileVersionStatus.ACTIVE})
        )
        return self._pointer_store.swap_active(
            tenant_id=tenant_id,
            task_class=task_class,
            artifact_type=artifact_type,
            new_active_version_id=restored.version_id,
            expected_active_version_id=expected_active_version_id,
        )


class SQLiteAdaptiveProfileMutationStore:
    """SQLite-backed atomic mutation store on unified adaptive profile database."""

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

    def _read_pointer(
        self,
        conn: sqlite3.Connection,
        *,
        tenant_id: str,
        task_class: str,
        artifact_type: ProfileArtifactType,
    ) -> ProfileActivePointer | None:
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

    def _read_version(
        self,
        conn: sqlite3.Connection,
        version_id: str,
    ) -> ProfileVersionRecord | None:
        row = conn.execute(
            "SELECT payload_json FROM profile_versions WHERE version_id = ?",
            (version_id,),
        ).fetchone()
        if row is None:
            return None
        return ProfileVersionRecord.model_validate_json(row["payload_json"])

    def _update_version_status(
        self,
        conn: sqlite3.Connection,
        version_id: str,
        target: ProfileVersionStatus,
    ) -> ProfileVersionRecord:
        record = self._read_version(conn, version_id)
        if record is None:
            raise ValueError(f"Unknown profile version: {version_id}")
        validate_profile_transition(current=record.status, target=target)
        updated = record.model_copy(update={"status": target})
        conn.execute(
            """
            UPDATE profile_versions
            SET status = ?, payload_json = ?
            WHERE version_id = ?
            """,
            (target.value, updated.model_dump_json(), version_id),
        )
        return updated

    def _write_pointer(
        self,
        conn: sqlite3.Connection,
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

    def commit_apply(
        self,
        *,
        tenant_id: str,
        task_class: str,
        artifact_type: ProfileArtifactType,
        target_version_id: str,
        expected_active_version_id: str | None,
    ) -> ProfileActivePointer:
        conn = self._connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            pointer = self._read_pointer(
                conn,
                tenant_id=tenant_id,
                task_class=task_class,
                artifact_type=artifact_type,
            )
            actual_active = pointer.active_version_id if pointer is not None else None
            if actual_active != expected_active_version_id:
                raise ProfileActivePointerConflictError(
                    "active profile pointer changed before mutation"
                )

            target = self._read_version(conn, target_version_id)
            if target is None:
                raise ValueError(f"Unknown profile version: {target_version_id}")
            _validate_scoped_version(
                target,
                tenant_id=tenant_id,
                task_class=task_class,
                artifact_type=artifact_type,
            )
            transitions = _plan_apply_status_transitions(target.status)

            retire_version_id: str | None = None
            if actual_active is not None and actual_active != target_version_id:
                previous = self._read_version(conn, actual_active)
                if previous is not None and previous.status == ProfileVersionStatus.ACTIVE:
                    validate_profile_transition(
                        current=ProfileVersionStatus.ACTIVE,
                        target=ProfileVersionStatus.RETIRED,
                    )
                    retire_version_id = previous.version_id

            active_version_id = target_version_id
            for next_status in transitions:
                updated = self._update_version_status(conn, active_version_id, next_status)
                active_version_id = updated.version_id

            if retire_version_id is not None:
                self._update_version_status(
                    conn,
                    retire_version_id,
                    ProfileVersionStatus.RETIRED,
                )

            result = self._write_pointer(
                conn,
                tenant_id=tenant_id,
                task_class=task_class,
                artifact_type=artifact_type,
                new_active_version_id=active_version_id,
                expected_active_version_id=expected_active_version_id,
            )
            conn.execute("COMMIT")
            return result
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def commit_rollback(
        self,
        *,
        tenant_id: str,
        task_class: str,
        artifact_type: ProfileArtifactType,
        expected_active_version_id: str,
    ) -> ProfileActivePointer:
        conn = self._connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            pointer = self._read_pointer(
                conn,
                tenant_id=tenant_id,
                task_class=task_class,
                artifact_type=artifact_type,
            )
            if pointer is None or pointer.previous_version_id is None:
                raise ValueError("No rollback pointer available for active profile version")
            if pointer.active_version_id != expected_active_version_id:
                raise ProfileActivePointerConflictError(
                    "active profile pointer changed before rollback"
                )

            current = self._read_version(conn, pointer.active_version_id)
            previous = self._read_version(conn, pointer.previous_version_id)
            if current is None or previous is None:
                raise ValueError("Rollback requires both current and previous profile versions")

            if current.status == ProfileVersionStatus.ACTIVE:
                validate_profile_transition(
                    current=ProfileVersionStatus.ACTIVE,
                    target=ProfileVersionStatus.DRAFT,
                )
            validate_profile_transition(
                current=previous.status,
                target=ProfileVersionStatus.ACTIVE,
            )

            if current.status == ProfileVersionStatus.ACTIVE:
                self._update_version_status(
                    conn,
                    current.version_id,
                    ProfileVersionStatus.DRAFT,
                )
            restored = self._update_version_status(
                conn,
                previous.version_id,
                ProfileVersionStatus.ACTIVE,
            )
            result = self._write_pointer(
                conn,
                tenant_id=tenant_id,
                task_class=task_class,
                artifact_type=artifact_type,
                new_active_version_id=restored.version_id,
                expected_active_version_id=expected_active_version_id,
            )
            conn.execute("COMMIT")
            return result
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()
