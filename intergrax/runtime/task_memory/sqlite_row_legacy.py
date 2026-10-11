# © Artur Czarnecki. All rights reserved.

"""SQLite row legacy read-compat for task memory (COMPAT-X-R2-R2)."""

from __future__ import annotations

import json
import sqlite3

from intergrax.runtime.task_memory.models import SCHEMA_TASK_MEMORY_RECORD_V1, TaskMemoryRecord

LEGACY_UNVERSIONED_PRE_V1 = "LEGACY_UNVERSIONED_PRE_V1"


def task_memory_record_from_sqlite_row(row: sqlite3.Row) -> TaskMemoryRecord:
    """Parse one task_memory row; migrate NULL schema_version after validation."""
    schema_version = row["schema_version"]
    record = TaskMemoryRecord(
        record_id=str(row["record_id"]),
        tenant_id=str(row["tenant_id"]),
        task_id=str(row["task_id"]),
        namespace=str(row["namespace"]),
        key=str(row["key"]),
        value=json.loads(row["value_json"]),
        created_at_utc=str(row["created_at_utc"]),
        updated_at_utc=str(row["updated_at_utc"]),
        provenance=json.loads(row["provenance_json"] or "{}"),
    )
    if schema_version == SCHEMA_TASK_MEMORY_RECORD_V1:
        return record
    if schema_version is None:
        return TaskMemoryRecord.model_validate(
            {**record.model_dump(mode="python"), "schema_version": SCHEMA_TASK_MEMORY_RECORD_V1}
        )
    raise ValueError(
        f"unsupported task memory schema version {schema_version!r}; "
        f"expected {SCHEMA_TASK_MEMORY_RECORD_V1!r} or {LEGACY_UNVERSIONED_PRE_V1}"
    )


def migrated_task_memory_schema_version(schema_version: str | None) -> str:
    if schema_version == SCHEMA_TASK_MEMORY_RECORD_V1:
        return schema_version
    if schema_version is None:
        return SCHEMA_TASK_MEMORY_RECORD_V1
    raise ValueError(
        f"unsupported task memory schema version {schema_version!r}; "
        f"expected {SCHEMA_TASK_MEMORY_RECORD_V1!r} or {LEGACY_UNVERSIONED_PRE_V1}"
    )


__all__ = [
    "LEGACY_UNVERSIONED_PRE_V1",
    "migrated_task_memory_schema_version",
    "task_memory_record_from_sqlite_row",
]
