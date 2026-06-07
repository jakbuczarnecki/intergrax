# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tenant-scoped task index stored in ``DistributedKVStore`` (broker-backed queues)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import List, Optional

from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.queueing.contracts.task_queue import TaskStatus, TaskSummary

TASK_INDEX_KEY = "__intergrax_task_index_v1__"
MAX_INDEX_ENTRIES = 5000
_COMPLETED_STATUSES = frozenset({TaskStatus.SUCCEEDED, TaskStatus.FAILED})


@dataclass(frozen=True)
class TaskIndexRecord:
    task_id: str
    task_name: str
    provider: str
    status: str
    updated_at_epoch: float


def _status_key(task_id: str) -> str:
    return f"task:{task_id}:status"


def _result_key(task_id: str) -> str:
    return f"task:{task_id}:result"


def load_task_index(kv_store: DistributedKVStore, tenant_id: str) -> List[TaskIndexRecord]:
    raw = kv_store.get(tenant_id, TASK_INDEX_KEY)
    if raw is None:
        return []
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    records: list[TaskIndexRecord] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        task_id = str(item.get("task_id") or "").strip()
        if not task_id:
            continue
        records.append(
            TaskIndexRecord(
                task_id=task_id,
                task_name=str(item.get("task_name") or ""),
                provider=str(item.get("provider") or ""),
                status=str(item.get("status") or TaskStatus.PENDING.value),
                updated_at_epoch=float(item.get("updated_at_epoch") or 0.0),
            )
        )
    records.sort(key=lambda row: row.updated_at_epoch, reverse=True)
    return records


def save_task_index(
    kv_store: DistributedKVStore,
    tenant_id: str,
    records: List[TaskIndexRecord],
) -> None:
    trimmed = records[:MAX_INDEX_ENTRIES]
    payload = [
        {
            "task_id": row.task_id,
            "task_name": row.task_name,
            "provider": row.provider,
            "status": row.status,
            "updated_at_epoch": row.updated_at_epoch,
        }
        for row in trimmed
    ]
    kv_store.set(tenant_id, TASK_INDEX_KEY, json.dumps(payload).encode("utf-8"))


def record_task_index(
    kv_store: DistributedKVStore,
    *,
    tenant_id: str,
    task_id: str,
    task_name: str,
    provider: str,
    status: TaskStatus,
    updated_at_epoch: float | None = None,
) -> None:
    now = updated_at_epoch if updated_at_epoch is not None else time.time()
    records = load_task_index(kv_store, tenant_id)
    updated = [
        row
        for row in records
        if not (row.task_id == task_id and row.provider == provider)
    ]
    updated.insert(
        0,
        TaskIndexRecord(
            task_id=task_id,
            task_name=task_name,
            provider=provider,
            status=status.value,
            updated_at_epoch=now,
        ),
    )
    save_task_index(kv_store, tenant_id, updated)


def update_task_index_status(
    kv_store: DistributedKVStore,
    *,
    tenant_id: str,
    task_id: str,
    provider: str,
    status: TaskStatus,
) -> None:
    now = time.time()
    records = load_task_index(kv_store, tenant_id)
    updated: list[TaskIndexRecord] = []
    found = False
    for record in records:
        if record.task_id == task_id and record.provider == provider:
            updated.append(
                TaskIndexRecord(
                    task_id=record.task_id,
                    task_name=record.task_name,
                    provider=record.provider,
                    status=status.value,
                    updated_at_epoch=now,
                )
            )
            found = True
        else:
            updated.append(record)
    if not found:
        record_task_index(
            kv_store,
            tenant_id=tenant_id,
            task_id=task_id,
            task_name="",
            provider=provider,
            status=status,
            updated_at_epoch=now,
        )
        return
    save_task_index(kv_store, tenant_id, updated)


def list_tasks_from_index(
    kv_store: DistributedKVStore,
    tenant_id: str,
    *,
    provider: str,
    limit: int = 50,
    status_filter: Optional[TaskStatus] = None,
) -> List[TaskSummary]:
    rows: list[TaskSummary] = []
    for record in load_task_index(kv_store, tenant_id):
        if record.provider != provider:
            continue
        status = TaskStatus(record.status)
        if status_filter is not None and status != status_filter:
            continue
        rows.append(
            TaskSummary(
                task_id=record.task_id,
                tenant_id=tenant_id,
                task_name=record.task_name,
                status=status,
                provider=record.provider,
            )
        )
        if len(rows) >= limit:
            break
    return rows


def purge_completed_tasks_from_index(
    kv_store: DistributedKVStore,
    *,
    tenant_id: str,
    provider: str,
    older_than_seconds: int = 0,
) -> int:
    now = time.time()
    records = load_task_index(kv_store, tenant_id)
    kept: list[TaskIndexRecord] = []
    purged = 0
    for record in records:
        if record.provider != provider:
            kept.append(record)
            continue
        try:
            status = TaskStatus(record.status)
        except ValueError:
            kept.append(record)
            continue
        if status not in _COMPLETED_STATUSES:
            kept.append(record)
            continue
        if older_than_seconds > 0 and (now - record.updated_at_epoch) < older_than_seconds:
            kept.append(record)
            continue
        kv_store.delete(tenant_id, _status_key(record.task_id))
        kv_store.delete(tenant_id, _result_key(record.task_id))
        purged += 1
    save_task_index(kv_store, tenant_id, kept)
    return purged
