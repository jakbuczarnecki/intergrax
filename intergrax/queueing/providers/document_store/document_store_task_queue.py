# © Artur Czarnecki. All rights reserved.

"""Durable TaskQueue backed by the provider-neutral DocumentStore contract.

Solves single-host product failure where HTTP-owned ``asyncio.create_task`` work
is lost on process restart. Reuses DocumentStore (MongoDB / in-memory / …) rather
than inventing an application-specific queue. Belongs below Tier-3 because any
product host can enqueue through the shared MessageBus / TaskQueue contract.
"""

from __future__ import annotations

import base64
import threading
import time
import uuid
from typing import List, Optional

from intergrax.integrations.contracts.document_store import DocumentRecord, DocumentStore
from intergrax.queueing.contracts.task_queue import (
    TaskHandle,
    TaskQueue,
    TaskRequest,
    TaskResult,
    TaskStatus,
    TaskSummary,
)

DOCUMENT_STORE_TASK_QUEUE_PROVIDER = "document_store"
_PARTITION_PREFIX = "intergrax.task_queue.v1"
_PENDING_INDEX_PARTITION = f"{_PARTITION_PREFIX}:__pending_index__"
_RUNNING_INDEX_PARTITION = f"{_PARTITION_PREFIX}:__running_index__"
_IDEMPOTENCY_PREFIX = "idempotency:"


def _partition(tenant_id: str) -> str:
    return f"{_PARTITION_PREFIX}:{tenant_id}"


def _index_row(tenant_id: str, task_id: str) -> str:
    return f"{tenant_id}:{task_id}"


class DocumentStoreTaskQueue(TaskQueue):
    """Crash-safe local MessageBus using DocumentStore as the durable log."""

    def __init__(self, document_store: DocumentStore) -> None:
        self._store = document_store
        self._lock = threading.RLock()

    @property
    def document_store(self) -> DocumentStore:
        return self._store

    def enqueue(self, request: TaskRequest) -> TaskHandle:
        with self._lock:
            if request.idempotency_key:
                existing_id = self._lookup_idempotency(
                    tenant_id=request.tenant_id,
                    idempotency_key=request.idempotency_key,
                )
                if existing_id:
                    return TaskHandle(
                        task_id=existing_id,
                        provider=DOCUMENT_STORE_TASK_QUEUE_PROVIDER,
                        tenant_id=request.tenant_id,
                    )

            task_id = f"dstq_{uuid.uuid4().hex}"
            now = time.time()
            payload = {
                "task_id": task_id,
                "tenant_id": request.tenant_id,
                "run_id": request.run_id,
                "task_name": request.task_name,
                "payload_base64": base64.b64encode(request.payload).decode("ascii"),
                "idempotency_key": request.idempotency_key,
                "status": TaskStatus.PENDING.value,
                "attempts": 0,
                "created_at_epoch": now,
                "updated_at_epoch": now,
                "error_message": None,
                "output_base64": None,
            }
            self._store.put(
                DocumentRecord(
                    partition_key=_partition(request.tenant_id),
                    row_key=task_id,
                    data=payload,
                )
            )
            self._store.put(
                DocumentRecord(
                    partition_key=_PENDING_INDEX_PARTITION,
                    row_key=_index_row(request.tenant_id, task_id),
                    data={"tenant_id": request.tenant_id, "task_id": task_id},
                )
            )
            if request.idempotency_key:
                self._store.put(
                    DocumentRecord(
                        partition_key=_partition(request.tenant_id),
                        row_key=f"{_IDEMPOTENCY_PREFIX}{request.idempotency_key}",
                        data={"task_id": task_id},
                    )
                )
            return TaskHandle(
                task_id=task_id,
                provider=DOCUMENT_STORE_TASK_QUEUE_PROVIDER,
                tenant_id=request.tenant_id,
            )

    def get_status(self, handle: TaskHandle) -> TaskStatus:
        record = self._load(handle)
        if record is None:
            return TaskStatus.PENDING
        return TaskStatus(str(record["status"]))

    def get_status_if_present(self, handle: TaskHandle) -> TaskStatus | None:
        record = self._load(handle)
        if record is None:
            return None
        return TaskStatus(str(record["status"]))

    def get_result(self, handle: TaskHandle) -> Optional[TaskResult]:
        record = self._load(handle)
        if record is None:
            return None
        status = TaskStatus(str(record["status"]))
        if status not in {TaskStatus.SUCCEEDED, TaskStatus.FAILED}:
            return None
        output: Optional[bytes] = None
        encoded = record.get("output_base64")
        if isinstance(encoded, str) and encoded:
            output = base64.b64decode(encoded.encode("ascii"))
        error = record.get("error_message")
        return TaskResult(
            status=status,
            output=output,
            error_message=str(error) if error else None,
            attempts=int(record.get("attempts") or 0),
        )

    def list_tasks(
        self,
        tenant_id: str,
        *,
        limit: int = 100,
        status_filter: Optional[TaskStatus] = None,
    ) -> List[TaskSummary]:
        rows = self._list_task_rows(tenant_id=tenant_id, limit=max(limit * 2, 100))
        summaries: list[TaskSummary] = []
        for row in rows:
            status = TaskStatus(str(row["status"]))
            if status_filter is not None and status is not status_filter:
                continue
            summaries.append(
                TaskSummary(
                    task_id=str(row["task_id"]),
                    tenant_id=tenant_id,
                    task_name=str(row["task_name"]),
                    status=status,
                    provider=DOCUMENT_STORE_TASK_QUEUE_PROVIDER,
                )
            )
            if len(summaries) >= limit:
                break
        return summaries

    def cancel(self, handle: TaskHandle) -> bool:
        with self._lock:
            record = self._load(handle)
            if record is None:
                return False
            if TaskStatus(str(record["status"])) is not TaskStatus.PENDING:
                return False
            record["status"] = TaskStatus.FAILED.value
            record["error_message"] = "cancelled"
            record["updated_at_epoch"] = time.time()
            self._put(handle.tenant_id or "", record)
            self._drop_indexes(handle.tenant_id or "", handle.task_id)
            return True

    def claim_pending(
        self,
        *,
        tenant_id: str | None = None,
        limit: int = 8,
    ) -> list[tuple[TaskHandle, TaskRequest]]:
        """Atomically transition PENDING → RUNNING and return claimed work."""
        with self._lock:
            claimed: list[tuple[TaskHandle, TaskRequest]] = []
            pending = self._store.query(_PENDING_INDEX_PARTITION, limit=max(limit * 4, 32))
            for doc in pending.documents:
                tid = str(doc.data.get("tenant_id") or "").strip()
                task_id = str(doc.data.get("task_id") or "").strip()
                if not tid or not task_id:
                    continue
                if tenant_id is not None and tid != tenant_id:
                    continue
                row = self._load(TaskHandle(task_id=task_id, provider=DOCUMENT_STORE_TASK_QUEUE_PROVIDER, tenant_id=tid))
                if row is None or TaskStatus(str(row["status"])) is not TaskStatus.PENDING:
                    self._store.delete(_PENDING_INDEX_PARTITION, _index_row(tid, task_id))
                    continue
                row["status"] = TaskStatus.RUNNING.value
                row["attempts"] = int(row.get("attempts") or 0) + 1
                row["updated_at_epoch"] = time.time()
                self._put(tid, row)
                self._store.delete(_PENDING_INDEX_PARTITION, _index_row(tid, task_id))
                self._store.put(
                    DocumentRecord(
                        partition_key=_RUNNING_INDEX_PARTITION,
                        row_key=_index_row(tid, task_id),
                        data={"tenant_id": tid, "task_id": task_id},
                    )
                )
                handle = TaskHandle(
                    task_id=task_id,
                    provider=DOCUMENT_STORE_TASK_QUEUE_PROVIDER,
                    tenant_id=tid,
                )
                payload = base64.b64decode(str(row["payload_base64"]).encode("ascii"))
                request = TaskRequest(
                    tenant_id=tid,
                    run_id=str(row["run_id"]),
                    task_name=str(row["task_name"]),
                    payload=payload,
                    idempotency_key=(
                        str(row["idempotency_key"]) if row.get("idempotency_key") else None
                    ),
                )
                claimed.append((handle, request))
                if len(claimed) >= limit:
                    break
            return claimed

    def mark_succeeded(
        self,
        handle: TaskHandle,
        *,
        output: bytes | None = None,
    ) -> None:
        with self._lock:
            record = self._load(handle)
            if record is None:
                return
            record["status"] = TaskStatus.SUCCEEDED.value
            record["output_base64"] = (
                base64.b64encode(output).decode("ascii") if output else None
            )
            record["error_message"] = None
            record["updated_at_epoch"] = time.time()
            self._put(handle.tenant_id or "", record)
            self._drop_indexes(handle.tenant_id or "", handle.task_id)

    def mark_failed(self, handle: TaskHandle, *, error_message: str) -> None:
        with self._lock:
            record = self._load(handle)
            if record is None:
                return
            record["status"] = TaskStatus.FAILED.value
            record["error_message"] = error_message[:500]
            record["updated_at_epoch"] = time.time()
            self._put(handle.tenant_id or "", record)
            self._drop_indexes(handle.tenant_id or "", handle.task_id)

    def recover_interrupted_running(
        self,
        *,
        error_message: str = "interrupted_by_host_restart",
    ) -> list[tuple[TaskHandle, TaskRequest]]:
        """Fail-closed recovery: RUNNING messages cannot remain permanently running.

        Returns the interrupted requests so product workers can update domain state.
        """
        with self._lock:
            interrupted: list[tuple[TaskHandle, TaskRequest]] = []
            running = self._store.query(_RUNNING_INDEX_PARTITION, limit=500)
            for doc in running.documents:
                tid = str(doc.data.get("tenant_id") or "").strip()
                task_id = str(doc.data.get("task_id") or "").strip()
                if not tid or not task_id:
                    continue
                handle = TaskHandle(
                    task_id=task_id,
                    provider=DOCUMENT_STORE_TASK_QUEUE_PROVIDER,
                    tenant_id=tid,
                )
                row = self._load(handle)
                if row is not None and TaskStatus(str(row["status"])) is TaskStatus.RUNNING:
                    payload = base64.b64decode(str(row["payload_base64"]).encode("ascii"))
                    request = TaskRequest(
                        tenant_id=tid,
                        run_id=str(row["run_id"]),
                        task_name=str(row["task_name"]),
                        payload=payload,
                        idempotency_key=(
                            str(row["idempotency_key"]) if row.get("idempotency_key") else None
                        ),
                    )
                    interrupted.append((handle, request))
                    row["status"] = TaskStatus.FAILED.value
                    row["error_message"] = error_message
                    row["updated_at_epoch"] = time.time()
                    self._put(tid, row)
                self._store.delete(_RUNNING_INDEX_PARTITION, _index_row(tid, task_id))
            return interrupted

    def _drop_indexes(self, tenant_id: str, task_id: str) -> None:
        self._store.delete(_PENDING_INDEX_PARTITION, _index_row(tenant_id, task_id))
        self._store.delete(_RUNNING_INDEX_PARTITION, _index_row(tenant_id, task_id))

    def _lookup_idempotency(self, *, tenant_id: str, idempotency_key: str) -> str | None:
        record = self._store.get(
            _partition(tenant_id),
            f"{_IDEMPOTENCY_PREFIX}{idempotency_key}",
        )
        if record is None:
            return None
        task_id = str(record.data.get("task_id") or "").strip()
        return task_id or None

    def _load(self, handle: TaskHandle) -> dict[str, object] | None:
        tenant_id = handle.tenant_id
        if not tenant_id:
            return None
        record = self._store.get(_partition(tenant_id), handle.task_id)
        if record is None:
            return None
        return dict(record.data)

    def _put(self, tenant_id: str, row: dict[str, object]) -> None:
        self._store.put(
            DocumentRecord(
                partition_key=_partition(tenant_id),
                row_key=str(row["task_id"]),
                data=row,
            )
        )

    def _list_task_rows(self, *, tenant_id: str, limit: int) -> list[dict[str, object]]:
        result = self._store.query(_partition(tenant_id), limit=limit)
        rows: list[dict[str, object]] = []
        for doc in result.documents:
            if str(doc.row_key).startswith(_IDEMPOTENCY_PREFIX):
                continue
            if "task_id" not in doc.data or "task_name" not in doc.data:
                continue
            rows.append(dict(doc.data))
        rows.sort(key=lambda item: float(item.get("created_at_epoch") or 0.0))
        return rows

    def health(self):  # type: ignore[override]
        from intergrax.integrations.contracts.base import HealthStatus

        return HealthStatus(healthy=True, detail="document_store_task_queue")
