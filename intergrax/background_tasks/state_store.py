# © Artur Czarnecki. All rights reserved.

"""Shared KV-backed task status/result store for broker-backed queues."""

from __future__ import annotations

import base64
from typing import Optional

from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.queueing.contracts.task_queue import TaskResult, TaskStatus
from intergrax.queueing.task_index import record_task_index, update_task_index_status


class TaskStateStore:
    """Persists task lifecycle status in a distributed KV store."""

    def __init__(self, *, kv_store: DistributedKVStore, provider: str) -> None:
        self._kv_store = kv_store
        self._provider = provider

    @staticmethod
    def _status_key(task_id: str) -> str:
        return f"task:{task_id}:status"

    def get_status(self, *, tenant_id: str, task_id: str) -> TaskStatus:
        raw = self._kv_store.get(tenant_id=tenant_id, key=self._status_key(task_id))
        if raw is None:
            return TaskStatus.PENDING
        return TaskStatus(raw.decode("utf-8"))

    def set_status(
        self,
        *,
        tenant_id: str,
        task_id: str,
        task_name: str,
        status: TaskStatus,
    ) -> None:
        self._kv_store.set(
            tenant_id=tenant_id,
            key=self._status_key(task_id),
            value=status.value.encode("utf-8"),
        )
        record_task_index(
            self._kv_store,
            tenant_id=tenant_id,
            task_id=task_id,
            task_name=task_name,
            provider=self._provider,
            status=status,
        )


class TaskResultStore:
    """Persists terminal task results in a distributed KV store."""

    def __init__(self, *, kv_store: DistributedKVStore) -> None:
        self._kv_store = kv_store

    @staticmethod
    def _result_key(task_id: str) -> str:
        return f"task:{task_id}:result"

    def get_result(self, *, tenant_id: str, task_id: str) -> Optional[TaskResult]:
        raw = self._kv_store.get(tenant_id=tenant_id, key=self._result_key(task_id))
        if raw is None:
            return None
        decoded = raw.decode("utf-8")
        parts = decoded.split("|", 3)
        status = TaskStatus(parts[0])
        attempts = int(parts[1])
        error_message: Optional[str] = None
        if len(parts) >= 3 and parts[2]:
            error_message = parts[2]
        output: Optional[bytes] = None
        if len(parts) == 4 and parts[3]:
            try:
                output = base64.b64decode(parts[3].encode("ascii"))
            except Exception:
                output = None
        return TaskResult(
            status=status,
            output=output,
            error_message=error_message,
            attempts=attempts,
        )

    def store_result(
        self,
        *,
        tenant_id: str,
        task_id: str,
        result: TaskResult,
        encoded_output: str = "",
    ) -> None:
        error = result.error_message or ""
        record = f"{result.status.value}|{result.attempts}|{error}|{encoded_output}"
        self._kv_store.set(
            tenant_id=tenant_id,
            key=self._result_key(task_id),
            value=record.encode("utf-8"),
        )

    def mark_failed(
        self,
        *,
        tenant_id: str,
        task_id: str,
        task_name: str,
        provider: str,
        error_message: str,
    ) -> None:
        self.store_result(
            tenant_id=tenant_id,
            task_id=task_id,
            result=TaskResult(
                status=TaskStatus.FAILED,
                error_message=error_message,
                attempts=1,
            ),
        )
        update_task_index_status(
            self._kv_store,
            tenant_id=tenant_id,
            task_id=task_id,
            provider=provider,
            status=TaskStatus.FAILED,
        )
