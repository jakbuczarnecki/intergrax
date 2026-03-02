# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import base64
from typing import Optional

from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.queueing.contracts.task_queue import (
    TaskQueue,
    TaskHandle,
    TaskStatus,
    TaskResult,
)


class BrokerBackedTaskQueueBase(TaskQueue):
    """
    Base class for broker-backed transports (Kafka, RabbitMQ, etc.).

    These transports do not provide built-in:
    - task status inspection
    - result backend
    - retry metadata

    Therefore status and result are stored in DistributedKVStore.

    Subclasses must implement:
    - enqueue()
    - transport publishing logic
    - worker bootstrap
    """

    def __init__(
        self,
        *,
        kv_store: DistributedKVStore,
        provider_name: str,
    ) -> None:
        self._kv_store: DistributedKVStore = kv_store
        self._provider_name: str = provider_name

    # ------------------------------------------------------------------
    # Storage keys
    # ------------------------------------------------------------------

    def _status_key(self, task_id: str) -> str:
        return f"task:{task_id}:status"

    def _result_key(self, task_id: str) -> str:
        return f"task:{task_id}:result"

    # ------------------------------------------------------------------
    # TaskQueue contract implementation
    # ------------------------------------------------------------------

    def get_status(
        self,
        handle: TaskHandle,
    ) -> TaskStatus:
        if handle.tenant_id is None:
            return TaskStatus.PENDING

        raw = self._kv_store.get(
            tenant_id=handle.tenant_id,
            key=self._status_key(handle.task_id),
        )

        if raw is None:
            return TaskStatus.PENDING

        return TaskStatus(raw.decode("utf-8"))

    def get_result(
        self,
        handle: TaskHandle,
    ) -> Optional[TaskResult]:
        if handle.tenant_id is None:
            return None

        raw = self._kv_store.get(
            tenant_id=handle.tenant_id,
            key=self._result_key(handle.task_id),
        )

        if raw is None:
            return None

        decoded = raw.decode("utf-8")

        # Expected format:
        # status|attempts|error|base64_output
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
                # Defensive fallback: treat invalid base64 as no output
                output = None

        return TaskResult(
            status=status,
            output=output,
            error_message=error_message,
            attempts=attempts,
        )