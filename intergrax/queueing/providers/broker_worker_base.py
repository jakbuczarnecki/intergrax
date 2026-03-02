# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import base64
import json
from abc import ABC, abstractmethod
from typing import Optional

from intergrax.contracts.idempotency_store import IdempotencyStore
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.queueing.contracts.task_queue import TaskStatus
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.queueing.worker.execution import execute_logical_task


class BrokerWorkerBase(ABC):
    """
    Transport-agnostic execution infrastructure for broker-backed workers.

    Responsibilities:
    - Deserialize incoming transport payload
    - Manage task lifecycle transitions (PENDING -> RUNNING -> SUCCEEDED / FAILED)
    - Delegate execution to execute_logical_task
    - Persist final state to DistributedKVStore

    Does NOT:
    - Poll transport
    - Acknowledge broker messages
    - Implement retry loop
    - Implement backoff
    """

    def __init__(
        self,
        *,
        registry: TaskExecutionRegistry,
        kv_store: DistributedKVStore,
        idempotency_store: Optional[IdempotencyStore] = None,
    ) -> None:
        self._registry: TaskExecutionRegistry = registry
        self._kv_store: DistributedKVStore = kv_store
        self._idempotency_store: Optional[IdempotencyStore] = idempotency_store

    # ------------------------------------------------------------------
    # Storage keys (aligned with BrokerBackedTaskQueueBase)
    # ------------------------------------------------------------------

    def _status_key(self, task_id: str) -> str:
        return f"task:{task_id}:status"

    def _result_key(self, task_id: str) -> str:
        return f"task:{task_id}:result"

    # ------------------------------------------------------------------
    # Public entry point (called by provider-specific worker)
    # ------------------------------------------------------------------

    def process_message(
        self,
        *,
        raw_payload: bytes,
    ) -> None:
        """
        Process single broker message.

        raw_payload must be JSON-encoded message created by TaskQueue.enqueue().
        """

        message = json.loads(raw_payload.decode("utf-8"))

        task_id: str = message["task_id"]
        tenant_id: str = message["tenant_id"]
        run_id: str = message["run_id"]
        task_name: str = message["task_name"]
        encoded_payload: str = message["payload"]
        payload_bytes: bytes = base64.b64decode(encoded_payload.encode("ascii"))
        idempotency_key: Optional[str] = message.get("idempotency_key")

        # Transition -> RUNNING
        self._kv_store.set(
            tenant_id=tenant_id,
            key=self._status_key(task_id),
            value=TaskStatus.RUNNING.value.encode("utf-8"),
        )

        try:
            result = execute_logical_task(
                registry=self._registry,
                logical_task_name=task_name,
                tenant_id=tenant_id,
                run_id=run_id,
                payload=payload_bytes,
                idempotency_key=idempotency_key,
                idempotency_store=self._idempotency_store,
            )

            encoded_output = ""
            if result.output is not None:
                # Serialize BaseModel → JSON → UTF-8 bytes → base64
                json_bytes = result.output.model_dump_json().encode("utf-8")
                encoded_output = base64.b64encode(json_bytes).decode("ascii")

            result_record = f"{TaskStatus.SUCCEEDED.value}|1||{encoded_output}"

            self._kv_store.set(
                tenant_id=tenant_id,
                key=self._status_key(task_id),
                value=TaskStatus.SUCCEEDED.value.encode("utf-8"),
            )

            self._kv_store.set(
                tenant_id=tenant_id,
                key=self._result_key(task_id),
                value=result_record.encode("utf-8"),
            )

        except Exception as exc:
            # Persist FAILED
            error_message = str(exc)
            result_record = f"{TaskStatus.FAILED.value}|1|{error_message}|"

            self._kv_store.set(
                tenant_id=tenant_id,
                key=self._status_key(task_id),
                value=TaskStatus.FAILED.value.encode("utf-8"),
            )

            self._kv_store.set(
                tenant_id=tenant_id,
                key=self._result_key(task_id),
                value=result_record.encode("utf-8"),
            )

            raise exc

    # ------------------------------------------------------------------
    # Transport hook
    # ------------------------------------------------------------------

    @abstractmethod
    def start(self) -> None:
        """
        Start transport consumption loop.

        Must be implemented by provider-specific worker.
        """
        ...