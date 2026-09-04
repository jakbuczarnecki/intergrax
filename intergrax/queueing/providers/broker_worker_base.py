# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import base64
import json
from abc import ABC, abstractmethod
from typing import Optional

from intergrax.background_tasks.events import TaskEvent, TaskEventEmitter, TaskEventName
from intergrax.contracts.idempotency_store import IdempotencyStore
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.queueing.contracts.task_queue import TaskStatus
from intergrax.queueing.task_index import record_task_index
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.queueing.worker.execution import execute_logical_task
from intergrax.runtime.background_execution.reentry_admission import (
    BackgroundExecutionReentryDisposition,
    admit_background_execution_reentry,
)
from intergrax.runtime.background_execution.identity_persistence import (
    BackgroundExecutionIdentityPersistence,
)
from intergrax.runtime.background_execution.required_audit_evidence import (
    admit_background_execution_handler,
)
from intergrax.runtime.background_execution.transport_ref import (
    BackgroundTransportExecutionRef,
)
from intergrax.runtime.execution.attempt_lifecycle.service import AttemptLifecycleService
from intergrax.runtime.execution.execution_terminal.service import ExecutionTerminalService
from intergrax.runtime.observability.causal_evidence_persistence import (
    CausalEvidencePersistence,
)


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
        event_emitter: TaskEventEmitter | None = None,
        provider_name: str = "broker",
        identity_persistence: BackgroundExecutionIdentityPersistence,
        causal_evidence_persistence: CausalEvidencePersistence,
        attempt_lifecycle: AttemptLifecycleService,
        execution_terminal: ExecutionTerminalService,
    ) -> None:
        self._registry: TaskExecutionRegistry = registry
        self._kv_store: DistributedKVStore = kv_store
        self._idempotency_store: Optional[IdempotencyStore] = idempotency_store
        self._event_emitter = event_emitter
        self._provider_name = provider_name
        self._identity_persistence = identity_persistence
        self._causal_evidence_persistence = causal_evidence_persistence
        self._attempt_lifecycle = attempt_lifecycle
        self._execution_terminal = execution_terminal

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

    def _emit_event(
        self,
        name: TaskEventName,
        *,
        task_id: str,
        tenant_id: str,
        run_id: str,
        task_name: str,
        idempotency_key: Optional[str],
        correlation_id: Optional[str] = None,
        status: Optional[str] = None,
        metadata: Optional[dict[str, object]] = None,
    ) -> None:
        if self._event_emitter is None:
            return
        self._event_emitter.emit(
            TaskEvent(
                name=name,
                task_id=task_id,
                tenant_id=tenant_id,
                run_id=run_id,
                task_name=task_name,
                provider=self._provider_name,
                correlation_id=correlation_id,
                idempotency_key=idempotency_key,
                status=status,
                metadata=dict(metadata or {}),
            )
        )

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
        encoded_payload: str = message.get("payload") or message.get("payload_base64", "")
        payload_bytes: bytes = base64.b64decode(encoded_payload.encode("ascii"))
        idempotency_key: Optional[str] = message.get("idempotency_key")
        correlation_id: Optional[str] = message.get("correlation_id")
        provider_name = str(message.get("provider") or self._provider_name)

        self._emit_event(
            TaskEventName.DISPATCHED,
            task_id=task_id,
            tenant_id=tenant_id,
            run_id=run_id,
            task_name=task_name,
            idempotency_key=idempotency_key,
            correlation_id=correlation_id,
            metadata={"intergrax.worker_runtime.received": True},
        )

        transport_ref = BackgroundTransportExecutionRef(
            tenant_id=tenant_id,
            provider=provider_name,
            transport_task_id=task_id,
        )
        reentry = admit_background_execution_reentry(
            transport_ref=transport_ref,
            identity_persistence=self._identity_persistence,
            attempt_lifecycle=self._attempt_lifecycle,
            execution_terminal=self._execution_terminal,
        )
        execution_identity = reentry.identity
        if reentry.disposition is BackgroundExecutionReentryDisposition.TERMINAL_ALREADY_RECORDED:
            return

        # Transition -> RUNNING
        self._kv_store.set(
            tenant_id=tenant_id,
            key=self._status_key(task_id),
            value=TaskStatus.RUNNING.value.encode("utf-8"),
        )
        record_task_index(
            self._kv_store,
            tenant_id=tenant_id,
            task_id=task_id,
            task_name=task_name,
            provider=provider_name,
            status=TaskStatus.RUNNING,
        )
        self._emit_event(
            TaskEventName.STARTED,
            task_id=task_id,
            tenant_id=tenant_id,
            run_id=run_id,
            task_name=task_name,
            idempotency_key=idempotency_key,
            correlation_id=correlation_id,
            status=TaskStatus.RUNNING.value,
        )

        try:
            result = admit_background_execution_handler(
                transport_ref=transport_ref,
                execution_identity=execution_identity,
                causal_evidence_persistence=self._causal_evidence_persistence,
                handler=lambda: execute_logical_task(
                    registry=self._registry,
                    logical_task_name=task_name,
                    tenant_id=execution_identity.tenant_id,
                    run_id=str(execution_identity.run_id),
                    payload=payload_bytes,
                    idempotency_key=idempotency_key,
                    idempotency_store=self._idempotency_store,
                    execution_identity=execution_identity,
                ),
            )

            if not result.success:
                tool_error = result.error
                if tool_error is not None:
                    error_message = (
                        f"{tool_error.error_code}: {tool_error.error_message}"
                    )
                else:
                    error_message = "handler_failed"
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
                record_task_index(
                    self._kv_store,
                    tenant_id=tenant_id,
                    task_id=task_id,
                    task_name=task_name,
                    provider=provider_name,
                    status=TaskStatus.FAILED,
                )
                self._emit_event(
                    TaskEventName.FAILED,
                    task_id=task_id,
                    tenant_id=tenant_id,
                    run_id=run_id,
                    task_name=task_name,
                    idempotency_key=idempotency_key,
                    correlation_id=correlation_id,
                    status=TaskStatus.FAILED.value,
                    metadata={"error": error_message},
                )
                return

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
            record_task_index(
                self._kv_store,
                tenant_id=tenant_id,
                task_id=task_id,
                task_name=task_name,
                provider=provider_name,
                status=TaskStatus.SUCCEEDED,
            )
            self._emit_event(
                TaskEventName.SUCCEEDED,
                task_id=task_id,
                tenant_id=tenant_id,
                run_id=run_id,
                task_name=task_name,
                idempotency_key=idempotency_key,
                correlation_id=correlation_id,
                status=TaskStatus.SUCCEEDED.value,
            )
            self._emit_event(
                TaskEventName.RESULT_STORED,
                task_id=task_id,
                tenant_id=tenant_id,
                run_id=run_id,
                task_name=task_name,
                idempotency_key=idempotency_key,
                correlation_id=correlation_id,
                status=TaskStatus.SUCCEEDED.value,
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
            record_task_index(
                self._kv_store,
                tenant_id=tenant_id,
                task_id=task_id,
                task_name=task_name,
                provider=provider_name,
                status=TaskStatus.FAILED,
            )
            self._emit_event(
                TaskEventName.FAILED,
                task_id=task_id,
                tenant_id=tenant_id,
                run_id=run_id,
                task_name=task_name,
                idempotency_key=idempotency_key,
                correlation_id=correlation_id,
                status=TaskStatus.FAILED.value,
                metadata={"error": error_message},
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