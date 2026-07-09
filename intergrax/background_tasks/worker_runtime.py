# © Artur Czarnecki. All rights reserved.

"""WorkerRuntime — minimal platform worker dispatch for broker-backed tasks."""

from __future__ import annotations

import base64
import json
from typing import Optional

from pydantic import BaseModel

from intergrax.background_tasks.events import TaskEvent, TaskEventEmitter, TaskEventName
from intergrax.background_tasks.registry import TaskRegistry, UnknownTaskError
from intergrax.background_tasks.state_store import TaskResultStore, TaskStateStore
from intergrax.contracts.idempotency_store import IdempotencyStore
from intergrax.queueing.contracts.task_queue import TaskRequest, TaskResult, TaskStatus
from intergrax.queueing.worker.execution import execute_logical_task
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.tools.execution_models import ToolExecutionResult


class WorkerRuntime:
    """
    Platform worker runtime for registered background tasks.

  Consumes TaskRequest envelopes, resolves handlers through TaskRegistry, persists
  status/result, and emits lifecycle events.
    """

    def __init__(
        self,
        *,
        registry: TaskRegistry,
        state_store: TaskStateStore,
        result_store: TaskResultStore,
        execution_registry: TaskExecutionRegistry,
        provider: str,
        idempotency_store: Optional[IdempotencyStore] = None,
        event_emitter: TaskEventEmitter | None = None,
    ) -> None:
        self._registry = registry
        self._state_store = state_store
        self._result_store = result_store
        self._execution_registry = execution_registry
        self._provider = provider
        self._idempotency_store = idempotency_store
        self._event_emitter = event_emitter

    def _emit(
        self,
        name: TaskEventName,
        *,
        request: TaskRequest,
        task_id: str,
        status: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        if self._event_emitter is None:
            return
        correlation_id = None
        if request.payload:
            try:
                raw = json.loads(request.payload.decode("utf-8"))
                if isinstance(raw, dict) and raw.get("correlation_id"):
                    correlation_id = str(raw["correlation_id"])
            except Exception:
                correlation_id = None
        self._event_emitter.emit(
            TaskEvent(
                name=name,
                task_id=task_id,
                tenant_id=request.tenant_id,
                run_id=request.run_id,
                task_name=request.task_name,
                provider=self._provider,
                correlation_id=correlation_id,
                idempotency_key=request.idempotency_key,
                status=status,
                metadata=dict(metadata or {}),
            )
        )

    def process_request(self, request: TaskRequest, *, task_id: str) -> TaskResult:
        if not self._registry.has_task(request.task_name):
            raise UnknownTaskError(f"unknown_task_name:{request.task_name}")

        self._emit(
            TaskEventName.DISPATCHED,
            request=request,
            task_id=task_id,
            metadata={"intergrax.worker_runtime.received": True},
        )
        self._state_store.set_status(
            tenant_id=request.tenant_id,
            task_id=task_id,
            task_name=request.task_name,
            status=TaskStatus.RUNNING,
        )
        self._emit(
            TaskEventName.STARTED,
            request=request,
            task_id=task_id,
            status=TaskStatus.RUNNING.value,
        )

        try:
            tool_result: ToolExecutionResult[BaseModel] = execute_logical_task(
                registry=self._execution_registry,
                logical_task_name=request.task_name,
                tenant_id=request.tenant_id,
                run_id=request.run_id,
                payload=request.payload,
                idempotency_key=request.idempotency_key,
                idempotency_store=self._idempotency_store,
            )
        except Exception as exc:
            self._state_store.set_status(
                tenant_id=request.tenant_id,
                task_id=task_id,
                task_name=request.task_name,
                status=TaskStatus.FAILED,
            )
            self._result_store.mark_failed(
                tenant_id=request.tenant_id,
                task_id=task_id,
                task_name=request.task_name,
                provider=self._provider,
                error_message=str(exc),
            )
            self._emit(
                TaskEventName.FAILED,
                request=request,
                task_id=task_id,
                status=TaskStatus.FAILED.value,
                metadata={"error": str(exc)},
            )
            raise

        if not tool_result.success:
            error_message = tool_result.error.error_message if tool_result.error else "handler_failed"
            self._state_store.set_status(
                tenant_id=request.tenant_id,
                task_id=task_id,
                task_name=request.task_name,
                status=TaskStatus.FAILED,
            )
            result = TaskResult(
                status=TaskStatus.FAILED,
                error_message=error_message,
                attempts=1,
            )
            self._result_store.store_result(
                tenant_id=request.tenant_id,
                task_id=task_id,
                result=result,
            )
            self._emit(
                TaskEventName.FAILED,
                request=request,
                task_id=task_id,
                status=TaskStatus.FAILED.value,
                metadata={"error": error_message},
            )
            return result

        encoded_output = ""
        if tool_result.output is not None:
            json_bytes = tool_result.output.model_dump_json().encode("utf-8")
            encoded_output = base64.b64encode(json_bytes).decode("ascii")

        result = TaskResult(status=TaskStatus.SUCCEEDED, attempts=1)
        self._state_store.set_status(
            tenant_id=request.tenant_id,
            task_id=task_id,
            task_name=request.task_name,
            status=TaskStatus.SUCCEEDED,
        )
        self._result_store.store_result(
            tenant_id=request.tenant_id,
            task_id=task_id,
            result=result,
            encoded_output=encoded_output,
        )
        self._emit(
            TaskEventName.SUCCEEDED,
            request=request,
            task_id=task_id,
            status=TaskStatus.SUCCEEDED.value,
        )
        self._emit(
            TaskEventName.RESULT_STORED,
            request=request,
            task_id=task_id,
            status=TaskStatus.SUCCEEDED.value,
        )
        return result
