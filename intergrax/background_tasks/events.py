# © Artur Czarnecki. All rights reserved.

"""Task lifecycle events for background task observability."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class TaskEventName(str, Enum):
    ENQUEUE_REQUESTED = "task.enqueue_requested"
    ENQUEUED = "task.enqueued"
    DISPATCHED = "task.dispatched"
    STARTED = "task.started"
    SUCCEEDED = "task.succeeded"
    FAILED = "task.failed"
    RESULT_STORED = "task.result_stored"
    ACKNOWLEDGED = "task.acknowledged"


@dataclass(frozen=True, slots=True)
class TaskEvent:
    """Structured lifecycle fact emitted during task execution."""

    name: TaskEventName
    task_id: str
    tenant_id: str
    run_id: str
    task_name: str
    provider: str
    correlation_id: str | None = None
    idempotency_key: str | None = None
    status: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        record = {
            "event": self.name.value,
            "intergrax.task_name": self.task_name,
            "intergrax.task_id": self.task_id,
            "intergrax.run_id": self.run_id,
            "intergrax.message_bus.provider": self.provider,
            "intergrax.worker_runtime.received": self.name
            in {TaskEventName.DISPATCHED, TaskEventName.STARTED},
            "intergrax.handler.id": self.task_name,
        }
        if self.correlation_id:
            record["intergrax.correlation_id"] = self.correlation_id
        if self.idempotency_key:
            record["intergrax.idempotency_key"] = self.idempotency_key
        if self.status:
            record["intergrax.task.status"] = self.status
        if self.metadata:
            record.update(self.metadata)
        return record


class TaskEventEmitter(Protocol):
    def emit(self, event: TaskEvent) -> None:
        ...
