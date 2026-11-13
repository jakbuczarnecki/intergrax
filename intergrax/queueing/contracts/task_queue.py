# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, StrEnum
from typing import Any

from intergrax.contracts.execution_identity import (
    validate_attempt_id,
    validate_execution_id,
    validate_run_id,
    validate_task_id,
)
from intergrax.queueing.task_priority import TaskPriority


class TaskStatus(str, Enum):  # noqa: UP042 - kept: str()/format() behaviour is relied upon
    """
    Task lifecycle status.
    """

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    # Terminal: the retry budget was exhausted (PDC-D7-R013). Providers may map this
    # to a native dead-letter queue; the status itself is provider-neutral.
    DEAD_LETTERED = "DEAD_LETTERED"


TERMINAL_TASK_STATUSES = frozenset({TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.DEAD_LETTERED})


def is_terminal_task_status(status: TaskStatus) -> bool:
    """True when no further execution of the task can occur."""
    return status in TERMINAL_TASK_STATUSES


class DeadLetterReason(StrEnum):
    """Typed reason a task reached ``TaskStatus.DEAD_LETTERED``."""

    RETRY_BUDGET_EXHAUSTED = "retry_budget_exhausted"


@dataclass(frozen=True)
class DeadLetterRecord:
    """Why and after how many attempts a task was dead-lettered."""

    reason: DeadLetterReason
    attempts: int
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason.value,
            "attempts": self.attempts,
            "last_error": self.last_error,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DeadLetterRecord:
        last_error = data.get("last_error")
        return cls(
            reason=DeadLetterReason(str(data["reason"])),
            attempts=int(data["attempts"]),
            last_error=str(last_error) if last_error else None,
        )


def _clean_optional(value: str | None, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty string without surrounding whitespace")
    return value


@dataclass(frozen=True)
class TaskExecutionIdentity:
    """
    Canonical execution identity CARRIED on the enqueue envelope (PDC-D7-R016).

    The queue never mints identity (ADR-PDC-07: no second identity authority). The
    enqueuing side supplies values minted by the canonical identity authority; they
    are validated here and handed to the worker unchanged.

    ``run_id`` here is the canonical ``RunId``, distinct from the transport
    correlation ``TaskRequest.run_id``.
    """

    task_id: str | None = None
    run_id: str | None = None
    attempt_id: str | None = None
    execution_id: str | None = None
    principal_id: str | None = None
    trace_id: str | None = None

    def __post_init__(self) -> None:
        if self.task_id is not None:
            validate_task_id(self.task_id)
        if self.run_id is not None:
            validate_run_id(self.run_id)
        if self.attempt_id is not None:
            validate_attempt_id(self.attempt_id)
        if self.execution_id is not None:
            validate_execution_id(self.execution_id)
        _clean_optional(self.principal_id, "principal_id")
        _clean_optional(self.trace_id, "trace_id")

    def has_canonical_execution_ids(self) -> bool:
        """True when the full TaskId/RunId/AttemptId/ExecutionId quadruple is carried."""
        return None not in (self.task_id, self.run_id, self.attempt_id, self.execution_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "execution_id": self.execution_id,
            "principal_id": self.principal_id,
            "trace_id": self.trace_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TaskExecutionIdentity:
        return cls(
            task_id=data.get("task_id"),
            run_id=data.get("run_id"),
            attempt_id=data.get("attempt_id"),
            execution_id=data.get("execution_id"),
            principal_id=data.get("principal_id"),
            trace_id=data.get("trace_id"),
        )


@dataclass(frozen=True)
class TaskRequest:
    """
    Immutable task submission request.

    Guarantees:
    - tenant isolation
    - run correlation
    - optional idempotency support
    - optional canonical execution identity (attempt/execution id, principal, trace)
    """

    tenant_id: str
    run_id: str
    task_name: str
    payload: bytes
    idempotency_key: str | None = None
    priority: TaskPriority = TaskPriority.NORMAL
    identity: TaskExecutionIdentity | None = None


@dataclass(frozen=True)
class TaskHandle:
    """
    Opaque task handle returned by provider.
    """

    task_id: str
    provider: str
    tenant_id: str | None = None


@dataclass(frozen=True)
class TaskSummary:
    """Lightweight task listing row for queue inspection tools."""

    task_id: str
    tenant_id: str
    task_name: str
    status: TaskStatus
    provider: str


@dataclass(frozen=True)
class TaskResult:
    """
    Final task result.

    `output` is raw bytes to keep Tier-0 backend-agnostic.
    """

    status: TaskStatus
    output: bytes | None = None
    error_message: str | None = None
    attempts: int = 0
    dead_letter: DeadLetterRecord | None = None


class TaskQueue(ABC):
    """
    Distributed task execution contract (Tier-0 capability).

    Backend-agnostic abstraction for:
    - asynchronous task submission
    - status inspection
    - result retrieval

    This contract does NOT define:
    - execution model
    - concurrency model
    - transport mechanism
    """

    @abstractmethod
    def enqueue(
        self,
        request: TaskRequest,
    ) -> TaskHandle:
        """
        Submit task for asynchronous execution.
        Must be crash-safe and idempotency-aware (backend dependent).
        """
        ...

    @abstractmethod
    def get_status(
        self,
        handle: TaskHandle,
    ) -> TaskStatus:
        """
        Retrieve current task status.
        """
        ...

    @abstractmethod
    def get_result(
        self,
        handle: TaskHandle,
    ) -> TaskResult | None:
        """
        Retrieve final task result.
        Returns None if task not completed.
        """
        ...

    def cancel(self, handle: TaskHandle) -> bool:
        """Request cancellation of a queued or running task. Default: unsupported."""
        return False

    def list_tasks(
        self,
        tenant_id: str,
        *,
        limit: int = 50,
        status_filter: TaskStatus | None = None,
    ) -> list[TaskSummary]:
        """List recent tasks for a tenant. Default: empty when backend has no index."""
        return []

    def purge_completed(
        self,
        tenant_id: str,
        *,
        older_than_seconds: int = 0,
    ) -> int:
        """Remove completed task records for a tenant. Default: unsupported."""
        return 0
