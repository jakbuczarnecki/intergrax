# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class TaskStatus(str, Enum):
    """
    Task lifecycle status.
    """

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class TaskRequest:
    """
    Immutable task submission request.

    Guarantees:
    - tenant isolation
    - run correlation
    - optional idempotency support
    """

    tenant_id: str
    run_id: str
    task_name: str
    payload: bytes
    idempotency_key: Optional[str] = None


@dataclass(frozen=True)
class TaskHandle:
    """
    Opaque task handle returned by provider.
    """

    task_id: str
    provider: str
    tenant_id: Optional[str] = None


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
    output: Optional[bytes] = None
    error_message: Optional[str] = None
    attempts: int = 0


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
    ) -> Optional[TaskResult]:
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
        status_filter: Optional[TaskStatus] = None,
    ) -> List[TaskSummary]:
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