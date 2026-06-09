# © Artur Czarnecki. All rights reserved.

"""Async task index protocol (IDEAL-3.4 / IDEAL-28.2)."""

from __future__ import annotations

from typing import Protocol

from intergrax.applications._shared.async_task_dispatch import AsyncTaskHandle
from intergrax.runtime.task.task import Task, TaskResult
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner


class AsyncTaskIndexProtocol(Protocol):
    async def enqueue(self, runner: UnifiedTaskRunner, task: Task) -> AsyncTaskHandle:
        ...

    def get(self, task_id: str) -> AsyncTaskHandle | None:
        ...
