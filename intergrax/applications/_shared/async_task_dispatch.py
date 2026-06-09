# © Artur Czarnecki. All rights reserved.

"""Deferred Nexus task dispatch (ORCH-6.1)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from intergrax.runtime.task.task import Task, TaskResult
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner


@dataclass(frozen=True, slots=True)
class AsyncTaskHandle:
    task_id: str
    status: str
    result: TaskResult | None = None
    error: str | None = None


class InMemoryAsyncTaskIndex:
    """Lightweight async handle store for single-process hosts without a message bus."""

    def __init__(self) -> None:
        self._handles: dict[str, AsyncTaskHandle] = {}
        self._tasks: dict[str, asyncio.Task[TaskResult]] = {}

    def get(self, task_id: str) -> AsyncTaskHandle | None:
        return self._handles.get(task_id)

    async def enqueue(self, runner: UnifiedTaskRunner, task: Task) -> AsyncTaskHandle:
        task_id = task.task_id
        self._handles[task_id] = AsyncTaskHandle(task_id=task_id, status="pending")

        async def _run() -> TaskResult:
            self._handles[task_id] = AsyncTaskHandle(task_id=task_id, status="running")
            try:
                result = await runner.run_task(task)
            except Exception as exc:
                self._handles[task_id] = AsyncTaskHandle(
                    task_id=task_id,
                    status="failed",
                    error=f"{exc.__class__.__name__}: {exc}",
                )
                raise
            self._handles[task_id] = AsyncTaskHandle(
                task_id=task_id,
                status=result.state.value,
                result=result,
            )
            return result

        self._tasks[task_id] = asyncio.create_task(_run())
        return self._handles[task_id]

    def clear_for_tests(self) -> None:
        self._handles.clear()
        self._tasks.clear()


_DEFAULT_INDEX = InMemoryAsyncTaskIndex()


async def run_async(
    runner: UnifiedTaskRunner,
    task: Task,
    *,
    index: InMemoryAsyncTaskIndex | None = None,
) -> dict[str, Any]:
    """Enqueue a Nexus task without blocking the caller."""
    store = index or _DEFAULT_INDEX
    handle = await store.enqueue(runner, task)
    return {
        "task_id": handle.task_id,
        "status": handle.status,
        "async": True,
    }


async def get_async_status(
    task_id: str,
    *,
    index: InMemoryAsyncTaskIndex | None = None,
) -> dict[str, Any]:
    store = index or _DEFAULT_INDEX
    handle = store.get(task_id)
    if handle is None:
        return {"task_id": task_id, "status": "not_found"}
    payload: dict[str, Any] = {
        "task_id": handle.task_id,
        "status": handle.status,
    }
    if handle.error:
        payload["error"] = handle.error
    if handle.result is not None:
        payload["state"] = handle.result.state.value
        payload["answer"] = handle.result.answer
    return payload
