# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tracked observer task registry for hosting coordinators."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from intergrax.hosting.engine.diagnostics import DiagnosticsRecorder, HostedApplicationFailurePhase

CoroutineFactory = Callable[[], Awaitable[None]]


class ObserverTaskRegistry:
    """Retain strong references to observer tasks and support bounded quiescent drain."""

    def __init__(self, diagnostics: DiagnosticsRecorder) -> None:
        self._tasks: set[asyncio.Task[None]] = set()
        self._diagnostics = diagnostics
        self._accepting_new_tasks = True

    @property
    def task_count(self) -> int:
        return len(self._tasks)

    @property
    def accepting_new_tasks(self) -> bool:
        return self._accepting_new_tasks

    def close_to_new_tasks(self) -> None:
        self._accepting_new_tasks = False

    def schedule(
        self,
        coro_factory: CoroutineFactory,
        *,
        phase: HostedApplicationFailurePhase,
        source_id: str,
    ) -> asyncio.Task[None] | None:
        if not self._accepting_new_tasks:
            self._close_rejected_coroutine(coro_factory)
            return None
        task = asyncio.create_task(
            self._run_observed(coro_factory(), phase=phase, source_id=source_id),
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    @staticmethod
    def _close_rejected_coroutine(coro_factory: CoroutineFactory) -> None:
        coro = coro_factory()
        if asyncio.iscoroutine(coro):
            coro.close()

    async def _run_observed(
        self,
        coro: Awaitable[None],
        *,
        phase: HostedApplicationFailurePhase,
        source_id: str,
    ) -> None:
        try:
            await coro
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._diagnostics.record_secondary_failure(
                phase=phase,
                source_kind="observer_task",
                source_id=source_id,
                exc=exc,
            )

    async def drain(self, timeout_seconds: float) -> None:
        """Drain observer tasks until quiescent or timeout."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds
        while self._tasks:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            pending = list(self._tasks)
            if not pending:
                break
            _done, still_pending = await asyncio.wait(
                pending,
                timeout=remaining,
                return_when=asyncio.ALL_COMPLETED,
            )
            if not self._tasks:
                return
            if still_pending and len(still_pending) == len(pending):
                continue
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    def cancel_remaining(self) -> None:
        for task in list(self._tasks):
            task.cancel()
