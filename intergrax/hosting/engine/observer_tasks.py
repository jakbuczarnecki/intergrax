# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tracked observer task registry for hosting coordinators."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable

from intergrax.hosting.engine.diagnostics import DiagnosticsRecorder, HostedApplicationFailurePhase


class ObserverTaskRegistry:
    """Retain strong references to observer tasks and support bounded drain."""

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
        coro: Awaitable[None],
        *,
        phase: HostedApplicationFailurePhase,
        source_id: str,
    ) -> asyncio.Task[None] | None:
        if not self._accepting_new_tasks:
            return None
        task = asyncio.create_task(self._run_observed(coro, phase=phase, source_id=source_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

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
        if not self._tasks:
            return
        pending = list(self._tasks)
        done, still_pending = await asyncio.wait(
            pending,
            timeout=timeout_seconds,
            return_when=asyncio.ALL_COMPLETED,
        )
        for task in still_pending:
            task.cancel()
        if still_pending:
            await asyncio.gather(*still_pending, return_exceptions=True)

    def cancel_remaining(self) -> None:
        for task in list(self._tasks):
            task.cancel()
