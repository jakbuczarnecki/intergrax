# © Artur Czarnecki. All rights reserved.

"""Product active-work controller for hosted shutdown over background workers (EX-D7-01, EX-D4-01).

``BackgroundActiveWorkController`` satisfies the hosting port ``HostedApplicationActiveWorkController``
(``intergrax.hosting.shutdown``) structurally; this module does not import ``intergrax.hosting``. Its
sources are workers that expose an intake gate and an in-flight count: ``WorkerRuntime`` and the
document-store ``DocumentStoreTaskWorker`` both do.

Cancellation is cooperative only. A synchronous handler thread cannot be preempted, so
``cancel_active_work`` keeps intake closed and waits ``cancel_grace_seconds``; when work is still in
flight afterwards ``active_work_count()`` stays above zero and the shutdown executor records the
shutdown as forced. Nothing here claims to stop a running handler.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

__all__ = ["ActiveWorkSource", "BackgroundActiveWorkController"]

_IDLE_POLL_SECONDS = 0.05


@runtime_checkable
class ActiveWorkSource(Protocol):
    """A worker whose intake can be closed and whose in-flight work can be counted and awaited."""

    def stop_intake(self) -> None: ...

    def in_flight_count(self) -> int: ...

    def wait_idle(self, timeout_seconds: float) -> bool: ...


class BackgroundActiveWorkController:
    """Drain controller over one or more ``ActiveWorkSource`` workers."""

    def __init__(self, sources: Sequence[ActiveWorkSource], *, cancel_grace_seconds: float = 1.0) -> None:
        if not math.isfinite(cancel_grace_seconds) or cancel_grace_seconds < 0:
            raise ValueError("cancel_grace_seconds must be finite and not negative")
        self._sources: tuple[ActiveWorkSource, ...] = tuple(sources)
        self._cancel_grace_seconds = cancel_grace_seconds

    async def stop_intake(self) -> None:
        for source in self._sources:
            source.stop_intake()

    def active_work_count(self) -> int:
        return sum(source.in_flight_count() for source in self._sources)

    async def wait_for_idle(self) -> None:
        """Return once every source is idle; the shutdown executor bounds the phase.

        Each source is awaited in short slices on a worker thread, so a phase timeout of the
        executor cancels this coroutine without leaving a thread blocked for long.
        """
        for source in self._sources:
            while not await asyncio.to_thread(source.wait_idle, _IDLE_POLL_SECONDS):
                continue

    async def cancel_active_work(self) -> None:
        """Cooperative: keep intake closed and give in-flight work a bounded grace period."""
        await self.stop_intake()
        deadline = asyncio.get_running_loop().time() + self._cancel_grace_seconds
        for source in self._sources:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return
            await asyncio.to_thread(source.wait_idle, remaining)
