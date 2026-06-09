# © Artur Czarnecki. All rights reserved.

"""Capacity evaluation scheduler (ECP-OBS.2)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from intergrax.runtime.capacity.collector import CapacitySignalCollector
from intergrax.runtime.capacity.evaluator import ScalingEvaluator
from intergrax.runtime.capacity.provisioner import ScalingProvisioner


TickFn = Callable[[], Awaitable[None]]


class CapacityScheduler:
    """Async cron driver that does not block Nexus."""

    def __init__(
        self,
        *,
        collector: CapacitySignalCollector,
        evaluator: ScalingEvaluator,
        provisioner: ScalingProvisioner,
        interval_seconds: float = 30.0,
    ) -> None:
        self._collector = collector
        self._evaluator = evaluator
        self._provisioner = provisioner
        self._interval = interval_seconds
        self._task: asyncio.Task[None] | None = None

    async def tick(self) -> None:
        signals = self._collector.collect()
        plan = self._evaluator.evaluate(signals)
        for action in plan.actions:
            self._provisioner.apply(action)

    async def _loop(self) -> None:
        while True:
            await self.tick()
            await asyncio.sleep(self._interval)

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
