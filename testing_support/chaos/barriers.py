# © Artur Czarnecki. All rights reserved.

"""Deterministic asyncio synchronization for chaos scenarios."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field


@dataclass
class PhaseGate:
    """Block workers until ``release()``; no sleeps."""

    _started: asyncio.Event = field(default_factory=asyncio.Event)
    _release: asyncio.Event = field(default_factory=asyncio.Event)
    labels: list[str] = field(default_factory=list)

    def mark_started(self, label: str) -> None:
        self.labels.append(label)
        self._started.set()

    async def wait_until_started(
        self, expected: frozenset[str], *, polls: int = 200
    ) -> None:
        for _ in range(polls):
            if frozenset(self.labels) == expected:
                return
            await asyncio.sleep(0)
        raise AssertionError(f"expected start labels {expected}, got {self.labels}")

    async def block(self) -> None:
        await self._release.wait()

    def release(self) -> None:
        self._release.set()
