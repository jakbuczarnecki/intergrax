# © Artur Czarnecki. All rights reserved.

"""Compensation flow pattern for failed graph steps (IDEAL-22.3)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any


CompensationHandler = Callable[[str, dict[str, Any]], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class CompensationStep:
    step_id: str
    handler_id: str


@dataclass
class CompensationFlow:
    """Ordered compensation handlers invoked when a step fails after side effects."""

    steps: list[CompensationStep]
    handlers: dict[str, CompensationHandler]

    async def run(self, failed_step_id: str, context: dict[str, Any]) -> list[str]:
        executed: list[str] = []
        for step in reversed(self.steps):
            if step.step_id != failed_step_id:
                continue
            handler = self.handlers.get(step.handler_id)
            if handler is None:
                continue
            await handler(failed_step_id, context)
            executed.append(step.handler_id)
        return executed
