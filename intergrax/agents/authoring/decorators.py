# © Artur Czarnecki. All rights reserved.

"""Step decorators for :class:`~intergrax.agents.authoring.base.IntergraxAgent`."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

from intergrax.contracts.agent_step import StepOutput
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext

StepHandler = Callable[[RuntimeExecutionContext], Awaitable[StepOutput | dict[str, object]]]
F = TypeVar("F", bound=StepHandler)


def step(step_id: str, *, trace_label: str = "") -> Callable[[F], F]:
    """Mark an async method as a UAEP step (order = definition order)."""

    def decorator(fn: F) -> F:
        fn.__intergrax_step_id__ = step_id  # type: ignore[attr-defined]
        fn.__intergrax_trace_label__ = trace_label or step_id  # type: ignore[attr-defined]
        return fn

    return decorator
