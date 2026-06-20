# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.hooks.hook_context import HookContext
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.middleware.trace_middleware import TraceEmittingMiddleware

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.mark.asyncio
async def test_trace_middleware_does_not_emit_step_completed_on_after_step() -> None:
    bus = RuntimeEventBus()
    middleware = TraceEmittingMiddleware(bus)
    ctx = HookContext(
        task_id="task-1",
        run_id="run-1",
        agent_id="echo",
        step_id="s1",
        phase=ExecutionPhase.STEP_EXECUTION,
    )
    await middleware.after(HookPoint.AFTER_STEP, ctx)
    assert not any(event.event_type == RuntimeEventType.STEP_COMPLETED for event in bus.history)
