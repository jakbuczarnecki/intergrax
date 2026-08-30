# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import asyncio

import pytest

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from testing_support.runtime_events import runtime_event_test_identity

pytestmark = pytest.mark.gate


def _sample_event(**updates) -> RuntimeEvent:
    base = RuntimeEvent(
        tenant_id="t1",
        event_type=RuntimeEventType.TASK_COMPLETED,
        phase=ExecutionPhase.COMPLETION,
        payload={},
        **runtime_event_test_identity(),
    )
    return base.model_copy(update=updates)


@pytest.mark.unit
def test_record_runs_async_handler_without_coroutine_warning() -> None:
    bus = RuntimeEventBus(record_history=False)
    seen: list[str] = []

    async def _async_handler(_event: RuntimeEvent) -> None:
        seen.append("handled")

    bus.subscribe(_async_handler, event_types={RuntimeEventType.TASK_COMPLETED})

    bus.record(_sample_event())

    assert seen == ["handled"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_record_schedules_async_handler_when_loop_running() -> None:
    bus = RuntimeEventBus(record_history=False)
    done: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    async def _async_handler(_event: RuntimeEvent) -> None:
        if not done.done():
            done.set_result(None)

    bus.subscribe(_async_handler, event_types={RuntimeEventType.TASK_COMPLETED})

    bus.record(_sample_event())

    await asyncio.wait_for(done, timeout=1.0)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_publish_notifies_subscribers_once() -> None:
    bus = RuntimeEventBus(record_history=False)
    seen: list[str] = []

    async def _async_handler(_event: RuntimeEvent) -> None:
        seen.append("handled")

    bus.subscribe(_async_handler, event_types={RuntimeEventType.TASK_COMPLETED})

    await bus.publish(_sample_event())

    assert seen == ["handled"]
