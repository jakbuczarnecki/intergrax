# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from intergrax.hosting.control import (
    HostedApplicationControlCoordinator,
    HostedApplicationControlIntent,
)
from intergrax.hosting.errors import HostedApplicationControlError
from tests.unit.hosting.engine._fakes import FixedClock

pytestmark = pytest.mark.unit


def _coordinator() -> HostedApplicationControlCoordinator:
    return HostedApplicationControlCoordinator(clock=FixedClock())


def test_duplicate_shutdown_idempotent() -> None:
    control = _coordinator()
    first = control.request_shutdown("stop.one")
    second = control.request_shutdown("stop.two")
    assert first.reason_code == second.reason_code == "stop.one"


def test_stop_overrides_restart() -> None:
    control = _coordinator()
    control.request_restart("restart.one")
    control.request_shutdown("stop.one")
    assert control.is_shutdown_requested()
    assert control.snapshot().effective_intent is HostedApplicationControlIntent.STOP


def test_restart_cannot_override_stop() -> None:
    control = _coordinator()
    control.request_shutdown("stop.one")
    with pytest.raises(HostedApplicationControlError):
        control.request_restart("restart.one")


def test_earlier_deadline_wins() -> None:
    control = _coordinator()
    clock = control.clock
    now = clock.now()
    control.request_shutdown("stop.one", deadline_at=now + timedelta(seconds=30))
    control.request_shutdown("stop.two", deadline_at=now + timedelta(seconds=10))
    current = control.current_request()
    assert current is not None
    assert current.reason_code == "stop.two"


def test_prepare_next_instance_clears_only_restart() -> None:
    control = _coordinator()
    control.request_restart("restart.one")
    control.prepare_next_instance()
    assert control.current_restart_request() is None
    control.request_shutdown("stop.one")
    control.prepare_next_instance()
    assert control.is_shutdown_requested()


@pytest.mark.asyncio
async def test_restart_wakes_shutdown_wait() -> None:
    control = _coordinator()

    async def _waiter() -> str:
        request = await control.wait_until_requested()
        return request.reason_code

    task = asyncio.create_task(_waiter())
    await asyncio.sleep(0)
    control.request_restart("restart.one")
    reason = await asyncio.wait_for(task, timeout=1.0)
    assert reason == "restart.one"


def test_health_probe_does_not_mutate_state() -> None:
    control = _coordinator()
    control.request_restart("restart.one")
    before = control.snapshot()
    after = control.health_probe()
    assert before == after
