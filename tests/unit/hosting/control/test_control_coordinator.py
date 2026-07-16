# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from intergrax.hosting.control import (
    HostedApplicationControlCoordinator,
    HostedApplicationControlIntent,
)
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


def test_restart_after_stop_preserves_stop() -> None:
    control = _coordinator()
    control.request_shutdown("stop.one", source_id="signal.sigterm")
    restart = control.request_restart("restart.one")
    assert control.is_shutdown_requested()
    assert control.snapshot().effective_intent is HostedApplicationControlIntent.STOP
    assert restart.reason_code == "stop.one"
    effective = control.current_effective_request()
    assert effective is not None
    assert effective.source_id == "signal.sigterm"


def test_stop_source_id_preserved() -> None:
    control = _coordinator()
    control.request_shutdown("stop.one", source_id="operator.cli")
    effective = control.current_effective_request()
    assert effective is not None
    assert effective.source_id == "operator.cli"


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
