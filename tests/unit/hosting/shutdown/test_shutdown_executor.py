# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import asyncio

import pytest

from intergrax.hosting import ShutdownPolicy
from intergrax.hosting.shutdown import HostedApplicationShutdownExecutor, HostedApplicationShutdownPhase
from tests.unit.hosting.engine._fakes import FixedClock

pytestmark = pytest.mark.unit


class FakeActiveWork:
    def __init__(self) -> None:
        self.intake_stopped = False
        self.active = 2
        self.cancelled = False

    async def stop_intake(self) -> None:
        self.intake_stopped = True

    def active_work_count(self) -> int:
        return self.active

    async def wait_for_idle(self) -> None:
        self.active = 0

    async def cancel_active_work(self) -> None:
        self.cancelled = True
        self.active = 0


class SlowActiveWork(FakeActiveWork):
    async def wait_for_idle(self) -> None:
        await asyncio.sleep(1.0)


class FakeFlush:
    def __init__(self, flush_id: str, *, fail: bool = False) -> None:
        self.flush_id = flush_id
        self.called = False
        self.fail = fail

    async def flush(self) -> None:
        self.called = True
        if self.fail:
            raise RuntimeError("flush failed")


@pytest.mark.asyncio
async def test_drain_then_cancel_strategy() -> None:
    work = SlowActiveWork()
    executor = HostedApplicationShutdownExecutor(
        shutdown_policy=ShutdownPolicy.drain_then_cancel(
            drain_timeout_seconds=0.01,
            cancel_timeout_seconds=0.5,
        ),
        clock=FixedClock(),
        active_work_controller=work,
    )
    snapshot = await executor.execute(request=None, budget_seconds=2.0)
    assert work.intake_stopped
    assert work.cancelled
    assert snapshot.forced
    assert any(record.phase is HostedApplicationShutdownPhase.DRAIN for record in snapshot.phase_records)


@pytest.mark.asyncio
async def test_cancel_immediately() -> None:
    work = FakeActiveWork()
    executor = HostedApplicationShutdownExecutor(
        shutdown_policy=ShutdownPolicy.cancel_immediately(),
        clock=FixedClock(),
        active_work_controller=work,
    )
    snapshot = await executor.execute(request=None, budget_seconds=2.0)
    assert work.cancelled
    assert snapshot.strategy.value == "cancel_immediately"


@pytest.mark.asyncio
async def test_flush_ordering() -> None:
    flush_b = FakeFlush("b")
    flush_a = FakeFlush("a")
    executor = HostedApplicationShutdownExecutor(
        shutdown_policy=ShutdownPolicy.standard(),
        clock=FixedClock(),
        flush_services=(flush_b, flush_a),
    )
    await executor.execute(request=None, budget_seconds=2.0)
    assert flush_a.called and flush_b.called
    flush_phases = [record.flush_id for record in executor.phase_records if record.phase.value == "flush"]
    assert flush_phases == ["a", "b"]
