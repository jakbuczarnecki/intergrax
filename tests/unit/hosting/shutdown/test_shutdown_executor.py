# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import asyncio

import pytest

from intergrax.hosting import ShutdownPolicy
from intergrax.hosting.shutdown import (
    HostedApplicationGlobalShutdownBudget,
    HostedApplicationShutdownExecutor,
    HostedApplicationShutdownPhase,
    ShutdownPhaseRecorder,
    build_shutdown_execution_snapshot,
)
from tests.unit.hosting.engine._fakes import FixedClock

pytestmark = pytest.mark.unit


class FakeMonotonic:
    def __init__(self, start: float = 0.0) -> None:
        self._value = start

    def monotonic(self) -> float:
        return self._value

    def advance(self, seconds: float) -> None:
        self._value += seconds


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


async def _execute(executor: HostedApplicationShutdownExecutor, budget_seconds: float = 2.0):
    clock = executor.clock
    monotonic = FakeMonotonic()
    recorder = ShutdownPhaseRecorder(clock=clock)
    budget = HostedApplicationGlobalShutdownBudget(
        deadline_monotonic=monotonic.monotonic() + budget_seconds,
        monotonic_clock=monotonic,
    )
    await executor.execute(request=None, budget=budget, recorder=recorder)
    return build_shutdown_execution_snapshot(
        shutdown_policy=executor.shutdown_policy,
        request=None,
        clock=clock,
        recorder=recorder,
        active_work_before=0,
        active_work_after=0,
    )


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
    snapshot = await _execute(executor)
    assert work.intake_stopped
    assert work.cancelled
    assert snapshot.timed_out
    assert snapshot.forced is True
    assert any(record.phase is HostedApplicationShutdownPhase.DRAIN for record in snapshot.phase_records)


@pytest.mark.asyncio
async def test_cancel_immediately_success_not_forced() -> None:
    work = FakeActiveWork()
    executor = HostedApplicationShutdownExecutor(
        shutdown_policy=ShutdownPolicy.cancel_immediately(),
        clock=FixedClock(),
        active_work_controller=work,
    )
    snapshot = await _execute(executor)
    assert work.cancelled
    assert snapshot.forced is False


@pytest.mark.asyncio
async def test_flush_ordering() -> None:
    flush_b = FakeFlush("b")
    flush_a = FakeFlush("a")
    executor = HostedApplicationShutdownExecutor(
        shutdown_policy=ShutdownPolicy.standard(),
        clock=FixedClock(),
        flush_services=(flush_b, flush_a),
    )
    snapshot = await _execute(executor)
    assert flush_a.called and flush_b.called
    flush_phases = [record.flush_id for record in snapshot.phase_records if record.phase.value == "flush"]
    assert flush_phases == ["a", "b"]
