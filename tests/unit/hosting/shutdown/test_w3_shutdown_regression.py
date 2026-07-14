# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from intergrax.hosting import ShutdownPolicy
from intergrax.hosting.contracts.lifecycle import HostedApplicationEffectiveControlRequest
from intergrax.hosting.shutdown import (
    HostedApplicationGlobalShutdownBudget,
    HostedApplicationShutdownExecutor,
    HostedApplicationShutdownPhase,
    HostedApplicationShutdownPhaseOutcome,
    ShutdownPhaseRecorder,
    build_shutdown_execution_snapshot,
    compute_shutdown_budget_seconds,
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
    def __init__(self, *, slow_drain: bool = False, fail_cancel: bool = False, fail_intake: bool = False) -> None:
        self.intake_stopped = False
        self.active = 1
        self.slow_drain = slow_drain
        self.fail_cancel = fail_cancel
        self.fail_intake = fail_intake
        self.cancelled = False

    async def stop_intake(self) -> None:
        if self.fail_intake:
            raise RuntimeError("intake failed")
        self.intake_stopped = True

    def active_work_count(self) -> int:
        return self.active

    async def wait_for_idle(self) -> None:
        if self.slow_drain:
            await asyncio.sleep(1.0)
            return
        self.active = 0

    async def cancel_active_work(self) -> None:
        if self.fail_cancel:
            raise RuntimeError("cancel failed")
        self.cancelled = True
        self.active = 0


class FakeFlush:
    def __init__(self, flush_id: str, *, fail: bool = False) -> None:
        self.flush_id = flush_id
        self.fail = fail

    async def flush(self) -> None:
        if self.fail:
            raise RuntimeError("flush failed")


async def _run(executor: HostedApplicationShutdownExecutor, budget_seconds: float = 0.05):
    monotonic = FakeMonotonic()
    recorder = ShutdownPhaseRecorder(clock=executor.clock)
    budget = HostedApplicationGlobalShutdownBudget(
        deadline_monotonic=monotonic.monotonic() + budget_seconds,
        monotonic_clock=monotonic,
    )
    request = HostedApplicationEffectiveControlRequest(
        intent="restart",
        reason_code="restart.test",
        requested_at=executor.clock.now(),
        deadline_at=executor.clock.now() + timedelta(seconds=budget_seconds),
    )
    await executor.execute(request=request, budget=budget, recorder=recorder)
    return build_shutdown_execution_snapshot(
        shutdown_policy=executor.shutdown_policy,
        request=request,
        clock=executor.clock,
        recorder=recorder,
        active_work_before=1,
        active_work_after=executor.active_work_controller.active_work_count()
        if executor.active_work_controller
        else 0,
    )


def test_restart_deadline_propagation_budget() -> None:
    clock = FixedClock()
    now = clock.now()
    budget = compute_shutdown_budget_seconds(
        shutdown_policy=ShutdownPolicy.standard(),
        blocking_hook_timeout=30.0,
        observer_drain_timeout=5.0,
        component_stop_budget=10.0,
        runtime_stop_budget=10.0,
        lease_release_timeout=2.0,
        explicit_deadline_at=now + timedelta(seconds=3),
        clock=clock,
        requested_at=now,
    )
    assert budget == pytest.approx(3.0, abs=0.01)


@pytest.mark.asyncio
async def test_one_global_budget_across_phases() -> None:
    work = FakeActiveWork(slow_drain=True)
    executor = HostedApplicationShutdownExecutor(
        shutdown_policy=ShutdownPolicy.drain_then_cancel(drain_timeout_seconds=5.0, cancel_timeout_seconds=5.0),
        clock=FixedClock(),
        active_work_controller=work,
    )
    snapshot = await _run(executor, budget_seconds=0.02)
    assert snapshot.timed_out or snapshot.forced
    assert any(record.phase is HostedApplicationShutdownPhase.STOP_INTAKE for record in snapshot.phase_records)


@pytest.mark.asyncio
async def test_stop_intake_failure_recorded() -> None:
    work = FakeActiveWork(fail_intake=True)
    executor = HostedApplicationShutdownExecutor(
        shutdown_policy=ShutdownPolicy.standard(),
        clock=FixedClock(),
        active_work_controller=work,
    )
    snapshot = await _run(executor)
    intake = next(r for r in snapshot.phase_records if r.phase is HostedApplicationShutdownPhase.STOP_INTAKE)
    assert intake.outcome is HostedApplicationShutdownPhaseOutcome.FAILED


@pytest.mark.asyncio
async def test_drain_timeout_then_cancel() -> None:
    work = FakeActiveWork(slow_drain=True)
    executor = HostedApplicationShutdownExecutor(
        shutdown_policy=ShutdownPolicy.drain_then_cancel(drain_timeout_seconds=5.0, cancel_timeout_seconds=5.0),
        clock=FixedClock(),
        active_work_controller=work,
    )
    snapshot = await _run(executor, budget_seconds=0.05)
    phases = [record.phase for record in snapshot.phase_records]
    assert HostedApplicationShutdownPhase.DRAIN in phases
    assert HostedApplicationShutdownPhase.CANCEL in phases


@pytest.mark.asyncio
async def test_cancel_failure_recorded() -> None:
    work = FakeActiveWork(fail_cancel=True)
    executor = HostedApplicationShutdownExecutor(
        shutdown_policy=ShutdownPolicy.cancel_immediately(),
        clock=FixedClock(),
        active_work_controller=work,
    )
    snapshot = await _run(executor)
    cancel = next(r for r in snapshot.phase_records if r.phase is HostedApplicationShutdownPhase.CANCEL)
    assert cancel.outcome is HostedApplicationShutdownPhaseOutcome.FAILED


@pytest.mark.asyncio
async def test_flush_failure_isolated() -> None:
    executor = HostedApplicationShutdownExecutor(
        shutdown_policy=ShutdownPolicy.standard(),
        clock=FixedClock(),
        flush_services=(FakeFlush("a", fail=True),),
    )
    snapshot = await _run(executor, budget_seconds=1.0)
    flush = next(r for r in snapshot.phase_records if r.phase is HostedApplicationShutdownPhase.FLUSH)
    assert flush.outcome is HostedApplicationShutdownPhaseOutcome.FAILED


@pytest.mark.asyncio
async def test_snapshot_always_returned() -> None:
    executor = HostedApplicationShutdownExecutor(
        shutdown_policy=ShutdownPolicy.standard(),
        clock=FixedClock(),
        active_work_controller=FakeActiveWork(),
    )
    snapshot = await _run(executor, budget_seconds=1.0)
    assert snapshot.completed_at is not None
    assert len(snapshot.phase_records) >= 1
