# © Artur Czarnecki. All rights reserved.

"""T-IGX-041-D / F-D4-01: task cancellation rolls back lease, components and runtime."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from intergrax.hosting import (
    HostedApplicationComponentRegistration,
    HostedApplicationLifecycleState,
    HostedApplicationStartupError,
    HostedApplicationSupervisor,
    HostedApplicationSupervisorLaunchContext,
)
from intergrax.hosting.control import HostedApplicationControlCoordinator
from tests.qualification.pdc_x.d4._support import (
    JournalRuntime,
    OrderedComponent,
    RecordingPublisher,
    SystemClock,
    build_engine,
    definition_of,
    make_profile,
)

pytestmark = [pytest.mark.integration]

_WAIT_SECONDS = 20.0


async def _wait_for(predicate, description: str) -> None:
    deadline = asyncio.get_running_loop().time() + _WAIT_SECONDS
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(f"timed out waiting for: {description}")
        await asyncio.sleep(0.02)


def _required(component: OrderedComponent, *dependencies: str) -> HostedApplicationComponentRegistration:
    return HostedApplicationComponentRegistration(
        component=component,
        required=True,
        dependencies=tuple(dependencies),
        start_timeout_seconds=15.0,
        stop_timeout_seconds=5.0,
    )


class _SlowStartComponent(OrderedComponent):
    def __init__(self, component_id: str, journal: list[str]) -> None:
        super().__init__(component_id, journal)
        self.entered = asyncio.Event()

    async def start(self, context) -> None:  # type: ignore[override]
        self._journal.append(f"start:{self.component_id}")
        self.entered.set()
        await asyncio.sleep(60)


class _SlowStopRuntime(JournalRuntime):
    def __init__(self, journal: list[str]) -> None:
        super().__init__(journal)
        self.stop_entered = asyncio.Event()

    async def stop(self, context) -> None:  # type: ignore[override]
        self.journal.append("stop:runtime:begin")
        self.stop_entered.set()
        await asyncio.sleep(0.3)
        self.journal.append("stop:runtime:end")
        self.started = False


async def _assert_second_engine_starts(root: Path) -> None:
    second, _, _ = build_engine(definition_of(make_profile(runtime=JournalRuntime())), root, instance_id="instance-002")
    await second.start()
    try:
        assert second.lifecycle_snapshot().state is HostedApplicationLifecycleState.READY
    finally:
        await second.stop(reason_code="test.cleanup")


async def test_cancel_during_startup_rolls_back_and_releases_the_lease(tmp_path: Path) -> None:
    journal: list[str] = []
    slow = _SlowStartComponent("slow", journal)
    profile = make_profile(
        runtime=JournalRuntime(journal),
        components=(_required(OrderedComponent("core", journal)), _required(slow, "core")),
    )
    engine, _, _ = build_engine(definition_of(profile), tmp_path)

    task = asyncio.ensure_future(engine.start())
    await asyncio.wait_for(slow.entered.wait(), _WAIT_SECONDS)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert journal[:2] == ["start:core", "start:slow"]
    assert "stop:core" in journal, "a started component must be rolled back on cancellation"
    diagnostics = engine.diagnostics_snapshot()
    assert diagnostics.instance_lease_released
    assert diagnostics.context_closed
    assert engine.lifecycle_snapshot().state is HostedApplicationLifecycleState.FAILED
    assert engine.reuse_blocked
    await _assert_second_engine_starts(tmp_path)


async def test_cancel_run_until_stopped_at_ready_stops_everything(tmp_path: Path) -> None:
    journal: list[str] = []
    profile = make_profile(
        runtime=JournalRuntime(journal),
        components=(_required(OrderedComponent("core", journal)),),
    )
    engine, _, _ = build_engine(definition_of(profile), tmp_path)
    task = asyncio.ensure_future(engine.run_until_stopped())
    await _wait_for(
        lambda: engine.lifecycle_snapshot().state is HostedApplicationLifecycleState.READY,
        "engine ready",
    )
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert journal[-2:] == ["stop:runtime", "stop:core"]
    diagnostics = engine.diagnostics_snapshot()
    assert diagnostics.instance_lease_released
    assert diagnostics.context_closed
    assert engine.lifecycle_snapshot().state is HostedApplicationLifecycleState.STOPPED
    await _assert_second_engine_starts(tmp_path)


async def test_cancel_during_stop_still_completes_the_stop_sequence(tmp_path: Path) -> None:
    journal: list[str] = []
    runtime = _SlowStopRuntime(journal)
    profile = make_profile(runtime=runtime, components=(_required(OrderedComponent("core", journal)),))
    engine, _, _ = build_engine(definition_of(profile), tmp_path)
    await engine.start()

    stop_task = asyncio.ensure_future(engine.stop(reason_code="test.cancel_stop"))
    await asyncio.wait_for(runtime.stop_entered.wait(), _WAIT_SECONDS)
    stop_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await stop_task

    assert "stop:runtime:end" in journal, "the stop sequence is driven to completion"
    assert journal[-1] == "stop:core"
    assert engine.diagnostics_snapshot().instance_lease_released
    assert engine.lifecycle_snapshot().state is HostedApplicationLifecycleState.STOPPED


async def test_cancel_supervisor_run_at_ready_leaves_a_clean_engine(tmp_path: Path) -> None:
    journal: list[str] = []
    profile = make_profile(runtime=JournalRuntime(journal))
    definition = definition_of(profile)
    clock = SystemClock()
    control = HostedApplicationControlCoordinator(clock=clock)
    publisher = RecordingPublisher()
    engines: list = []

    def engine_factory(launch: HostedApplicationSupervisorLaunchContext):
        engine, _, _ = build_engine(
            launch.definition,
            tmp_path,
            publisher=publisher,
            control=launch.control,
            instance_id=launch.instance_id,
        )
        engines.append(engine)
        return engine

    supervisor = HostedApplicationSupervisor(
        definition=definition,
        engine_factory=engine_factory,
        control=control,
        event_publisher=publisher,
        clock=clock,
        instance_id_generator=lambda: "cancel-attempt-0",
    )
    task = asyncio.ensure_future(supervisor.run())
    await _wait_for(
        lambda: bool(engines) and engines[0].lifecycle_snapshot().state is HostedApplicationLifecycleState.READY,
        "supervised engine ready",
    )
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    diagnostics = engines[0].diagnostics_snapshot()
    assert diagnostics.instance_lease_released
    assert diagnostics.context_closed
    assert "stop:runtime" in journal


async def test_ordinary_startup_failure_keeps_the_existing_rollback(tmp_path: Path) -> None:
    """Negative control: the non-cancellation path is unchanged."""
    journal: list[str] = []
    profile = make_profile(
        runtime=JournalRuntime(journal),
        components=(
            _required(OrderedComponent("alpha", journal)),
            _required(OrderedComponent("bravo", journal, fail_start=True), "alpha"),
        ),
    )
    engine, _, _ = build_engine(definition_of(profile), tmp_path)
    with pytest.raises(HostedApplicationStartupError):
        await engine.start()
    assert [item for item in journal if item.startswith("stop:")] == ["stop:alpha"]
    assert engine.diagnostics_snapshot().instance_lease_released
