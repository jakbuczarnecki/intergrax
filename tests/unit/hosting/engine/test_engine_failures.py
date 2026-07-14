# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import asyncio

import pytest

from intergrax.hosting import (
    HostedApplicationEngine,
    HostedApplicationFailurePhase,
    HostedApplicationLifecycleState,
    resolve_hosted_application_definition,
)
from intergrax.hosting.errors import HostedApplicationStartupError
from tests.unit.hosting.engine._fakes import (
    FakeInstanceGuard,
    FakeRuntime,
    FakeShutdownCoordinator,
    FixedClock,
    NoopLogger,
    RecordingPublisher,
    build_engine_paths,
    build_process_identity,
    minimal_profile_with_runtime,
)

pytestmark = pytest.mark.unit


def _build_engine(
    runtime: FakeRuntime | None = None,
    *,
    guard: FakeInstanceGuard | None = None,
) -> HostedApplicationEngine:
    clock = FixedClock()
    profile = minimal_profile_with_runtime(runtime)
    definition = resolve_hosted_application_definition(profile)
    return HostedApplicationEngine(
        definition=definition,
        instance_id="instance-001",
        paths=build_engine_paths(),
        process_identity=build_process_identity(clock),
        clock=clock,
        logger=NoopLogger(),
        shutdown=FakeShutdownCoordinator(),
        event_publisher=RecordingPublisher(),
        instance_guard=guard or FakeInstanceGuard(),
        health_poll_interval_seconds=0.01,
    )


@pytest.mark.asyncio
async def test_instance_acquire_failure_leaves_created_reusable() -> None:
    guard = FakeInstanceGuard(fail_acquire=True)
    engine = _build_engine(guard=guard)
    with pytest.raises(HostedApplicationStartupError):
        await engine.start()
    assert engine.lifecycle_snapshot().state is HostedApplicationLifecycleState.CREATED
    diagnostics = engine.diagnostics_snapshot()
    assert diagnostics.current_failure is not None
    assert diagnostics.current_failure.phase is HostedApplicationFailurePhase.INSTANCE_ACQUIRE
    guard.fail_acquire = False
    await engine.start()
    assert engine.lifecycle_snapshot().state is HostedApplicationLifecycleState.READY


@pytest.mark.asyncio
async def test_start_from_ready_rejected() -> None:
    engine = _build_engine()
    await engine.start()
    with pytest.raises(HostedApplicationStartupError):
        await engine.start()


@pytest.mark.asyncio
async def test_start_from_terminal_rejected() -> None:
    runtime = FakeRuntime(fail_start=True)
    engine = _build_engine(runtime)
    with pytest.raises(HostedApplicationStartupError):
        await engine.start()
    with pytest.raises(HostedApplicationStartupError):
        await engine.start()


@pytest.mark.asyncio
async def test_concurrent_stop_during_startup_is_deterministic() -> None:
    runtime = FakeRuntime()
    engine = _build_engine(runtime)
    shutdown: FakeShutdownCoordinator = engine.shutdown  # type: ignore[assignment]

    async def stop_soon() -> None:
        await asyncio.sleep(0.02)
        shutdown.request_shutdown("concurrent.stop")
        await engine.stop(reason_code="concurrent.stop")

    stop_task = asyncio.create_task(stop_soon())
    await engine.start()
    await stop_task
    state = engine.lifecycle_snapshot().state
    assert state in {
        HostedApplicationLifecycleState.STOPPED,
        HostedApplicationLifecycleState.STOPPING,
        HostedApplicationLifecycleState.READY,
    }


@pytest.mark.asyncio
async def test_primary_failure_preserved_through_rollback_failures() -> None:
    runtime = FakeRuntime(fail_start=True)

    async def failing_stop(_context) -> None:
        raise RuntimeError("rollback stop failed")

    runtime.stop = failing_stop  # type: ignore[method-assign]
    engine = _build_engine(runtime)
    with pytest.raises(HostedApplicationStartupError):
        await engine.start()
    diagnostics = engine.diagnostics_snapshot()
    assert diagnostics.current_failure is not None
    assert diagnostics.current_failure.phase is HostedApplicationFailurePhase.RUNTIME_START
    assert diagnostics.secondary_failures
