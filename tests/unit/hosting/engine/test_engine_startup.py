# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.hosting import HostedApplicationEngine, HostedApplicationLifecycleState, resolve_hosted_application_definition
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


def _build_engine(runtime: FakeRuntime | None = None) -> HostedApplicationEngine:
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
        instance_guard=FakeInstanceGuard(),
        health_poll_interval_seconds=0.01,
    )


@pytest.mark.asyncio
async def test_successful_startup_and_shutdown() -> None:
    engine = _build_engine()
    await engine.start()
    assert engine.lifecycle_snapshot().state is HostedApplicationLifecycleState.READY
    assert engine.accepts_new_work
    result = await engine.stop(reason_code="test.complete")
    assert result.terminal_state is HostedApplicationLifecycleState.STOPPED


@pytest.mark.asyncio
async def test_run_until_stopped_waits_for_shutdown_request() -> None:
    engine = _build_engine()
    shutdown: FakeShutdownCoordinator = engine.shutdown  # type: ignore[assignment]

    async def request_later() -> None:
        await __import__("asyncio").sleep(0.05)
        shutdown.request_shutdown("test.complete")

    task = __import__("asyncio").create_task(request_later())
    result = await engine.run_until_stopped()
    await task
    assert result.terminal_state is HostedApplicationLifecycleState.STOPPED


@pytest.mark.asyncio
async def test_startup_failure_reaches_failed_and_releases_lease() -> None:
    runtime = FakeRuntime(fail_start=True)
    guard = FakeInstanceGuard()
    engine = _build_engine(runtime)
    engine.instance_guard = guard
    with pytest.raises(HostedApplicationStartupError):
        await engine.start()
    assert engine.lifecycle_snapshot().state is HostedApplicationLifecycleState.FAILED
    assert guard.lease.released
