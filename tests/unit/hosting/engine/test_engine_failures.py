# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import asyncio

import pytest

from intergrax.hosting import (
    HostedApplicationEngine,
    HostedApplicationEventType,
    HostedApplicationFailurePhase,
    HostedApplicationLifecycleState,
    resolve_hosted_application_definition,
)
from intergrax.hosting.errors import HostedApplicationStartupError
from tests.unit.hosting.engine._fakes import (
    FakeInstanceGuard,
    FakeRuntime,
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
    publisher: RecordingPublisher | None = None,
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
        shutdown=__import__(
            "tests.unit.hosting.engine._fakes",
            fromlist=["FakeShutdownCoordinator"],
        ).FakeShutdownCoordinator(),
        event_publisher=publisher or RecordingPublisher(),
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
    assert diagnostics.last_failure is not None
    guard.fail_acquire = False
    await engine.start()
    assert engine.lifecycle_snapshot().state is HostedApplicationLifecycleState.READY
    diagnostics = engine.diagnostics_snapshot()
    assert diagnostics.current_failure is None
    assert diagnostics.last_failure is not None
    assert diagnostics.last_failure.phase is HostedApplicationFailurePhase.INSTANCE_ACQUIRE
    assert engine.accepts_new_work


@pytest.mark.asyncio
async def test_acquire_retry_does_not_chain_historical_exception() -> None:
    guard = FakeInstanceGuard(fail_acquire=True)
    runtime = FakeRuntime()
    engine = _build_engine(runtime, guard=guard)
    with pytest.raises(HostedApplicationStartupError):
        await engine.start()
    guard.fail_acquire = False
    runtime.fail_start = True
    with pytest.raises(HostedApplicationStartupError) as exc_info:
        await engine.start()
    cause = exc_info.value.__cause__
    assert cause is not None
    assert "instance acquire" not in str(cause).lower()
    diagnostics = engine.diagnostics_snapshot()
    assert diagnostics.current_failure is not None
    assert diagnostics.current_failure.phase is HostedApplicationFailurePhase.RUNTIME_START


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
async def test_concurrent_stop_during_startup_finishes_stopped() -> None:
    runtime = FakeRuntime(start_delay=0.1)
    guard = FakeInstanceGuard()
    publisher = RecordingPublisher()
    before_stop_calls = {"count": 0}

    from intergrax.hosting import HostedApplicationHook, HostedApplicationHooks, HostedApplicationProfile

    base_profile = minimal_profile_with_runtime(runtime)

    async def before_stop_hook(_context) -> None:
        before_stop_calls["count"] += 1

    profile = HostedApplicationProfile(
        application_id=base_profile.application_id,
        application_factory=base_profile.application_factory,
        application_factory_id=base_profile.application_factory_id,
        hooks=HostedApplicationHooks(
            before_stop=(
                HostedApplicationHook(
                    hook_id="before_stop_once",
                    handler=before_stop_hook,
                    handler_id="tests.before_stop_once",
                ),
            ),
        ),
    )
    definition = resolve_hosted_application_definition(profile)
    engine = HostedApplicationEngine(
        definition=definition,
        instance_id="instance-001",
        paths=build_engine_paths(),
        process_identity=build_process_identity(FixedClock()),
        clock=FixedClock(),
        logger=NoopLogger(),
        shutdown=__import__(
            "tests.unit.hosting.engine._fakes",
            fromlist=["FakeShutdownCoordinator"],
        ).FakeShutdownCoordinator(),
        event_publisher=publisher,
        instance_guard=guard,
        health_poll_interval_seconds=0.01,
    )

    start_task = asyncio.create_task(engine.start())
    stop_task = asyncio.create_task(engine.stop(reason_code="concurrent.stop"))
    await asyncio.gather(start_task, stop_task, return_exceptions=True)

    assert engine.lifecycle_snapshot().state is HostedApplicationLifecycleState.STOPPED
    terminal = await engine.stop(reason_code="concurrent.stop")
    assert terminal.terminal_state is HostedApplicationLifecycleState.STOPPED
    assert before_stop_calls["count"] == 1
    assert guard.lease.released
    assert runtime.stop_count <= 1
    stopped_events = [
        event
        for event in publisher.events
        if event.event_type is HostedApplicationEventType.APPLICATION_STOPPED
    ]
    assert stopped_events
    assert all(
        event.lifecycle_state is HostedApplicationLifecycleState.STOPPED
        for event in stopped_events
    )


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
