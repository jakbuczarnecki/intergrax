# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.hosting import HostedApplicationEngine, HostedApplicationLifecycleState
from intergrax.hosting.errors import HostedApplicationShutdownError
from tests.unit.hosting.engine._fakes import (
    FakeRuntime,
    FakeShutdownCoordinator,
    FixedClock,
    NoopLogger,
    RecordingPublisher,
    build_engine_paths,
    build_process_identity,
    minimal_profile_with_runtime,
)
from intergrax.hosting import resolve_hosted_application_definition

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
        instance_guard=__import__("tests.unit.hosting.engine._fakes", fromlist=["FakeInstanceGuard"]).FakeInstanceGuard(),
        health_poll_interval_seconds=0.01,
    )


@pytest.mark.asyncio
async def test_stop_before_start_rejected() -> None:
    engine = _build_engine()
    with pytest.raises(HostedApplicationShutdownError):
        await engine.stop()


@pytest.mark.asyncio
async def test_terminal_observers_drained_after_stop() -> None:
    engine = _build_engine()
    await engine.start()
    result = await engine.stop(reason_code="test.complete")
    assert result.terminal_state is HostedApplicationLifecycleState.STOPPED
    assert result.diagnostics.observer_task_count == 0


@pytest.mark.asyncio
async def test_component_stop_failure_continues_cleanup() -> None:
    from intergrax.hosting import HostedApplicationComponentRegistration, HostedApplicationProfile
    from tests.unit.hosting.engine._fakes import FakeComponent, FakeInstanceGuard, minimal_profile_with_runtime

    class FailingStopComponent(FakeComponent):
        async def stop(self, context) -> None:
            raise RuntimeError("stop failed")

    clock = FixedClock()
    component = FailingStopComponent("worker")
    base_profile = minimal_profile_with_runtime()
    profile = HostedApplicationProfile(
        application_id=base_profile.application_id,
        application_factory=base_profile.application_factory,
        application_factory_id=base_profile.application_factory_id,
        components=(
            HostedApplicationComponentRegistration(
                component=component,
                component_type_id="tests.failing_stop_component",
            ),
        ),
    )
    definition = resolve_hosted_application_definition(profile)
    guard = FakeInstanceGuard()
    engine = HostedApplicationEngine(
        definition=definition,
        instance_id="instance-001",
        paths=build_engine_paths(),
        process_identity=build_process_identity(clock),
        clock=clock,
        logger=NoopLogger(),
        shutdown=FakeShutdownCoordinator(),
        event_publisher=RecordingPublisher(),
        instance_guard=guard,
        health_poll_interval_seconds=0.01,
    )
    await engine.start()
    result = await engine.stop(reason_code="test.complete")
    assert result.terminal_state is HostedApplicationLifecycleState.STOPPED
    assert guard.lease.released
