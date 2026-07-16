# © Artur Czarnecki. All rights reserved.

"""Prove engine refreshes component health after blocking before_ready hooks."""

from __future__ import annotations

import pytest

from intergrax.hosting import (
    HostedApplicationEngine,
    HostedApplicationHook,
    HostedApplicationHooks,
    HostedApplicationLifecycleState,
    HostedApplicationProfile,
    resolve_hosted_application_definition,
)
from intergrax.hosting.errors import HostedApplicationStartupError
from tests.unit.hosting.engine._fakes import (
    FakeComponent,
    FakeInstanceGuard,
    FakeRuntime,
    FakeShutdownCoordinator,
    FixedClock,
    NoopLogger,
    RecordingPublisher,
    build_engine_paths,
    build_process_identity,
    component_registration,
    minimal_profile_with_runtime,
)

pytestmark = pytest.mark.unit


def _build_engine_with_boundary_component(
    component: FakeComponent,
    *,
    mark_ready_in_hook: bool,
) -> HostedApplicationEngine:
    clock = FixedClock()
    base = minimal_profile_with_runtime(FakeRuntime())

    async def before_ready_hook(_context) -> None:
        if mark_ready_in_hook:
            component.ready = True

    profile = HostedApplicationProfile(
        application_id=base.application_id,
        application_factory=base.application_factory,
        application_factory_id=base.application_factory_id,
        components=(component_registration(component, required=True),),
        hooks=HostedApplicationHooks(
            before_ready=(
                HostedApplicationHook(
                    hook_id="mark_boundary_ready",
                    handler=before_ready_hook,
                    handler_id="tests.unit.hosting.engine.mark_boundary_ready",
                ),
            ),
        ),
    )
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
async def test_refresh_after_before_ready_allows_startup_gate() -> None:
    component = FakeComponent("boundary", required=True, healthy=True, ready=False)
    engine = _build_engine_with_boundary_component(component, mark_ready_in_hook=True)

    await engine.start()

    assert component.started is True
    assert component.ready is True
    assert engine.lifecycle_snapshot().state is HostedApplicationLifecycleState.READY
    health = engine.health_snapshot().component_snapshots
    by_id = {item.component_id: item for item in health}
    assert by_id["boundary"].healthy is True
    assert by_id["boundary"].ready is True
    await engine.stop(reason_code="test.complete")


@pytest.mark.asyncio
async def test_startup_gate_fails_when_component_stays_not_ready_after_before_ready() -> (
    None
):
    component = FakeComponent("boundary", required=True, healthy=True, ready=False)
    engine = _build_engine_with_boundary_component(component, mark_ready_in_hook=False)

    with pytest.raises(HostedApplicationStartupError):
        await engine.start()

    assert component.started is True
    assert component.ready is False
    assert engine.lifecycle_snapshot().state is HostedApplicationLifecycleState.FAILED
