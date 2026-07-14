# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.hosting import (
    HostedApplicationEngine,
    HostedApplicationLifecycleState,
    HostedApplicationProfile,
    resolve_hosted_application_definition,
)
from intergrax.hosting.contracts.policies import ComponentFailureAction
from intergrax.hosting.engine.diagnostics import HostedApplicationFailurePhase
from intergrax.hosting.engine.health import HostedApplicationHealthCoordinator
from intergrax.hosting.engine.lifecycle import HostedApplicationLifecycleController
from intergrax.hosting.errors import HostedApplicationStartupError
from tests.unit.hosting.engine._fakes import (
    FakeComponent,
    FakeInstanceGuard,
    FakeLease,
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


def _build_engine(runtime: FakeRuntime | None = None, **kwargs) -> HostedApplicationEngine:
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
        instance_guard=kwargs.get("instance_guard", FakeInstanceGuard()),
        health_poll_interval_seconds=0.01,
    )


def _health_coordinator() -> HostedApplicationHealthCoordinator:
    clock = FixedClock()
    lifecycle = HostedApplicationLifecycleController(clock)
    return HostedApplicationHealthCoordinator(lifecycle, clock)


def test_runtime_not_ready_blocks_gate() -> None:
    coordinator = _health_coordinator()
    coordinator.set_runtime_ready(False)
    coordinator.set_lease(FakeLease())
    result = coordinator.evaluate_startup_readiness_gate()
    assert not result.passed
    assert result.reason_code == "runtime_not_ready"


def test_invalid_lease_blocks_gate() -> None:
    coordinator = _health_coordinator()
    coordinator.set_runtime_ready(True)
    coordinator.set_lease(FakeLease(valid=False))
    result = coordinator.evaluate_startup_readiness_gate()
    assert not result.passed
    assert result.reason_code == "invalid_lease"


def test_required_unhealthy_blocks_gate() -> None:
    from intergrax.hosting import HostedApplicationComponentHealth, HostedApplicationComponentState

    coordinator = _health_coordinator()
    coordinator.set_runtime_ready(True)
    coordinator.set_lease(FakeLease())
    coordinator.update_component_health(
        {
            "worker": HostedApplicationComponentHealth(
                component_id="worker",
                enabled=True,
                required=True,
                state=HostedApplicationComponentState.FAILED,
                healthy=False,
                ready=False,
            )
        },
        mark_not_ready_failed=frozenset(),
        degraded_component_ids=frozenset(),
    )
    result = coordinator.evaluate_startup_readiness_gate()
    assert not result.passed
    assert result.reason_code.startswith("blocking_components:")


@pytest.mark.asyncio
async def test_startup_gate_failure_reaches_failed() -> None:
    runtime = FakeRuntime(ready=False)
    engine = _build_engine(runtime)
    with pytest.raises(HostedApplicationStartupError):
        await engine.start()
    assert engine.lifecycle_snapshot().state is HostedApplicationLifecycleState.FAILED
    diagnostics = engine.diagnostics_snapshot()
    assert diagnostics.current_failure is not None
    assert diagnostics.current_failure.phase is HostedApplicationFailurePhase.HEALTH_EVALUATION


@pytest.mark.asyncio
async def test_optional_degraded_allows_ready() -> None:
    clock = FixedClock()
    degraded = FakeComponent("degraded")
    profile = HostedApplicationProfile(
        application_id="test_app",
        application_factory=minimal_profile_with_runtime().application_factory,
        components=(
            component_registration(
                degraded,
                required=False,
                failure_action=ComponentFailureAction.MARK_DEGRADED,
            ),
        ),
    )
    degraded.fail_start = True
    definition = resolve_hosted_application_definition(profile)
    engine = HostedApplicationEngine(
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
    await engine.start()
    assert engine.lifecycle_snapshot().state is HostedApplicationLifecycleState.READY
