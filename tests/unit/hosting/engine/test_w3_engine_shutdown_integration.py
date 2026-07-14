# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path

import pytest

from intergrax.hosting import (
    HostedApplicationEngine,
    HostedApplicationProfile,
    InstancePolicy,
    ShutdownPolicy,
    resolve_hosted_application_definition,
)
from intergrax.hosting.contracts.events import HostedApplicationEventType
from intergrax.hosting.contracts.lifecycle import HostedApplicationEffectiveControlRequest
from intergrax.hosting.control import HostedApplicationControlCoordinator
from intergrax.hosting.errors import HostedApplicationInstanceGuardError
from intergrax.hosting.instance.contracts import HostedApplicationInstanceIdentity
from intergrax.hosting.instance.file_guard import (
    FileHostedApplicationInstanceGuard,
    FileHostedApplicationInstanceLease,
    lease_native_lock_for_tests,
    lease_release_verified_for_tests,
)
from intergrax.hosting.shutdown import (
    HostedApplicationShutdownPhase,
    HostedApplicationShutdownPhaseOutcome,
)
from intergrax.hosting.supervisor.classification import HostedApplicationExitClassifier
from tests.unit.hosting.engine._fakes import (
    FakeInstanceGuard,
    FakeLease,
    FakeMonotonicClock,
    FakeRuntime,
    FixedClock,
    HangingReleaseLease,
    NoopLogger,
    RecordingPublisher,
    build_engine_paths,
    build_process_identity,
    minimal_profile_with_runtime,
)

pytestmark = pytest.mark.unit


def _engine(
    *,
    shutdown: HostedApplicationControlCoordinator | None = None,
    publisher: RecordingPublisher | None = None,
    lease=None,
    monotonic: FakeMonotonicClock | None = None,
    shutdown_policy=None,
    runtime: FakeRuntime | None = None,
) -> tuple[HostedApplicationEngine, HostedApplicationControlCoordinator, RecordingPublisher]:
    clock = FixedClock()
    control = shutdown or HostedApplicationControlCoordinator(clock=clock)
    publisher = publisher or RecordingPublisher()
    profile = minimal_profile_with_runtime(runtime or FakeRuntime())
    if shutdown_policy is not None:
        profile = HostedApplicationProfile(
            application_id=profile.application_id,
            application_factory=profile.application_factory,
            application_factory_id=profile.application_factory_id,
            shutdown=shutdown_policy,
        )
    definition = resolve_hosted_application_definition(profile)
    engine = HostedApplicationEngine(
        definition=definition,
        instance_id="instance-001",
        paths=build_engine_paths(),
        process_identity=build_process_identity(clock),
        clock=clock,
        logger=NoopLogger(),
        shutdown=control,
        event_publisher=publisher,
        instance_guard=FakeInstanceGuard(lease or FakeLease()),
        health_poll_interval_seconds=0.01,
        monotonic_clock=monotonic or FakeMonotonicClock(),
    )
    return engine, control, publisher


@pytest.mark.asyncio
async def test_expired_deadline_sets_shutdown_timed_out() -> None:
    clock = FixedClock()
    control = HostedApplicationControlCoordinator(clock=clock)
    engine, _, _ = _engine(shutdown=control, monotonic=FakeMonotonicClock())
    await engine.start()
    deadline = clock.now() + timedelta(milliseconds=1)
    request = HostedApplicationEffectiveControlRequest(
        intent="stop",
        reason_code="deadline.expired",
        requested_at=clock.now(),
        deadline_at=deadline,
    )
    clock.advance(1.0)
    result = await engine.stop(reason_code="deadline.expired", control_request=request)
    snapshot = result.diagnostics.shutdown_execution
    assert snapshot is not None
    assert snapshot.timed_out is True


@pytest.mark.asyncio
async def test_hanging_lease_release_bounded() -> None:
    clock = FixedClock()
    control = HostedApplicationControlCoordinator(clock=clock)
    deadline = clock.now() + timedelta(milliseconds=50)
    engine, control, _ = _engine(
        shutdown=control,
        lease=HangingReleaseLease(),
        monotonic=FakeMonotonicClock(),
    )
    await engine.start()
    control.request_shutdown("stop.now", deadline_at=deadline)
    clock.advance(1.0)
    result = await engine.run_until_stopped()
    snapshot = result.diagnostics.shutdown_execution
    assert snapshot is not None
    lease_phase = next(r for r in snapshot.phase_records if r.phase is HostedApplicationShutdownPhase.LEASE_RELEASE)
    assert lease_phase.outcome in {
        HostedApplicationShutdownPhaseOutcome.TIMED_OUT,
        HostedApplicationShutdownPhaseOutcome.SKIPPED,
    }


@pytest.mark.asyncio
async def test_terminal_subscriber_drain_in_snapshot() -> None:
    engine, control, _ = _engine()
    await engine.start()
    control.request_shutdown("stop.now")
    result = await engine.run_until_stopped()
    snapshot = result.diagnostics.shutdown_execution
    assert snapshot is not None
    phases = [record.phase for record in snapshot.phase_records]
    assert phases[-1] is HostedApplicationShutdownPhase.TERMINAL_SUBSCRIBER_DRAIN


@pytest.mark.asyncio
async def test_cancel_immediately_success_not_forced_termination() -> None:
    engine, control, _ = _engine(shutdown_policy=ShutdownPolicy.cancel_immediately())
    await engine.start()
    control.request_shutdown("stop.now")
    result = await engine.run_until_stopped()
    classifier = HostedApplicationExitClassifier()
    exit_record = classifier.classify_terminal_result(
        result,
        application_id=engine.definition.application_id,
        instance_id=engine.instance_id,
        profile_digest=engine.definition.profile_digest,
        occurred_at=FixedClock().now(),
        shutdown_execution=result.diagnostics.shutdown_execution,
    )
    assert exit_record.shutdown_forced is False
    assert exit_record.exit_kind.value != "forced_termination"


@pytest.mark.asyncio
async def test_startup_abort_preserves_restart_deadline() -> None:
    clock = FixedClock()
    control = HostedApplicationControlCoordinator(clock=clock)
    runtime = FakeRuntime(start_delay=0.2)
    profile = HostedApplicationProfile(
        application_id="test_app",
        application_factory=minimal_profile_with_runtime(runtime).application_factory,
        application_factory_id="tests.unit.hosting.engine._fakes.test_app_runtime_factory",
    )
    definition = resolve_hosted_application_definition(profile)
    engine = HostedApplicationEngine(
        definition=definition,
        instance_id="instance-001",
        paths=build_engine_paths(),
        process_identity=build_process_identity(clock),
        clock=clock,
        logger=NoopLogger(),
        shutdown=control,
        event_publisher=RecordingPublisher(),
        instance_guard=FakeInstanceGuard(),
        health_poll_interval_seconds=0.01,
    )
    deadline = clock.now() + timedelta(seconds=1)

    async def _request_restart_during_startup() -> None:
        await asyncio.sleep(0.05)
        control.request_restart("restart.during.startup", deadline_at=deadline)

    restart_task = asyncio.create_task(_request_restart_during_startup())
    await engine.start()
    await restart_task
    snapshot = engine.diagnostics_snapshot().shutdown_execution
    assert snapshot is not None
    assert snapshot.effective_deadline_at == deadline
    effective = control.current_effective_request()
    assert effective is not None
    assert effective.intent == "restart"


@pytest.mark.asyncio
async def test_ftruncate_failure_no_instance_released_event(tmp_path: Path) -> None:
    import os

    clock = FixedClock()
    guard = FileHostedApplicationInstanceGuard(
        run_directory=tmp_path,
        instance_policy=InstancePolicy(),
        process_identity=build_process_identity(clock),
        clock=clock,
    )
    identity = HostedApplicationInstanceIdentity(
        application_id="test_app",
        instance_id="instance-001",
        profile_digest="sha256:" + "a" * 64,
        process_identity=build_process_identity(clock),
    )
    lease = (await guard.acquire(identity)).lease
    assert isinstance(lease, FileHostedApplicationInstanceLease)
    native = lease_native_lock_for_tests(lease)
    original_ftruncate = os.ftruncate

    def failing_ftruncate(fd: int, length: int) -> None:
        raise OSError("ftruncate failed")

    os.ftruncate = failing_ftruncate  # type: ignore[assignment]
    try:
        with pytest.raises(HostedApplicationInstanceGuardError):
            await lease.release()
    finally:
        os.ftruncate = original_ftruncate
    assert lease_release_verified_for_tests(lease) is False
    assert native.held is False

    class GuardErrorLease(FakeLease):
        async def release(self) -> None:
            raise HostedApplicationInstanceGuardError("lease_truncate_failed")

    publisher = RecordingPublisher()
    control = HostedApplicationControlCoordinator(clock=clock)
    engine, _, pub = _engine(shutdown=control, publisher=publisher, lease=GuardErrorLease())
    await engine.start()
    control.request_shutdown("stop.now")
    await engine.stop(reason_code="stop.now")
    released = [event for event in pub.events if event.event_type is HostedApplicationEventType.INSTANCE_RELEASED]
    assert released == []
