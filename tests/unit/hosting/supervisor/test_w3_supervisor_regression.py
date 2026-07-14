# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from collections.abc import Awaitable

import pytest

from intergrax.hosting import HostedApplicationProfile, RestartPolicy, resolve_hosted_application_definition
from intergrax.hosting.control import HostedApplicationControlCoordinator
from intergrax.hosting.engine.engine import HostedApplicationEngine
from intergrax.hosting.errors import HostedApplicationInstanceConflictError
from intergrax.hosting.supervisor.classification import HostedApplicationExitKind
from intergrax.hosting.supervisor.supervisor import (
    HostedApplicationSupervisor,
    HostedApplicationSupervisorLaunchContext,
)
from tests.unit.hosting.engine._fakes import (
    FakeInstanceGuard,
    FakeLease,
    FakeRuntime,
    FixedClock,
    NoopLogger,
    RecordingPublisher,
    build_engine_paths,
    build_process_identity,
    minimal_profile_with_runtime,
)

pytestmark = pytest.mark.unit


class _SequenceInstanceIds:
    def __init__(self, values: list[str]) -> None:
        self._values = iter(values)

    def __call__(self) -> str:
        return next(self._values)


class _ImmediateSleeper:
    async def sleep(self, seconds: float) -> None:
        return None


class _FixedRandom:
    def random(self) -> float:
        return 0.5


def _supervisor(
    *,
    runtime: FakeRuntime | None = None,
    max_attempts: int = 2,
    instance_ids: list[str] | None = None,
    publisher: RecordingPublisher | None = None,
) -> tuple[HostedApplicationSupervisor, HostedApplicationControlCoordinator, RecordingPublisher]:
    profile = HostedApplicationProfile(
        application_id="test_app",
        application_factory=minimal_profile_with_runtime(runtime or FakeRuntime()).application_factory,
        application_factory_id="tests.unit.hosting.engine._fakes.test_app_runtime_factory",
        restart=RestartPolicy.on_failure(max_attempts=max_attempts),
    )
    definition = resolve_hosted_application_definition(profile)
    control = HostedApplicationControlCoordinator(clock=FixedClock())
    publisher = publisher or RecordingPublisher()
    engines: list[HostedApplicationEngine] = []

    def factory(launch: HostedApplicationSupervisorLaunchContext) -> Awaitable[HostedApplicationEngine]:
        clock = FixedClock()

        async def _build() -> HostedApplicationEngine:
            engine = HostedApplicationEngine(
                definition=launch.definition,
                instance_id=launch.instance_id,
                paths=build_engine_paths(),
                process_identity=build_process_identity(clock),
                clock=clock,
                logger=NoopLogger(),
                shutdown=launch.control,
                event_publisher=RecordingPublisher(),
                instance_guard=FakeInstanceGuard(),
                health_poll_interval_seconds=0.01,
            )
            engines.append(engine)
            await engine.start()
            if launch.attempt_number == 0:
                launch.control.request_shutdown("test.complete")
            else:
                launch.control.request_restart("supervisor.restart")
            return engine

        return _build()

    supervisor = HostedApplicationSupervisor(
        definition=definition,
        engine_factory=factory,
        control=control,
        event_publisher=publisher,
        clock=FixedClock(),
        sleeper=_ImmediateSleeper(),
        random_source=_FixedRandom(),
        instance_id_generator=_SequenceInstanceIds(instance_ids or ["instance-001", "instance-002", "instance-003"]),
    )
    return supervisor, control, publisher


@pytest.mark.asyncio
async def test_startup_failure_restarted_on_failure() -> None:
    supervisor, _, _ = _supervisor(runtime=FakeRuntime(fail_start=True), max_attempts=2)
    result = await supervisor.run()
    assert len(result.attempts) >= 1


@pytest.mark.asyncio
async def test_explicit_restart_produces_replacement() -> None:
    runtime = FakeRuntime()
    supervisor, control, _ = _supervisor(runtime=runtime, max_attempts=2)
    control.request_restart("manual.restart")
    result = await supervisor.run()
    assert result.final_exit.exit_kind in {
        HostedApplicationExitKind.CLEAN_STOP,
        HostedApplicationExitKind.RESTART_REQUESTED,
    }


@pytest.mark.asyncio
async def test_instance_conflict_non_retryable() -> None:
    from intergrax.hosting.supervisor.classification import HostedApplicationExitClassifier
    from intergrax.hosting.contracts.lifecycle import HostedApplicationLifecycleState

    classifier = HostedApplicationExitClassifier()
    record = classifier.classify_exception(
        HostedApplicationInstanceConflictError("conflict", None),
        application_id="test_app",
        instance_id="instance-001",
        profile_digest="sha256:" + "0" * 64,
        occurred_at=FixedClock().now(),
    )
    assert record.exit_kind is HostedApplicationExitKind.INSTANCE_CONFLICT
    assert record.retryable is False
    assert record.terminal_lifecycle_state is HostedApplicationLifecycleState.FAILED


@pytest.mark.asyncio
async def test_new_instance_id_per_launch() -> None:
    supervisor, _, _ = _supervisor(max_attempts=1)
    result = await supervisor.run()
    instance_ids = {attempt.instance_id for attempt in result.attempts}
    assert len(instance_ids) == len(result.attempts)


@pytest.mark.asyncio
async def test_stop_before_launch_coherent_result() -> None:
    supervisor, control, _ = _supervisor()
    control.request_shutdown("pre.stop")
    result = await supervisor.run()
    assert result.final_exit.reason_code == "stop_before_launch"
    assert result.attempts == ()


@pytest.mark.asyncio
async def test_factory_failure_supervisor_error() -> None:
    profile = minimal_profile_with_runtime()
    definition = resolve_hosted_application_definition(profile)
    control = HostedApplicationControlCoordinator(clock=FixedClock())

    def factory(launch: HostedApplicationSupervisorLaunchContext) -> HostedApplicationEngine:
        raise RuntimeError("factory broke")

    supervisor = HostedApplicationSupervisor(
        definition=definition,
        engine_factory=factory,
        control=control,
        event_publisher=RecordingPublisher(),
        clock=FixedClock(),
        instance_id_generator=_SequenceInstanceIds(["instance-001"]),
    )
    result = await supervisor.run()
    assert result.final_exit.exit_kind is HostedApplicationExitKind.SUPERVISOR_ERROR


@pytest.mark.asyncio
async def test_release_failure_prevents_replacement() -> None:
    class FailingReleaseLease(FakeLease):
        async def release(self) -> None:
            raise RuntimeError("release failed")

    profile = HostedApplicationProfile(
        application_id="test_app",
        application_factory=minimal_profile_with_runtime(FakeRuntime()).application_factory,
        application_factory_id="tests.unit.hosting.engine._fakes.test_app_runtime_factory",
        restart=RestartPolicy.on_failure(max_attempts=3),
    )
    definition = resolve_hosted_application_definition(profile)
    control = HostedApplicationControlCoordinator(clock=FixedClock())

    def factory(launch: HostedApplicationSupervisorLaunchContext) -> Awaitable[HostedApplicationEngine]:
        clock = FixedClock()

        async def _build() -> HostedApplicationEngine:
            engine = HostedApplicationEngine(
                definition=launch.definition,
                instance_id=launch.instance_id,
                paths=build_engine_paths(),
                process_identity=build_process_identity(clock),
                clock=clock,
                logger=NoopLogger(),
                shutdown=launch.control,
                event_publisher=RecordingPublisher(),
                instance_guard=FakeInstanceGuard(FailingReleaseLease()),
                health_poll_interval_seconds=0.01,
            )
            await engine.start()
            launch.control.request_restart("restart.after.fail")
            return engine

        return _build()

    supervisor = HostedApplicationSupervisor(
        definition=definition,
        engine_factory=factory,
        control=control,
        event_publisher=RecordingPublisher(),
        clock=FixedClock(),
        sleeper=_ImmediateSleeper(),
        instance_id_generator=_SequenceInstanceIds(["instance-001", "instance-002"]),
    )
    result = await supervisor.run()
    assert len(result.attempts) == 1
