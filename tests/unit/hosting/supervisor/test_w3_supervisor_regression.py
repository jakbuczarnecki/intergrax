# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import asyncio
from collections.abc import Awaitable

import pytest

from intergrax.hosting.contracts.policies import RestartMode
from intergrax.hosting import HostedApplicationProfile, RestartPolicy, resolve_hosted_application_definition
from intergrax.hosting.contracts.events import HostedApplicationEventType
from intergrax.hosting.contracts.lifecycle import HostedApplicationLifecycleState
from intergrax.hosting.control import HostedApplicationControlCoordinator
from intergrax.hosting.engine.engine import HostedApplicationEngine
from intergrax.hosting.errors import HostedApplicationInstanceConflictError
from intergrax.hosting.supervisor.classification import HostedApplicationExitClassifier, HostedApplicationExitKind
from intergrax.hosting.supervisor.restart import HostedApplicationRestartPolicyEvaluator
from intergrax.hosting.supervisor.supervisor import (
    HostedApplicationSupervisor,
    HostedApplicationSupervisorLaunchContext,
)
from tests.unit.hosting.engine._fakes import (
    AdvancingSleeper,
    FakeInstanceGuard,
    FakeMonotonicClock,
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


class _FixedRandom:
    def random(self) -> float:
        return 0.5


def _build_supervisor(
    *,
    runtime: FakeRuntime | None = None,
    max_attempts: int = 2,
    instance_ids: list[str] | None = None,
    publisher: RecordingPublisher | None = None,
    factory_mode: str = "auto_stop",
    monotonic: FakeMonotonicClock | None = None,
    ready_event: asyncio.Event | None = None,
    fail_start_attempts: frozenset[int] | None = None,
    prestop_replacement: bool = True,
) -> tuple[HostedApplicationSupervisor, HostedApplicationControlCoordinator, RecordingPublisher, list[HostedApplicationEngine]]:
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
    monotonic = monotonic or FakeMonotonicClock()
    sleeper = AdvancingSleeper(monotonic)

    def factory(launch: HostedApplicationSupervisorLaunchContext) -> Awaitable[HostedApplicationEngine]:
        clock = FixedClock()
        from tests.unit.hosting.engine import _fakes as fakes_module

        if fail_start_attempts is not None:
            fakes_module._RUNTIME_HOLDER["runtime"] = FakeRuntime(
                fail_start=launch.attempt_number in fail_start_attempts
            )

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
                monotonic_clock=monotonic,
            )
            engines.append(engine)
            if factory_mode == "unstarted":
                if prestop_replacement and launch.attempt_number > 0:
                    launch.control.request_shutdown("test.complete")
                return engine
            await engine.start()
            if ready_event is not None and launch.attempt_number == 0:
                ready_event.set()
            if factory_mode == "auto_stop":
                launch.control.request_shutdown("test.complete")
            return engine

        return _build()

    supervisor = HostedApplicationSupervisor(
        definition=definition,
        engine_factory=factory,
        control=control,
        event_publisher=publisher,
        clock=FixedClock(),
        monotonic_clock=monotonic,
        sleeper=sleeper,
        random_source=_FixedRandom(),
        instance_id_generator=_SequenceInstanceIds(
            instance_ids or ["instance-001", "instance-002", "instance-003", "instance-004"]
        ),
    )
    return supervisor, control, publisher, engines


@pytest.mark.asyncio
async def test_startup_failure_restarted_on_failure() -> None:
    supervisor, _, _, _ = _build_supervisor(
        max_attempts=2,
        factory_mode="unstarted",
        fail_start_attempts=frozenset({0}),
    )
    result = await supervisor.run()
    assert len(result.attempts) == 2
    assert result.attempts[0].instance_id != result.attempts[1].instance_id
    assert result.attempts[0].exit_record is not None
    assert result.attempts[0].exit_record.exit_kind is HostedApplicationExitKind.STARTUP_FAILURE
    assert result.attempts[1].exit_record is not None
    assert result.attempts[1].exit_record.exit_kind is HostedApplicationExitKind.CLEAN_STOP
    assert result.attempts[0].exit_record.profile_digest == result.profile_digest
    assert result.attempts[0].exit_record.profile_digest == result.attempts[1].exit_record.profile_digest


@pytest.mark.asyncio
async def test_explicit_restart_produces_replacement() -> None:
    monotonic = FakeMonotonicClock()
    supervisor, control, _, engines = _build_supervisor(
        max_attempts=2,
        factory_mode="unstarted",
        monotonic=monotonic,
    )

    supervisor_task = asyncio.create_task(supervisor.run())
    for _ in range(500):
        if (
            engines
            and engines[0].lifecycle_snapshot().state is HostedApplicationLifecycleState.READY
        ):
            control.request_restart("manual.restart")
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("engine never reached READY")
    result = await supervisor_task

    assert len(result.attempts) == 2
    assert result.attempts[0].exit_record is not None
    assert result.attempts[0].exit_record.exit_kind is HostedApplicationExitKind.RESTART_REQUESTED
    assert result.attempts[0].instance_id != result.attempts[1].instance_id
    first_diag = result.attempts[0].terminal_result
    assert first_diag is not None
    assert first_diag.diagnostics.instance_lease_released is True


@pytest.mark.asyncio
async def test_deterministic_backoff_without_wall_clock() -> None:
    monotonic = FakeMonotonicClock()
    supervisor, _, _, _ = _build_supervisor(
        max_attempts=3,
        factory_mode="unstarted",
        monotonic=monotonic,
        fail_start_attempts=frozenset({0}),
    )
    result = await supervisor.run()
    assert len(result.attempts) == 2
    assert monotonic.monotonic() >= 1.0


@pytest.mark.asyncio
async def test_backoff_stop_interruption() -> None:
    monotonic = FakeMonotonicClock()
    sleeper = AdvancingSleeper(monotonic)
    control = HostedApplicationControlCoordinator(clock=FixedClock())
    evaluator = HostedApplicationRestartPolicyEvaluator(
        policy=RestartPolicy(
            mode=RestartMode.ON_FAILURE,
            max_attempts=3,
            initial_backoff_seconds=2.0,
            jitter_ratio=0.0,
        ),
        clock=FixedClock(),
        monotonic_clock=monotonic,
        random_source=_FixedRandom(),
    )
    control.request_shutdown("stop.during.backoff")
    allowed = await evaluator.wait_backoff(2.0, control=control, sleeper=sleeper)
    assert allowed is False
    assert sleeper.sleep_calls == []
    assert monotonic.monotonic() == 0.0


@pytest.mark.asyncio
async def test_factory_exhaustion_reports_restart_exhausted() -> None:
    profile = HostedApplicationProfile(
        application_id="test_app",
        application_factory=minimal_profile_with_runtime(FakeRuntime(fail_start=True)).application_factory,
        application_factory_id="tests.unit.hosting.engine._fakes.test_app_runtime_factory",
        restart=RestartPolicy.always(max_attempts=1),
    )
    definition = resolve_hosted_application_definition(profile)
    control = HostedApplicationControlCoordinator(clock=FixedClock())
    publisher = RecordingPublisher()

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
            return engine

        return _build()

    supervisor = HostedApplicationSupervisor(
        definition=definition,
        engine_factory=factory,
        control=control,
        event_publisher=publisher,
        clock=FixedClock(),
        instance_id_generator=_SequenceInstanceIds(["instance-001", "instance-002", "instance-003"]),
    )
    result = await supervisor.run()
    assert result.restart_exhausted is True
    assert len(result.attempts) == 2
    exhausted_events = [event for event in publisher.events if event.event_type is HostedApplicationEventType.RESTART_EXHAUSTED]
    assert len(exhausted_events) == 1


@pytest.mark.asyncio
async def test_instance_conflict_non_retryable() -> None:
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
async def test_stop_before_launch_coherent_result() -> None:
    supervisor, control, _, _ = _build_supervisor()
    control.request_shutdown("pre.stop")
    result = await supervisor.run()
    assert result.final_exit.reason_code == "stop_before_launch"
    assert result.attempts == ()


@pytest.mark.asyncio
async def test_release_failure_prevents_replacement() -> None:
    from tests.unit.hosting.engine._fakes import FakeLease

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
        instance_id_generator=_SequenceInstanceIds(["instance-001", "instance-002"]),
    )
    result = await supervisor.run()
    assert len(result.attempts) == 1
