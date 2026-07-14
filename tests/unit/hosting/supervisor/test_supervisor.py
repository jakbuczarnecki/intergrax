# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from collections.abc import Awaitable

import pytest

from intergrax.hosting import HostedApplicationProfile, RestartPolicy, resolve_hosted_application_definition
from intergrax.hosting.contracts.policies import RestartMode
from intergrax.hosting.control import HostedApplicationControlCoordinator
from intergrax.hosting.engine.engine import HostedApplicationEngine
from intergrax.hosting.errors import HostedApplicationSupervisorError
from intergrax.hosting.supervisor.classification import HostedApplicationExitKind
from intergrax.hosting.supervisor.restart import HostedApplicationRestartPolicyEvaluator
from intergrax.hosting.supervisor.supervisor import (
    HostedApplicationSupervisor,
    HostedApplicationSupervisorLaunchContext,
)
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


class _SequenceInstanceIds:
    def __init__(self, values: list[str]) -> None:
        self._values = iter(values)

    def __call__(self) -> str:
        return next(self._values)


class _FixedRandom:
    def random(self) -> float:
        return 0.5


class _ImmediateSleeper:
    async def sleep(self, seconds: float) -> None:
        return None


def _definition_with_restart(*, max_attempts: int = 2) -> object:
    profile = HostedApplicationProfile(
        application_id="test_app",
        application_factory=minimal_profile_with_runtime(FakeRuntime()).application_factory,
        application_factory_id="tests.unit.hosting.engine._fakes.test_app_runtime_factory",
        restart=RestartPolicy.on_failure(max_attempts=max_attempts),
    )
    return resolve_hosted_application_definition(profile)


@pytest.mark.asyncio
async def test_clean_single_run() -> None:
    profile = HostedApplicationProfile(
        application_id="test_app",
        application_factory=minimal_profile_with_runtime(FakeRuntime()).application_factory,
        application_factory_id="tests.unit.hosting.engine._fakes.test_app_runtime_factory",
        restart=RestartPolicy.never(),
    )
    definition = resolve_hosted_application_definition(profile)
    control = HostedApplicationControlCoordinator(clock=FixedClock())
    publisher = RecordingPublisher()
    engines: list[HostedApplicationEngine] = []

    def factory(launch: HostedApplicationSupervisorLaunchContext) -> Awaitable[HostedApplicationEngine]:
        clock = FixedClock()
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

        async def _run() -> HostedApplicationEngine:
            await engine.start()
            launch.control.request_shutdown("test.complete")
            return engine

        return _run()

    supervisor = HostedApplicationSupervisor(
        definition=definition,
        engine_factory=factory,
        control=control,
        event_publisher=publisher,
        clock=FixedClock(),
        sleeper=_ImmediateSleeper(),
        random_source=_FixedRandom(),
        instance_id_generator=_SequenceInstanceIds(["instance-001"]),
    )
    result = await supervisor.run()
    assert result.final_exit.exit_kind is HostedApplicationExitKind.CLEAN_STOP
    assert len(result.attempts) == 1


@pytest.mark.asyncio
async def test_engine_contract_mismatch_rejected() -> None:
    definition = resolve_hosted_application_definition(minimal_profile_with_runtime())
    control = HostedApplicationControlCoordinator(clock=FixedClock())

    def factory(launch: HostedApplicationSupervisorLaunchContext) -> HostedApplicationEngine:
        clock = FixedClock()
        return HostedApplicationEngine(
            definition=launch.definition,
            instance_id="wrong-id",
            paths=build_engine_paths(),
            process_identity=build_process_identity(clock),
            clock=clock,
            logger=NoopLogger(),
            shutdown=launch.control,
            event_publisher=RecordingPublisher(),
            instance_guard=FakeInstanceGuard(),
            health_poll_interval_seconds=0.01,
        )

    supervisor = HostedApplicationSupervisor(
        definition=definition,
        engine_factory=factory,
        control=control,
        event_publisher=RecordingPublisher(),
        clock=FixedClock(),
        instance_id_generator=_SequenceInstanceIds(["instance-001"]),
    )
    with pytest.raises(HostedApplicationSupervisorError):
        await supervisor.run()


def test_restart_policy_custom_classifier_signature() -> None:
    from intergrax.hosting.supervisor.classification import HostedApplicationExitRecord, HostedApplicationExitKind
    from intergrax.hosting.contracts.lifecycle import HostedApplicationLifecycleState

    policy = RestartPolicy(
        mode=RestartMode.CUSTOM,
        max_attempts=1,
        custom_classifier=lambda exit_record: exit_record.retryable,
        custom_classifier_id="tests.custom_classifier",
    )
    evaluator = HostedApplicationRestartPolicyEvaluator(policy=policy, clock=FixedClock())
    record = HostedApplicationExitRecord(
        exit_kind=HostedApplicationExitKind.STARTUP_FAILURE,
        retryable=True,
        reason_code="startup_failure",
        application_id="test_app",
        instance_id="instance-001",
        profile_digest="sha256:" + "0" * 64,
        terminal_lifecycle_state=HostedApplicationLifecycleState.FAILED,
        occurred_at=FixedClock().now(),
    )
    decision = evaluator.evaluate(record, attempt_number=0)
    assert decision.should_restart
