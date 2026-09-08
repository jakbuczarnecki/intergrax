# © Artur Czarnecki. All rights reserved.

"""DG-001D R2 — supervisor pre-engine failure producer tests."""

from __future__ import annotations

import ast
from typing import cast
import json
from collections.abc import Awaitable
from dataclasses import replace
from pathlib import Path

import pytest

from intergrax.applications._shared.hosted_application_failure_projection import (
    hosted_application_failure_to_problem_signal,
)
from intergrax.hosting import HostedApplicationProfile, RestartPolicy, resolve_hosted_application_definition
from intergrax.hosting.contracts.events import HostedApplicationEvent, HostedApplicationEventType
from intergrax.hosting.contracts.lifecycle import HostedApplicationLifecycleState
from intergrax.hosting.control import HostedApplicationControlCoordinator
from intergrax.hosting.engine.definition import HostedApplicationDefinition
from intergrax.hosting.eventing import HostingObservabilityAttributes
from intergrax.hosting.errors import (
    HostedApplicationSupervisorError,
    HostedApplicationSupervisorFailureReason,
)
from intergrax.hosting.process_bootstrap import HostedProcessBootstrapPhase
from intergrax.hosting.supervisor.classification import HostedApplicationExitKind
from intergrax.hosting.supervisor.failure_projection import (
    HOSTED_APPLICATION_SUPERVISOR_PROCESS_ROLE,
    supervisor_pre_engine_failure_to_hosted_event,
)
from intergrax.hosting.engine.engine import HostedApplicationEngine
from intergrax.hosting.supervisor.supervisor import (
    HostedApplicationEngineFactory,
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

_SECRET_SENTINEL = "DG001D-R2-SECRET-SENTINEL"
_PUBLISHER_SENTINEL = "DG001D-R2-PUBLISHER-SENTINEL"


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


class _FailingFailurePublisher(RecordingPublisher):
    async def publish(self, event: HostedApplicationEvent) -> None:
        if event.event_type is HostedApplicationEventType.APPLICATION_FAILED:
            raise RuntimeError(_PUBLISHER_SENTINEL)
        await super().publish(event)


def _definition() -> HostedApplicationDefinition:
    profile = HostedApplicationProfile(
        application_id="test_app",
        application_factory=minimal_profile_with_runtime(FakeRuntime()).application_factory,
        application_factory_id="tests.unit.hosting.engine._fakes.test_app_runtime_factory",
        restart=RestartPolicy.never(),
    )
    return resolve_hosted_application_definition(profile)


def _build_engine(
    launch: HostedApplicationSupervisorLaunchContext,
    *,
    instance_id: str | None = None,
    definition_override: HostedApplicationDefinition | None = None,
) -> HostedApplicationEngine:
    clock = FixedClock()
    return HostedApplicationEngine(
        definition=definition_override or launch.definition,
        instance_id=instance_id or launch.instance_id,
        paths=build_engine_paths(),
        process_identity=build_process_identity(clock),
        clock=clock,
        logger=NoopLogger(),
        shutdown=launch.control,
        event_publisher=RecordingPublisher(),
        instance_guard=FakeInstanceGuard(),
        health_poll_interval_seconds=0.01,
    )


def _supervisor(
    *,
    factory: HostedApplicationEngineFactory,
    publisher: RecordingPublisher | None = None,
    control: HostedApplicationControlCoordinator | None = None,
    instance_ids: list[str] | None = None,
) -> tuple[HostedApplicationSupervisor, HostedApplicationControlCoordinator, RecordingPublisher]:
    definition = _definition()
    control = control or HostedApplicationControlCoordinator(clock=FixedClock())
    publisher = publisher or RecordingPublisher()
    supervisor = HostedApplicationSupervisor(
        definition=definition,
        engine_factory=factory,
        control=control,
        event_publisher=publisher,
        clock=FixedClock(),
        sleeper=_ImmediateSleeper(),
        random_source=_FixedRandom(),
        instance_id_generator=_SequenceInstanceIds(instance_ids or ["instance-001"]),
    )
    return supervisor, control, publisher


def _failed_events(publisher: RecordingPublisher) -> list[HostedApplicationEvent]:
    return [
        event
        for event in publisher.events
        if event.event_type is HostedApplicationEventType.APPLICATION_FAILED
    ]


def _assert_single_pre_engine_failure(
    publisher: RecordingPublisher,
    *,
    instance_id: str,
    phase: HostedProcessBootstrapPhase,
    reason_code: str,
) -> HostedApplicationEvent:
    events = _failed_events(publisher)
    assert len(events) == 1
    event = events[0]
    assert event.lifecycle_state is HostedApplicationLifecycleState.FAILED
    assert event.application_id == "test_app"
    assert event.instance_id == instance_id
    assert event.instance_id != "supervisor"
    payload = event.payload
    assert payload is not None
    assert payload["phase"] == phase.value
    assert payload["reason_code"] == reason_code
    assert payload["exception_type"] == HostedApplicationSupervisorError.__name__
    assert payload["process_role"] == HOSTED_APPLICATION_SUPERVISOR_PROCESS_ROLE
    assert _SECRET_SENTINEL not in json.dumps(payload)
    return event


@pytest.mark.asyncio
async def test_factory_raises_sync_emits_application_failed() -> None:
    def factory(launch: HostedApplicationSupervisorLaunchContext) -> HostedApplicationEngine:
        raise RuntimeError(_SECRET_SENTINEL)

    supervisor, _, publisher = _supervisor(factory=factory)
    result = await supervisor.run()
    _assert_single_pre_engine_failure(
        publisher,
        instance_id="instance-001",
        phase=HostedProcessBootstrapPhase.ENGINE_CONSTRUCTION,
        reason_code=HostedApplicationSupervisorFailureReason.ENGINE_FACTORY_FAILED.value,
    )
    assert result.final_exit.exit_kind is HostedApplicationExitKind.SUPERVISOR_ERROR
    assert result.final_exit.retryable is False


@pytest.mark.asyncio
async def test_factory_raises_async_emits_application_failed() -> None:
    def factory(launch: HostedApplicationSupervisorLaunchContext) -> Awaitable[HostedApplicationEngine]:
        async def _raise() -> HostedApplicationEngine:
            raise RuntimeError(_SECRET_SENTINEL)

        return _raise()

    supervisor, _, publisher = _supervisor(factory=factory)
    await supervisor.run()
    _assert_single_pre_engine_failure(
        publisher,
        instance_id="instance-001",
        phase=HostedProcessBootstrapPhase.ENGINE_CONSTRUCTION,
        reason_code=HostedApplicationSupervisorFailureReason.ENGINE_FACTORY_FAILED.value,
    )


@pytest.mark.asyncio
async def test_factory_invalid_type_emits_application_failed() -> None:
    def factory(launch: HostedApplicationSupervisorLaunchContext) -> HostedApplicationEngine:
        return cast(HostedApplicationEngine, object())

    supervisor, _, publisher = _supervisor(factory=factory)
    await supervisor.run()
    _assert_single_pre_engine_failure(
        publisher,
        instance_id="instance-001",
        phase=HostedProcessBootstrapPhase.ENGINE_CONSTRUCTION,
        reason_code=HostedApplicationSupervisorFailureReason.ENGINE_FACTORY_INVALID_RESULT.value,
    )


@pytest.mark.asyncio
async def test_instance_id_mismatch_emits_application_failed() -> None:
    def factory(launch: HostedApplicationSupervisorLaunchContext) -> HostedApplicationEngine:
        return _build_engine(launch, instance_id="wrong-id")

    supervisor, _, publisher = _supervisor(factory=factory)
    await supervisor.run()
    _assert_single_pre_engine_failure(
        publisher,
        instance_id="instance-001",
        phase=HostedProcessBootstrapPhase.ENGINE_CONTRACT_VALIDATION,
        reason_code=HostedApplicationSupervisorFailureReason.ENGINE_INSTANCE_ID_MISMATCH.value,
    )


@pytest.mark.asyncio
async def test_profile_digest_mismatch_emits_application_failed() -> None:
    def factory(launch: HostedApplicationSupervisorLaunchContext) -> HostedApplicationEngine:
        wrong_definition = replace(
            launch.definition,
            profile_digest="sha256:" + "1" * 64,
        )
        return _build_engine(launch, definition_override=wrong_definition)

    supervisor, _, publisher = _supervisor(factory=factory)
    await supervisor.run()
    _assert_single_pre_engine_failure(
        publisher,
        instance_id="instance-001",
        phase=HostedProcessBootstrapPhase.ENGINE_CONTRACT_VALIDATION,
        reason_code=HostedApplicationSupervisorFailureReason.ENGINE_PROFILE_DIGEST_MISMATCH.value,
    )


@pytest.mark.asyncio
async def test_definition_digest_mismatch_emits_application_failed() -> None:
    def factory(launch: HostedApplicationSupervisorLaunchContext) -> HostedApplicationEngine:
        wrong_definition = replace(
            launch.definition,
            definition_digest="sha256:" + "2" * 64,
        )
        return _build_engine(launch, definition_override=wrong_definition)

    supervisor, _, publisher = _supervisor(factory=factory)
    await supervisor.run()
    _assert_single_pre_engine_failure(
        publisher,
        instance_id="instance-001",
        phase=HostedProcessBootstrapPhase.ENGINE_CONTRACT_VALIDATION,
        reason_code=HostedApplicationSupervisorFailureReason.ENGINE_DEFINITION_DIGEST_MISMATCH.value,
    )


@pytest.mark.asyncio
async def test_application_id_mismatch_emits_application_failed() -> None:
    def factory(launch: HostedApplicationSupervisorLaunchContext) -> HostedApplicationEngine:
        wrong_definition = replace(launch.definition, application_id="other_app")
        return _build_engine(launch, definition_override=wrong_definition)

    supervisor, _, publisher = _supervisor(factory=factory)
    await supervisor.run()
    _assert_single_pre_engine_failure(
        publisher,
        instance_id="instance-001",
        phase=HostedProcessBootstrapPhase.ENGINE_CONTRACT_VALIDATION,
        reason_code=HostedApplicationSupervisorFailureReason.ENGINE_APPLICATION_ID_MISMATCH.value,
    )


@pytest.mark.asyncio
async def test_success_emits_no_pre_engine_failure() -> None:
    def factory(launch: HostedApplicationSupervisorLaunchContext) -> Awaitable[HostedApplicationEngine]:
        async def _build() -> HostedApplicationEngine:
            engine = _build_engine(launch)
            await engine.start()
            launch.control.request_shutdown("test.complete")
            return engine

        return _build()

    supervisor, _, publisher = _supervisor(factory=factory)
    result = await supervisor.run()
    assert _failed_events(publisher) == []
    assert result.final_exit.exit_kind is HostedApplicationExitKind.CLEAN_STOP


@pytest.mark.asyncio
async def test_stop_before_launch_emits_no_failure() -> None:
    def factory(launch: HostedApplicationSupervisorLaunchContext) -> HostedApplicationEngine:
        return _build_engine(launch)

    control = HostedApplicationControlCoordinator(clock=FixedClock())
    control.request_shutdown("test.stop")
    supervisor, _, publisher = _supervisor(factory=factory, control=control)
    result = await supervisor.run()
    assert _failed_events(publisher) == []
    assert result.final_exit.reason_code == "stop_before_launch"


@pytest.mark.asyncio
async def test_publisher_failure_isolated() -> None:
    def factory(launch: HostedApplicationSupervisorLaunchContext) -> HostedApplicationEngine:
        raise RuntimeError(_SECRET_SENTINEL)

    baseline_publisher = RecordingPublisher()
    baseline_supervisor, _, _ = _supervisor(factory=factory, publisher=baseline_publisher)
    baseline_result = await baseline_supervisor.run()

    failing_publisher = _FailingFailurePublisher()
    failing_supervisor, _, _ = _supervisor(factory=factory, publisher=failing_publisher)
    failing_result = await failing_supervisor.run()

    assert failing_result.final_exit.exit_kind == baseline_result.final_exit.exit_kind
    assert failing_result.final_exit.retryable == baseline_result.final_exit.retryable
    assert failing_result.final_exit.reason_code == baseline_result.final_exit.reason_code
    assert failing_result.restart_exhausted == baseline_result.restart_exhausted
    assert _failed_events(failing_publisher) == []


@pytest.mark.asyncio
async def test_runtime_engine_failure_no_supervisor_duplicate() -> None:
    from tests.unit.hosting.engine import _fakes as fakes_module

    def factory(launch: HostedApplicationSupervisorLaunchContext) -> Awaitable[HostedApplicationEngine]:
        async def _build() -> HostedApplicationEngine:
            fakes_module._RUNTIME_HOLDER["runtime"] = FakeRuntime(fail_start=True)
            engine = _build_engine(launch)
            return engine

        return _build()

    supervisor, _, publisher = _supervisor(factory=factory)
    result = await supervisor.run()
    assert _failed_events(publisher) == []
    assert result.final_exit.exit_kind is HostedApplicationExitKind.STARTUP_FAILURE


@pytest.mark.asyncio
async def test_failure_emitted_before_restart_events() -> None:
    def factory(launch: HostedApplicationSupervisorLaunchContext) -> HostedApplicationEngine:
        raise RuntimeError(_SECRET_SENTINEL)

    supervisor, _, publisher = _supervisor(factory=factory)
    await supervisor.run()
    failed_index = next(
        index
        for index, event in enumerate(publisher.events)
        if event.event_type is HostedApplicationEventType.APPLICATION_FAILED
    )
    restart_indices = [
        index
        for index, event in enumerate(publisher.events)
        if event.event_type
        in {
            HostedApplicationEventType.RESTART_REQUESTED,
            HostedApplicationEventType.RESTART_SCHEDULED,
            HostedApplicationEventType.RESTART_STARTED,
        }
    ]
    assert restart_indices == []
    assert failed_index == len(publisher.events) - 1


def test_supervisor_pre_engine_failure_projector_compatibility() -> None:
    failure = HostedApplicationSupervisorError(
        "engine factory failed",
        reason=HostedApplicationSupervisorFailureReason.ENGINE_FACTORY_FAILED,
        phase=HostedProcessBootstrapPhase.ENGINE_CONSTRUCTION,
    )
    event = supervisor_pre_engine_failure_to_hosted_event(
        application_id="test_app",
        instance_id="instance-001",
        failure=failure,
        occurred_at=FixedClock().now(),
    )
    signal = hosted_application_failure_to_problem_signal(event)
    assert signal is not None
    assert signal.error_code == HostedApplicationSupervisorFailureReason.ENGINE_FACTORY_FAILED.value
    assert signal.source_component == HostedProcessBootstrapPhase.ENGINE_CONSTRUCTION.value
    assert signal.application_attributes is not None
    assert isinstance(signal.application_attributes, HostingObservabilityAttributes)
    application_attributes = signal.application_attributes
    assert application_attributes.application_id == "test_app"
    assert application_attributes.instance_id == "instance-001"


def test_bootstrap_phase_enum_backward_compatibility() -> None:
    assert HostedProcessBootstrapPhase.CONFIGURATION.value == "configuration"
    assert HostedProcessBootstrapPhase.COMPOSITION.value == "composition"
    assert HostedProcessBootstrapPhase.DEPENDENCY_RESOLUTION.value == "dependency_resolution"
    assert HostedProcessBootstrapPhase.WORKER_CONSTRUCTION.value == "worker_construction"
    assert HostedProcessBootstrapPhase.STARTUP.value == "startup"
    assert HostedProcessBootstrapPhase.ENGINE_CONSTRUCTION.value == "engine_construction"
    assert HostedProcessBootstrapPhase.ENGINE_CONTRACT_VALIDATION.value == "engine_contract_validation"
    serialized = {phase.name: phase.value for phase in HostedProcessBootstrapPhase}
    assert serialized["ENGINE_CONSTRUCTION"] == "engine_construction"
    assert serialized["ENGINE_CONTRACT_VALIDATION"] == "engine_contract_validation"


_FORBIDDEN_IMPORT_MODULES = frozenset(
    {
        "intergrax.runtime.diagnostics",
        "intergrax.applications",
        "local_workspace_application",
    }
)

_SUPERVISOR_MODULE_PATHS = tuple(
    sorted(
        (Path(__file__).resolve().parents[4] / "intergrax" / "hosting" / "supervisor").glob("*.py")
    )
)


def _collect_import_boundary_violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    rel = path.relative_to(Path(__file__).resolve().parents[4]).as_posix()
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            for forbidden in _FORBIDDEN_IMPORT_MODULES:
                if node.module == forbidden or node.module.startswith(f"{forbidden}."):
                    violations.append(f"{rel}:{node.lineno} imports from {node.module}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                for forbidden in _FORBIDDEN_IMPORT_MODULES:
                    if alias.name == forbidden or alias.name.startswith(f"{forbidden}."):
                        violations.append(f"{rel}:{node.lineno} imports {alias.name}")
    return violations


def test_supervisor_production_import_boundaries() -> None:
    violations: list[str] = []
    for path in _SUPERVISOR_MODULE_PATHS:
        violations.extend(_collect_import_boundary_violations(path))
    assert violations == []
