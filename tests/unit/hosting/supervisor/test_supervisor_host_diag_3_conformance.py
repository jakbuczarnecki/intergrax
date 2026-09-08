# © Artur Czarnecki. All rights reserved.

"""DG-001D R3 — supervisor pre-engine failure HOST-DIAG-3 conformance tests."""

from __future__ import annotations

import json
from collections.abc import Awaitable
from dataclasses import dataclass, field
import pytest

from intergrax.applications._shared.hosted_application_diagnostic_wiring import (
    HostedApplicationDiagnosticEventPublisher,
    HostedDiagnosticTenantBinding,
)
from intergrax.applications._shared.hosted_application_failure_projection import (
    hosted_application_failure_to_problem_signal,
)
from intergrax.hosting import HostedApplicationProfile, RestartPolicy, resolve_hosted_application_definition
from intergrax.hosting.contracts.events import HostedApplicationEvent, HostedApplicationEventType
from intergrax.hosting.control import HostedApplicationControlCoordinator
from intergrax.hosting.engine.definition import HostedApplicationDefinition
from intergrax.hosting.engine.engine import HostedApplicationEngine
from intergrax.hosting.errors import (
    HostedApplicationSupervisorError,
    HostedApplicationSupervisorFailureReason,
)
from intergrax.hosting.eventing import ObservabilityHostedApplicationEventPublisher
from intergrax.hosting.process_bootstrap import HostedProcessBootstrapPhase
from intergrax.hosting.supervisor.classification import HostedApplicationExitKind
from intergrax.hosting.supervisor.failure_projection import HOSTED_APPLICATION_SUPERVISOR_PROCESS_ROLE
from intergrax.hosting.supervisor.supervisor import (
    HostedApplicationEngineFactory,
    HostedApplicationSupervisor,
    HostedApplicationSupervisorLaunchContext,
)
from intergrax.runtime.diagnostics.diagnostic_orchestrator import DiagnosticOrchestrator
from intergrax.runtime.diagnostics.diagnostic_read_service import DiagnosticReadService
from intergrax.runtime.diagnostics.diagnostic_subject import DiagnosticSubjectKind
from intergrax.runtime.diagnostics.problem_occurrence_persistence import (
    ProblemOccurrencePersistence,
)
from intergrax.runtime.diagnostics.in_memory_problem_persistence import (
    InMemoryProblemPersistence,
)
from intergrax.runtime.observability.export_boundary import ExportRecordKind, InMemoryObservabilityExporter
from intergrax.runtime.observability.export_policy import ObservabilityExportPolicy
from tests.unit.applications._shared.test_hosted_application_diagnostic_integration import (
    _build_diagnostic_test_stack,
    _failing_orchestrator_from_stack,
    _ordering_orchestrator_from_stack,
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
from tests.unit.runtime.diagnostics.problem_persistence_test_support import (
    query_all_occurrences_for_problem,
)

pytestmark = pytest.mark.unit

_TENANT_ID = "dg001d-r3-test"
_APP_ID = "dg001d_r3_test_app"
_SECRET_SENTINEL = "DG001D-R3-SECRET-SENTINEL"
_PUBLISHER_SENTINEL = "DG001D-R3-PUBLISHER-SENTINEL"


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


class _FailingOuterPublisher(HostedApplicationDiagnosticEventPublisher):
    async def publish(self, event: HostedApplicationEvent) -> None:
        if event.event_type is HostedApplicationEventType.APPLICATION_FAILED:
            raise RuntimeError(_PUBLISHER_SENTINEL)
        await super().publish(event)


@dataclass
class _SupervisorDiagnosticHarness:
    orchestrator: DiagnosticOrchestrator
    read_service: DiagnosticReadService
    problem_persistence: InMemoryProblemPersistence
    occurrence_persistence: ProblemOccurrencePersistence
    exporter: InMemoryObservabilityExporter
    tenant_binding: HostedDiagnosticTenantBinding
    publisher: HostedApplicationDiagnosticEventPublisher
    published_events: list[HostedApplicationEvent] = field(default_factory=list)


def _build_supervisor_diagnostic_harness(
    *,
    orchestrator: DiagnosticOrchestrator | None = None,
    exporter: InMemoryObservabilityExporter | None = None,
    publisher: HostedApplicationDiagnosticEventPublisher | None = None,
) -> _SupervisorDiagnosticHarness:
    stack = _build_diagnostic_test_stack()
    resolved_orchestrator = orchestrator or stack.orchestrator
    resolved_exporter = exporter or InMemoryObservabilityExporter()
    tenant_binding = HostedDiagnosticTenantBinding(tenant_id=_TENANT_ID)
    published_events: list[HostedApplicationEvent] = []

    if publisher is None:
        class _RecordingDiagnosticPublisher(HostedApplicationDiagnosticEventPublisher):
            async def publish(self, event: HostedApplicationEvent) -> None:
                published_events.append(event)
                await super().publish(event)

        publisher = _RecordingDiagnosticPublisher(
            observability_publisher=ObservabilityHostedApplicationEventPublisher(
                resolved_exporter,
                policy=ObservabilityExportPolicy(enabled=True),
            ),
            tenant_binding=tenant_binding,
            orchestrator=resolved_orchestrator,
        )
    return _SupervisorDiagnosticHarness(
        orchestrator=resolved_orchestrator,
        read_service=stack.read_service,
        problem_persistence=stack.problem_persistence,
        occurrence_persistence=stack.occurrence_persistence,
        exporter=resolved_exporter,
        tenant_binding=tenant_binding,
        publisher=publisher,
        published_events=published_events,
    )


def _definition() -> HostedApplicationDefinition:
    profile = HostedApplicationProfile(
        application_id=_APP_ID,
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
    harness: _SupervisorDiagnosticHarness,
    control: HostedApplicationControlCoordinator | None = None,
    instance_ids: list[str] | None = None,
) -> tuple[HostedApplicationSupervisor, HostedApplicationControlCoordinator]:
    control = control or HostedApplicationControlCoordinator(clock=FixedClock())
    supervisor = HostedApplicationSupervisor(
        definition=_definition(),
        engine_factory=factory,
        control=control,
        event_publisher=harness.publisher,
        clock=FixedClock(),
        sleeper=_ImmediateSleeper(),
        random_source=_FixedRandom(),
        instance_id_generator=_SequenceInstanceIds(instance_ids or ["instance-001"]),
    )
    return supervisor, control


def _failed_events(harness: _SupervisorDiagnosticHarness) -> list[HostedApplicationEvent]:
    return [
        event
        for event in harness.published_events
        if event.event_type is HostedApplicationEventType.APPLICATION_FAILED
    ]


def _occurrences_for_single_problem(harness: _SupervisorDiagnosticHarness):
    problems = harness.read_service.list_problems(tenant_id=_TENANT_ID)
    assert problems.total_count == 1
    return query_all_occurrences_for_problem(
        harness.occurrence_persistence,
        tenant_id=_TENANT_ID,
        problem_id=problems.problems[0].problem_id,
    )


def _assert_identity_fidelity(harness: _SupervisorDiagnosticHarness) -> None:
    event = _failed_events(harness)[0]
    occurrences = _occurrences_for_single_problem(harness)
    assert len(occurrences) == 1
    app_ref = occurrences[0].subject_ref.application_instance()
    assert app_ref is not None
    assert event.application_id == app_ref.application_id
    assert event.instance_id == app_ref.instance_id
    assert app_ref.tenant_id == harness.tenant_binding.tenant_id


def _assert_sentinel_absent(harness: _SupervisorDiagnosticHarness) -> None:
    event = _failed_events(harness)[0]
    signal = hosted_application_failure_to_problem_signal(event)
    assert signal is not None
    serialized_parts = [
        json.dumps(event.payload),
        signal.model_dump_json(),
        repr(harness.read_service.list_problems(tenant_id=_TENANT_ID).problems[0]),
    ]
    occurrences = _occurrences_for_single_problem(harness)
    serialized_parts.append(repr(occurrences[0]))
    for part in serialized_parts:
        assert _SECRET_SENTINEL not in part


@pytest.mark.asyncio
async def test_factory_failure_projects_to_problem_via_real_supervisor() -> None:
    harness = _build_supervisor_diagnostic_harness()

    def factory(launch: HostedApplicationSupervisorLaunchContext) -> HostedApplicationEngine:
        raise RuntimeError(_SECRET_SENTINEL)

    supervisor, _ = _supervisor(factory=factory, harness=harness)
    result = await supervisor.run()

    events = _failed_events(harness)
    assert len(events) == 1
    event = events[0]
    assert event.application_id == _APP_ID
    assert event.instance_id == "instance-001"
    assert event.payload.get("phase") == HostedProcessBootstrapPhase.ENGINE_CONSTRUCTION.value
    assert event.payload.get("reason_code") == (
        HostedApplicationSupervisorFailureReason.ENGINE_FACTORY_FAILED.value
    )
    assert event.payload.get("exception_type") == HostedApplicationSupervisorError.__name__
    assert event.payload.get("process_role") == HOSTED_APPLICATION_SUPERVISOR_PROCESS_ROLE

    problems = harness.read_service.list_problems(tenant_id=_TENANT_ID)
    assert problems.total_count == 1
    assert problems.problems[0].occurrence_count == 1
    assert result.final_exit.exit_kind is HostedApplicationExitKind.SUPERVISOR_ERROR


@pytest.mark.asyncio
async def test_contract_validation_failure_projects_to_problem() -> None:
    harness = _build_supervisor_diagnostic_harness()

    def factory(launch: HostedApplicationSupervisorLaunchContext) -> HostedApplicationEngine:
        return _build_engine(launch, instance_id="wrong-instance-id")

    supervisor, _ = _supervisor(factory=factory, harness=harness)
    await supervisor.run()

    event = _failed_events(harness)[0]
    assert event.payload.get("phase") == HostedProcessBootstrapPhase.ENGINE_CONTRACT_VALIDATION.value
    assert event.payload.get("reason_code") == (
        HostedApplicationSupervisorFailureReason.ENGINE_INSTANCE_ID_MISMATCH.value
    )
    assert harness.read_service.list_problems(tenant_id=_TENANT_ID).total_count == 1


@pytest.mark.asyncio
async def test_application_instance_subject_without_execution_identity() -> None:
    harness = _build_supervisor_diagnostic_harness()

    def factory(launch: HostedApplicationSupervisorLaunchContext) -> HostedApplicationEngine:
        raise RuntimeError(_SECRET_SENTINEL)

    supervisor, _ = _supervisor(factory=factory, harness=harness)
    await supervisor.run()

    occurrences = _occurrences_for_single_problem(harness)
    app_ref = occurrences[0].subject_ref.application_instance()
    assert app_ref is not None
    assert app_ref.kind is DiagnosticSubjectKind.APPLICATION_INSTANCE
    assert occurrences[0].subject_ref.execution() is None
    signal = hosted_application_failure_to_problem_signal(_failed_events(harness)[0])
    assert signal is not None
    assert signal.task_id == ""
    assert signal.run_id == ""


@pytest.mark.asyncio
async def test_identity_fidelity_between_event_and_problem() -> None:
    harness = _build_supervisor_diagnostic_harness()

    def factory(launch: HostedApplicationSupervisorLaunchContext) -> HostedApplicationEngine:
        raise RuntimeError(_SECRET_SENTINEL)

    supervisor, _ = _supervisor(factory=factory, harness=harness)
    await supervisor.run()
    _assert_identity_fidelity(harness)


@pytest.mark.asyncio
async def test_raw_failure_secret_not_persisted() -> None:
    harness = _build_supervisor_diagnostic_harness()

    def factory(launch: HostedApplicationSupervisorLaunchContext) -> HostedApplicationEngine:
        raise RuntimeError(_SECRET_SENTINEL)

    supervisor, _ = _supervisor(factory=factory, harness=harness)
    await supervisor.run()
    _assert_sentinel_absent(harness)


@pytest.mark.asyncio
async def test_exactly_one_failure_event_and_problem_per_attempt() -> None:
    harness = _build_supervisor_diagnostic_harness()

    def factory(launch: HostedApplicationSupervisorLaunchContext) -> HostedApplicationEngine:
        raise RuntimeError(_SECRET_SENTINEL)

    supervisor, _ = _supervisor(factory=factory, harness=harness)
    await supervisor.run()

    assert len(_failed_events(harness)) == 1
    problems = harness.read_service.list_problems(tenant_id=_TENANT_ID)
    assert problems.total_count == 1
    assert problems.problems[0].occurrence_count == 1


@pytest.mark.asyncio
async def test_observability_export_before_diagnostics() -> None:
    exporter = InMemoryObservabilityExporter()
    observability_count_at_diagnostic: list[int] = []
    stack = _build_diagnostic_test_stack()
    ordering_orchestrator = _ordering_orchestrator_from_stack(
        stack,
        envelope_count=lambda: len(exporter.envelopes),
        recorded=observability_count_at_diagnostic,
    )
    harness = _build_supervisor_diagnostic_harness(
        orchestrator=ordering_orchestrator,
        exporter=exporter,
    )

    def factory(launch: HostedApplicationSupervisorLaunchContext) -> HostedApplicationEngine:
        raise RuntimeError(_SECRET_SENTINEL)

    supervisor, _ = _supervisor(factory=factory, harness=harness)
    await supervisor.run()

    assert observability_count_at_diagnostic == [len(exporter.envelopes)]
    assert observability_count_at_diagnostic[0] >= 1
    platform_exports = [
        envelope
        for envelope in exporter.envelopes
        if envelope.record_kind is ExportRecordKind.PLATFORM_SIGNAL
    ]
    assert len(platform_exports) >= 1


@pytest.mark.asyncio
async def test_diagnostic_projection_failure_isolated() -> None:
    stack = _build_diagnostic_test_stack()
    harness = _build_supervisor_diagnostic_harness(
        orchestrator=_failing_orchestrator_from_stack(stack),
    )

    def factory(launch: HostedApplicationSupervisorLaunchContext) -> HostedApplicationEngine:
        raise RuntimeError(_SECRET_SENTINEL)

    supervisor, _ = _supervisor(factory=factory, harness=harness)
    result = await supervisor.run()

    assert result.final_exit.exit_kind is HostedApplicationExitKind.SUPERVISOR_ERROR
    assert harness.read_service.list_problems(tenant_id=_TENANT_ID).total_count == 0
    platform_exports = [
        envelope
        for envelope in harness.exporter.envelopes
        if envelope.record_kind is ExportRecordKind.PLATFORM_SIGNAL
    ]
    assert len(platform_exports) >= 1


@pytest.mark.asyncio
async def test_supervisor_truth_preserved_when_diagnostics_fail() -> None:
    def factory(launch: HostedApplicationSupervisorLaunchContext) -> HostedApplicationEngine:
        raise RuntimeError(_SECRET_SENTINEL)

    baseline_harness = _build_supervisor_diagnostic_harness()
    baseline_supervisor, _ = _supervisor(factory=factory, harness=baseline_harness)
    baseline_result = await baseline_supervisor.run()

    stack = _build_diagnostic_test_stack()
    failing_harness = _build_supervisor_diagnostic_harness(
        orchestrator=_failing_orchestrator_from_stack(stack),
    )
    failing_supervisor, _ = _supervisor(factory=factory, harness=failing_harness)
    failing_result = await failing_supervisor.run()

    assert failing_result.final_exit.exit_kind == baseline_result.final_exit.exit_kind
    assert failing_result.final_exit.retryable == baseline_result.final_exit.retryable
    assert failing_result.final_exit.reason_code == baseline_result.final_exit.reason_code
    assert failing_result.restart_exhausted == baseline_result.restart_exhausted


@pytest.mark.asyncio
async def test_success_path_creates_no_pre_engine_problem() -> None:
    harness = _build_supervisor_diagnostic_harness()

    def factory(launch: HostedApplicationSupervisorLaunchContext) -> Awaitable[HostedApplicationEngine]:
        async def _build() -> HostedApplicationEngine:
            engine = _build_engine(launch)
            await engine.start()
            launch.control.request_shutdown("test.complete")
            return engine

        return _build()

    supervisor, _ = _supervisor(factory=factory, harness=harness)
    result = await supervisor.run()

    assert _failed_events(harness) == []
    assert harness.read_service.list_problems(tenant_id=_TENANT_ID).total_count == 0
    assert result.final_exit.exit_kind is HostedApplicationExitKind.CLEAN_STOP


@pytest.mark.asyncio
async def test_runtime_engine_failure_no_supervisor_pre_engine_duplicate() -> None:
    from tests.unit.hosting.engine import _fakes as fakes_module

    harness = _build_supervisor_diagnostic_harness()

    def factory(launch: HostedApplicationSupervisorLaunchContext) -> Awaitable[HostedApplicationEngine]:
        async def _build() -> HostedApplicationEngine:
            fakes_module._RUNTIME_HOLDER["runtime"] = FakeRuntime(fail_start=True)
            return _build_engine(launch)

        return _build()

    supervisor, _ = _supervisor(factory=factory, harness=harness)
    result = await supervisor.run()

    assert _failed_events(harness) == []
    assert result.final_exit.exit_kind is HostedApplicationExitKind.STARTUP_FAILURE


@pytest.mark.asyncio
async def test_stop_before_launch_emits_no_failure_or_problem() -> None:
    harness = _build_supervisor_diagnostic_harness()

    def factory(launch: HostedApplicationSupervisorLaunchContext) -> HostedApplicationEngine:
        return _build_engine(launch)

    control = HostedApplicationControlCoordinator(clock=FixedClock())
    control.request_shutdown("test.stop")
    supervisor, _ = _supervisor(factory=factory, harness=harness, control=control)
    result = await supervisor.run()

    assert _failed_events(harness) == []
    assert harness.read_service.list_problems(tenant_id=_TENANT_ID).total_count == 0
    assert result.final_exit.reason_code == "stop_before_launch"


@pytest.mark.asyncio
async def test_recurrence_groups_across_two_supervisor_instance_ids() -> None:
    harness = _build_supervisor_diagnostic_harness()

    def factory(launch: HostedApplicationSupervisorLaunchContext) -> HostedApplicationEngine:
        raise RuntimeError(_SECRET_SENTINEL)

    for instance_id in ("instance-001", "instance-002"):
        supervisor, _ = _supervisor(
            factory=factory,
            harness=harness,
            instance_ids=[instance_id],
        )
        await supervisor.run()

    problems = harness.read_service.list_problems(tenant_id=_TENANT_ID)
    assert problems.total_count == 1
    assert problems.problems[0].occurrence_count == 2
    occurrences = _occurrences_for_single_problem(harness)
    instance_ids = set()
    for occurrence in occurrences:
        app_ref = occurrence.subject_ref.application_instance()
        assert app_ref is not None
        instance_ids.add(app_ref.instance_id)
    assert instance_ids == {"instance-001", "instance-002"}


@pytest.mark.asyncio
async def test_grouping_does_not_depend_on_instance_id() -> None:
    harness = _build_supervisor_diagnostic_harness()

    def factory(launch: HostedApplicationSupervisorLaunchContext) -> HostedApplicationEngine:
        raise RuntimeError(_SECRET_SENTINEL)

    for instance_id in ("alpha-instance", "beta-instance"):
        supervisor, _ = _supervisor(
            factory=factory,
            harness=harness,
            instance_ids=[instance_id],
        )
        await supervisor.run()

    assert harness.read_service.list_problems(tenant_id=_TENANT_ID).total_count == 1


@pytest.mark.asyncio
async def test_composed_publisher_outer_failure_isolated() -> None:
    stack = _build_diagnostic_test_stack()
    exporter = InMemoryObservabilityExporter()
    tenant_binding = HostedDiagnosticTenantBinding(tenant_id=_TENANT_ID)
    failing_publisher = _FailingOuterPublisher(
        observability_publisher=ObservabilityHostedApplicationEventPublisher(
            exporter,
            policy=ObservabilityExportPolicy(enabled=True),
        ),
        tenant_binding=tenant_binding,
        orchestrator=stack.orchestrator,
    )
    harness = _build_supervisor_diagnostic_harness(publisher=failing_publisher)

    def factory(launch: HostedApplicationSupervisorLaunchContext) -> HostedApplicationEngine:
        raise RuntimeError(_SECRET_SENTINEL)

    baseline_supervisor, _ = _supervisor(
        factory=factory,
        harness=_build_supervisor_diagnostic_harness(),
    )
    baseline_result = await baseline_supervisor.run()

    failing_supervisor, _ = _supervisor(factory=factory, harness=harness)
    failing_result = await failing_supervisor.run()

    assert failing_result.final_exit.exit_kind == baseline_result.final_exit.exit_kind
    assert failing_result.final_exit.reason_code == baseline_result.final_exit.reason_code
    assert harness.read_service.list_problems(tenant_id=_TENANT_ID).total_count == 0


@pytest.mark.asyncio
async def test_diagnostic_read_service_get_problem() -> None:
    harness = _build_supervisor_diagnostic_harness()

    def factory(launch: HostedApplicationSupervisorLaunchContext) -> HostedApplicationEngine:
        raise RuntimeError(_SECRET_SENTINEL)

    supervisor, _ = _supervisor(factory=factory, harness=harness)
    await supervisor.run()

    listed = harness.read_service.list_problems(tenant_id=_TENANT_ID)
    problem_id = listed.problems[0].problem_id
    fetched = harness.read_service.get_problem(
        tenant_id=_TENANT_ID,
        problem_id=problem_id,
    )
    assert fetched is not None
    assert fetched.problem_id == problem_id
    assert fetched.occurrence_count == 1
