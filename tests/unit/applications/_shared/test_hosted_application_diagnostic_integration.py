# © Artur Czarnecki. All rights reserved.

"""HOST-DIAG-3 — HostedApplication failure → central diagnostic integration tests."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from intergrax.applications._shared.hosted_application_diagnostic_wiring import (
    HostedApplicationDiagnosticEventPublisher,
    HostedDiagnosticTenantBinding,
    build_hosted_application_diagnostic_event_publisher,
)
from intergrax.applications._shared.hosted_application_failure_projection import (
    hosted_application_failure_to_problem_signal,
)
from intergrax.hosting import (
    HostedApplicationEventType,
    HostedApplicationLifecycleState,
    HostedApplicationProfile,
    InstancePolicy,
    RestartPolicy,
    resolve_hosted_application_definition,
)
from intergrax.hosting.contracts.context import (
    HostedApplicationContext,
    HostedApplicationEventPublisher,
    HostedApplicationPaths,
)
from intergrax.hosting.contracts.events import HostedApplicationEvent
from intergrax.hosting.contracts.policies import InstanceExclusivityMode, RestartMode
from intergrax.hosting.engine.ports import HostedApplicationRuntime
from intergrax.hosting.eventing import ObservabilityHostedApplicationEventPublisher
from intergrax.hosting.runner import _RunnerFactories, _run_resolved_hosted_application
from intergrax.runtime.diagnostics.deterministic_problem_grouping import (
    STRATEGY_ID,
    DeterministicProblemGroupingStrategy,
)
from intergrax.runtime.diagnostics.diagnostic_assessment import DiagnosticAssessmentBuilder
from intergrax.runtime.diagnostics.diagnostic_orchestration_models import (
    DiagnosticOrchestrationRequest,
    DiagnosticSignalSubjectScope,
)
from intergrax.runtime.diagnostics.diagnostic_orchestrator import DiagnosticOrchestrator
from intergrax.runtime.diagnostics.diagnostic_read_service import DiagnosticReadService
from intergrax.runtime.diagnostics.execution_reconstruction import ExecutionReconstructor
from intergrax.runtime.diagnostics.in_memory_problem_persistence import InMemoryProblemPersistence
from intergrax.runtime.diagnostics.lifecycle_analysis import LifecycleAnomalyAnalyzer
from intergrax.runtime.diagnostics.problem_grouping import (
    ProblemGroupingEngine,
    ProblemGroupingStrategyRegistry,
)
from intergrax.runtime.diagnostics.problem_lifecycle import ProblemLifecycleEngine
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.observability.export_boundary import (
    ExportRecordKind,
    InMemoryObservabilityExporter,
)
from intergrax.runtime.observability.export_policy import ObservabilityExportPolicy
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from tests.unit.runtime.diagnostics.problem_persistence_test_support import (
    document_store_occurrence_persistence_for_tests,
    in_memory_document_store_for_problem_tests,
    lifecycle_engine_for_tests,
    query_all_occurrences_for_problem,
    query_all_problems_for_tenant,
    read_service_for_tests,
)
from intergrax.runtime.observability.problem_signal import (
    PROBLEM_KIND_PLATFORM_APPLICATION_FAILURE,
)
from tests.unit.hosting.engine._fakes import (
    FakeInstanceGuard,
    FakeRuntime,
    FixedClock,
    NoopLogger,
    build_engine_paths,
    build_process_identity,
)

pytestmark = pytest.mark.unit

_TENANT_A = "tenant-a"
_TENANT_B = "tenant-b"
_APP_ID = "host_diag_test_app"
_OBSERVED_AT = datetime(2026, 8, 26, 10, 0, tzinfo=UTC)


class _ShutdownOnStartRuntime:
    async def start(self, context: HostedApplicationContext) -> None:
        context.shutdown.request_shutdown("test.complete")

    async def stop(self, context: HostedApplicationContext) -> None:
        return None

    async def ready(self, context: HostedApplicationContext) -> bool:
        return True


def _profile_with_runtime(
    runtime_factory: Callable[[], HostedApplicationRuntime],
    *,
    restart: RestartPolicy | None = None,
) -> HostedApplicationProfile:
    return HostedApplicationProfile(
        application_id=_APP_ID,
        application_factory=runtime_factory,
        application_factory_id="tests.hosted_application_diagnostic_integration",
        restart=restart or RestartPolicy.never(),
        instance=InstancePolicy(exclusivity_mode=InstanceExclusivityMode.MULTI_INSTANCE),
    )


def _fast_restart_policy(*, max_attempts: int = 2) -> RestartPolicy:
    return RestartPolicy(
        mode=RestartMode.ON_FAILURE,
        max_attempts=max_attempts,
        initial_backoff_seconds=0.001,
        max_backoff_seconds=0.001,
        jitter_ratio=0.0,
    )


class _SequenceInstanceIds:
    def __init__(self, instance_ids: list[str]) -> None:
        self._instance_ids = list(instance_ids)
        self._index = 0
        self._extra = 0

    def __call__(self) -> str:
        if self._index < len(self._instance_ids):
            instance_id = self._instance_ids[self._index]
            self._index += 1
            return instance_id
        self._extra += 1
        return f"instance-extra-{self._extra:04d}"


def _build_orchestrator_stack() -> tuple[
    DiagnosticOrchestrator,
    InMemoryProblemPersistence,
    DiagnosticReadService,
    object,
]:
    from tests.unit.runtime.diagnostics.problem_persistence_test_support import (
        build_diagnostic_orchestrator_stack_for_tests,
    )

    return build_diagnostic_orchestrator_stack_for_tests()


@dataclass
class _HostedHarness:
    orchestrator: DiagnosticOrchestrator
    persistence: InMemoryProblemPersistence
    read_service: DiagnosticReadService
    exporter: InMemoryObservabilityExporter
    tenant_binding: HostedDiagnosticTenantBinding
    published_events: list[HostedApplicationEvent] = field(default_factory=list)
    custom_publisher_factory: Callable[[], HostedApplicationEventPublisher] | None = None

    def event_publisher_factory(
        self,
    ) -> Callable[[], HostedApplicationEventPublisher]:
        if self.custom_publisher_factory is not None:
            return self.custom_publisher_factory
        harness = self

        class _RecordingDiagnosticPublisher(HostedApplicationDiagnosticEventPublisher):
            async def publish(self, event: HostedApplicationEvent) -> None:
                harness.published_events.append(event)
                await super().publish(event)

        def factory() -> HostedApplicationEventPublisher:
            observability = ObservabilityHostedApplicationEventPublisher(
                harness.exporter,
                policy=ObservabilityExportPolicy(enabled=True),
            )
            return _RecordingDiagnosticPublisher(
                observability_publisher=observability,
                tenant_binding=harness.tenant_binding,
                orchestrator=harness.orchestrator,
            )

        return factory


def _runner_factories(
    tmp_path: Path,
    harness: _HostedHarness,
    *,
    instance_id_generator: _SequenceInstanceIds | None = None,
) -> _RunnerFactories:
    shared_clock = FixedClock(_OBSERVED_AT)
    return _RunnerFactories(
        create_paths=lambda definition: HostedApplicationPaths(
            data_home=(tmp_path / "data" / definition.application_id).resolve(),
            run_directory=(tmp_path / "run").resolve(),
        ),
        create_clock=lambda: shared_clock,
        create_monotonic_clock=lambda: __import__(
            "intergrax.hosting.shutdown",
            fromlist=["SystemMonotonicClock"],
        ).SystemMonotonicClock(),
        create_logger=lambda _application_id: NoopLogger(),
        create_event_publisher=harness.event_publisher_factory(),
        create_process_identity=lambda clock: build_process_identity(clock),
        create_instance_guard=lambda definition, paths, process_identity, clock: FakeInstanceGuard(),
        create_signal_adapter=lambda control: type(
            "SignalAdapter",
            (),
            {"install": lambda self: None, "restore": lambda self: None},
        )(),
        instance_id_generator=instance_id_generator,
    )


async def _run_failure_profile(
    tmp_path: Path,
    harness: _HostedHarness,
    profile: HostedApplicationProfile,
    *,
    instance_id_generator: _SequenceInstanceIds | None = None,
) -> None:
    definition = resolve_hosted_application_definition(profile)
    factories = _runner_factories(
        tmp_path,
        harness,
        instance_id_generator=instance_id_generator,
    )
    await _run_resolved_hosted_application(definition, factories)


def test_projector_maps_bounded_failure_facts() -> None:
    event = HostedApplicationEvent(
        event_type=HostedApplicationEventType.APPLICATION_FAILED,
        occurred_at=_OBSERVED_AT,
        application_id=_APP_ID,
        instance_id="instance-i1",
        lifecycle_state=HostedApplicationLifecycleState.FAILED,
        payload={
            "failure_id": "failure-0001",
            "reason_code": "runtime_start_failed",
            "phase": "runtime_start",
            "source_kind": "runtime",
            "source_id": "start",
            "exception_type": "RuntimeError",
        },
    )
    signal = hosted_application_failure_to_problem_signal(event)
    assert signal is not None
    assert signal.problem_kind == PROBLEM_KIND_PLATFORM_APPLICATION_FAILURE
    assert signal.source_component == "runtime_start"
    assert signal.error_code == "runtime_start_failed"
    assert signal.task_id == ""
    assert signal.run_id == ""


@pytest.mark.asyncio
async def test_application_failure_creates_problem(tmp_path: Path) -> None:
    orchestrator, persistence, read_service, occurrence_persistence = _build_orchestrator_stack()
    harness = _HostedHarness(
        orchestrator=orchestrator,
        persistence=persistence,
        read_service=read_service,
        exporter=InMemoryObservabilityExporter(),
        tenant_binding=HostedDiagnosticTenantBinding(tenant_id=_TENANT_A),
    )
    profile = _profile_with_runtime(lambda: FakeRuntime(fail_start=True))  # type: ignore[return-value]
    instance_ids = _SequenceInstanceIds(["instance-i1"])

    await _run_failure_profile(
        tmp_path,
        harness,
        profile,
        instance_id_generator=instance_ids,
    )

    failed_events = [
        event
        for event in harness.published_events
        if event.event_type is HostedApplicationEventType.APPLICATION_FAILED
    ]
    assert len(failed_events) == 1
    assert failed_events[0].instance_id == "instance-i1"
    assert failed_events[0].payload.get("phase") == "runtime_start"

    platform_exports = [
        envelope
        for envelope in harness.exporter.envelopes
        if envelope.record_kind is ExportRecordKind.PLATFORM_SIGNAL
    ]
    assert len(platform_exports) >= 1

    problems = read_service.list_problems(tenant_id=_TENANT_A)
    assert problems.total_count == 1
    problem = problems.problems[0]
    assert problem.occurrence_count == 1
    stored = query_all_problems_for_tenant(persistence, _TENANT_A)[0]
    occurrences = query_all_occurrences_for_problem(
        occurrence_persistence,
        tenant_id=_TENANT_A,
        problem_id=stored.problem_id,
    )
    app_ref = occurrences[0].subject_ref.application_instance()
    assert app_ref is not None
    assert app_ref.application_id == _APP_ID
    assert app_ref.instance_id == "instance-i1"
    assert app_ref.tenant_id == _TENANT_A


@pytest.mark.asyncio
async def test_observability_export_before_diagnostics(tmp_path: Path) -> None:
    exporter = InMemoryObservabilityExporter()
    observability_count_at_diagnostic: list[int] = []
    orchestrator, persistence, read_service, _ = _build_orchestrator_stack()

    class _RecordingDiagnosticPublisher(HostedApplicationDiagnosticEventPublisher):
        async def publish(self, event: HostedApplicationEvent) -> None:
            await self._observability_publisher.publish(event)
            if event.event_type is HostedApplicationEventType.APPLICATION_FAILED:
                observability_count_at_diagnostic.append(len(exporter.envelopes))
            if event.event_type is not HostedApplicationEventType.APPLICATION_FAILED:
                return
            signal = hosted_application_failure_to_problem_signal(event)
            if signal is None:
                return
            scope = DiagnosticSignalSubjectScope(
                tenant_id=_TENANT_A,
                application_id=event.application_id,
                instance_id=event.instance_id,
                problem_signals=(signal,),
            )
            request = DiagnosticOrchestrationRequest(
                tenant_id=_TENANT_A,
                grouping_strategy_id=STRATEGY_ID,
                observed_at=event.occurred_at,
                signal_subjects=(scope,),
            )
            orchestrator.run(request)

    harness = _HostedHarness(
        orchestrator=orchestrator,
        persistence=persistence,
        read_service=read_service,
        exporter=exporter,
        tenant_binding=HostedDiagnosticTenantBinding(tenant_id=_TENANT_A),
    )
    harness.custom_publisher_factory = lambda: _RecordingDiagnosticPublisher(
        observability_publisher=ObservabilityHostedApplicationEventPublisher(
            exporter,
            policy=ObservabilityExportPolicy(enabled=True),
        ),
        tenant_binding=harness.tenant_binding,
        orchestrator=orchestrator,
    )

    profile = _profile_with_runtime(lambda: FakeRuntime(fail_start=True))  # type: ignore[return-value]
    instance_ids = _SequenceInstanceIds(["instance-i1"])

    await _run_failure_profile(
        tmp_path,
        harness,
        profile,
        instance_id_generator=instance_ids,
    )

    assert observability_count_at_diagnostic == [len(exporter.envelopes)]
    assert observability_count_at_diagnostic[0] >= 1


@pytest.mark.asyncio
async def test_recurrence_across_instances_and_replay(tmp_path: Path) -> None:
    orchestrator, persistence, read_service, _ = _build_orchestrator_stack()
    harness = _HostedHarness(
        orchestrator=orchestrator,
        persistence=persistence,
        read_service=read_service,
        exporter=InMemoryObservabilityExporter(),
        tenant_binding=HostedDiagnosticTenantBinding(tenant_id=_TENANT_A),
    )
    profile = _profile_with_runtime(
        lambda: FakeRuntime(fail_start=True),  # type: ignore[return-value]
        restart=_fast_restart_policy(max_attempts=1),
    )
    instance_ids = _SequenceInstanceIds(["instance-i1", "instance-i2"])
    await _run_failure_profile(
        tmp_path,
        harness,
        profile,
        instance_id_generator=instance_ids,
    )

    problems = read_service.list_problems(tenant_id=_TENANT_A)
    assert problems.total_count == 1
    problem = problems.problems[0]
    assert problem.occurrence_count == 2

    replay_scope = DiagnosticSignalSubjectScope(
        tenant_id=_TENANT_A,
        application_id=_APP_ID,
        instance_id="instance-i1",
        problem_signals=(
            hosted_application_failure_to_problem_signal(
                HostedApplicationEvent(
                    event_type=HostedApplicationEventType.APPLICATION_FAILED,
                    occurred_at=_OBSERVED_AT + timedelta(hours=2),
                    application_id=_APP_ID,
                    instance_id="instance-i1",
                    lifecycle_state=HostedApplicationLifecycleState.FAILED,
                    payload={
                        "failure_id": "failure-replay",
                        "reason_code": "runtime_start_failed",
                        "phase": "runtime_start",
                        "source_kind": "runtime",
                        "source_id": "start",
                        "exception_type": "RuntimeError",
                    },
                ),
            ),
        ),
    )
    assert replay_scope.problem_signals[0] is not None
    orchestrator.run(
        DiagnosticOrchestrationRequest(
            tenant_id=_TENANT_A,
            grouping_strategy_id=STRATEGY_ID,
            observed_at=_OBSERVED_AT + timedelta(hours=2),
            signal_subjects=(replay_scope,),
        ),
    )
    replayed = read_service.list_problems(tenant_id=_TENANT_A).problems[0]
    assert replayed.occurrence_count == 2


@pytest.mark.asyncio
async def test_clean_lifecycle_creates_no_problems(tmp_path: Path) -> None:
    orchestrator, persistence, read_service, _ = _build_orchestrator_stack()
    harness = _HostedHarness(
        orchestrator=orchestrator,
        persistence=persistence,
        read_service=read_service,
        exporter=InMemoryObservabilityExporter(),
        tenant_binding=HostedDiagnosticTenantBinding(tenant_id=_TENANT_A),
    )
    profile = _profile_with_runtime(lambda: _ShutdownOnStartRuntime())  # type: ignore[return-value]
    await _run_failure_profile(tmp_path, harness, profile)

    lifecycle_types = {event.event_type for event in harness.published_events}
    assert HostedApplicationEventType.APPLICATION_STARTING in lifecycle_types
    assert HostedApplicationEventType.APPLICATION_STOPPED in lifecycle_types
    assert HostedApplicationEventType.APPLICATION_FAILED not in lifecycle_types
    assert read_service.list_problems(tenant_id=_TENANT_A).total_count == 0
    assert len(harness.exporter.envelopes) >= 1


def _raise_factory_error() -> HostedApplicationRuntime:
    raise RuntimeError("runtime factory failed")


@pytest.mark.asyncio
async def test_different_failure_signature_isolation(tmp_path: Path) -> None:
    orchestrator, persistence, read_service, _ = _build_orchestrator_stack()
    harness = _HostedHarness(
        orchestrator=orchestrator,
        persistence=persistence,
        read_service=read_service,
        exporter=InMemoryObservabilityExporter(),
        tenant_binding=HostedDiagnosticTenantBinding(tenant_id=_TENANT_A),
    )

    profile_start = _profile_with_runtime(lambda: FakeRuntime(fail_start=True))  # type: ignore[return-value]
    await _run_failure_profile(tmp_path, harness, profile_start)

    profile_factory = HostedApplicationProfile(
        application_id=_APP_ID,
        application_factory=lambda _ctx: _raise_factory_error(),
        application_factory_id="tests.hosted_application_diagnostic_integration.factory_fail",
        restart=RestartPolicy.never(),
        instance=InstancePolicy(exclusivity_mode=InstanceExclusivityMode.MULTI_INSTANCE),
    )
    await _run_failure_profile(tmp_path, harness, profile_factory)

    problems = read_service.list_problems(tenant_id=_TENANT_A)
    assert problems.total_count == 2


@pytest.mark.asyncio
async def test_tenant_isolation(tmp_path: Path) -> None:
    persistence = InMemoryProblemPersistence()
    occurrence_store = in_memory_document_store_for_problem_tests()
    occurrence_persistence = document_store_occurrence_persistence_for_tests(occurrence_store)
    registry = ProblemGroupingStrategyRegistry()
    registry.register(DeterministicProblemGroupingStrategy())
    reconstructor = ExecutionReconstructor(
        runtime_events=InMemoryRuntimeEventStore(),
        causal_evidence=InMemoryCausalEvidencePersistence(),
    )

    def _orchestrator() -> DiagnosticOrchestrator:
        return DiagnosticOrchestrator(
            execution_reconstructor=reconstructor,
            lifecycle_analyzer=LifecycleAnomalyAnalyzer(),
            assessment_builder=DiagnosticAssessmentBuilder(),
            grouping_engine=ProblemGroupingEngine(registry),
            problem_lifecycle_engine=lifecycle_engine_for_tests(
                persistence,
                occurrence_persistence,
                document_store=occurrence_store,
            ),
        )

    read_service = read_service_for_tests(
        persistence,
        reconstructor,
        occurrence_persistence=occurrence_persistence,
        document_store=occurrence_store,
    )

    harness_a = _HostedHarness(
        orchestrator=_orchestrator(),
        persistence=persistence,
        read_service=read_service,
        exporter=InMemoryObservabilityExporter(),
        tenant_binding=HostedDiagnosticTenantBinding(tenant_id=_TENANT_A),
    )
    harness_b = _HostedHarness(
        orchestrator=_orchestrator(),
        persistence=persistence,
        read_service=read_service,
        exporter=InMemoryObservabilityExporter(),
        tenant_binding=HostedDiagnosticTenantBinding(tenant_id=_TENANT_B),
    )

    profile = _profile_with_runtime(lambda: FakeRuntime(fail_start=True))  # type: ignore[return-value]
    await _run_failure_profile(tmp_path, harness_a, profile)
    await _run_failure_profile(tmp_path, harness_b, profile)

    problem_a = read_service.list_problems(tenant_id=_TENANT_A).problems[0].problem_id
    problem_b = read_service.list_problems(tenant_id=_TENANT_B).problems[0].problem_id
    assert problem_a != problem_b


def test_build_hosted_application_diagnostic_event_publisher_factory() -> None:
    orchestrator, persistence, _, _ = _build_orchestrator_stack()
    publisher = build_hosted_application_diagnostic_event_publisher(
        tenant_binding=HostedDiagnosticTenantBinding(tenant_id=_TENANT_A),
        orchestrator=orchestrator,
        observability_exporter=InMemoryObservabilityExporter(),
    )
    assert isinstance(publisher, HostedApplicationDiagnosticEventPublisher)
