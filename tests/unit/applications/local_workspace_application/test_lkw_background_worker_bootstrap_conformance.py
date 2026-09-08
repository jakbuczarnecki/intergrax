# © Artur Czarnecki. All rights reserved.

"""DG-001B R4 — LKW background worker bootstrap diagnostic conformance."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from intergrax.applications._shared.diagnostic_read_wiring import (
    build_diagnostic_read_service,
    resolve_host_diagnostic_read_dependencies,
)
from intergrax.applications._shared.hosted_application_diagnostic_wiring import (
    HostedApplicationDiagnosticEventPublisher,
    HostedDiagnosticTenantBinding,
    build_hosted_application_diagnostic_event_publisher,
)
from intergrax.hosting import (
    BOOTSTRAP_UNHANDLED_EXCEPTION_REASON_CODE,
    HostedApplicationEventType,
    HostedApplicationLifecycleState,
    HostedProcessBootstrapContext,
    HostedProcessBootstrapPhase,
)
from intergrax.hosting.contracts.context import HostedApplicationEventPublisher
from intergrax.hosting.contracts.events import HostedApplicationEvent
from intergrax.hosting.eventing import ObservabilityHostedApplicationEventPublisher
from intergrax.runtime.diagnostics.diagnostic_orchestrator import DiagnosticOrchestrator
from intergrax.runtime.diagnostics.diagnostic_orchestration_models import (
    DiagnosticOrchestrationRequest,
    DiagnosticOrchestrationResult,
)
from intergrax.runtime.diagnostics.diagnostic_read_service import DiagnosticReadService
from intergrax.runtime.diagnostics.in_memory_problem_persistence import InMemoryProblemPersistence
from intergrax.runtime.diagnostics.problem_occurrence_persistence import ProblemOccurrencePersistence
from intergrax.runtime.observability.export_boundary import (
    InMemoryObservabilityExporter,
    ObservabilityExportEnvelope,
    ObservabilityExporter,
)
from intergrax.runtime.observability.export_policy import ObservabilityExportPolicy
from local_workspace_application.host.background_worker_factory import (
    LocalWorkspaceBackgroundWorkerWiring,
)
from local_workspace_application.host.background_worker_main import (
    _build_worker_diagnostic_runtime,
    _run_guarded_worker_bootstrap,
    activate_local_workspace_reference_production_authority,
    build_local_workspace_worker_bootstrap_diagnostics,
)
from local_workspace_application.host.environment_profile import (
    build_local_workspace_environment_profile,
)
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.manifest import LOCAL_WORKSPACE_APPLICATION_MANIFEST
from local_workspace_application.workspaces.document_store_factory import (
    resolve_lkw_runtime_document_store,
)
from tests.unit.runtime.diagnostics.problem_persistence_test_support import (
    build_diagnostic_orchestrator_stack_for_tests,
    query_all_problems_for_tenant,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_WORKER_MAIN_PATH = (
    _REPO_ROOT / "applications/local_workspace_application/host/background_worker_main.py"
)
_PROCESS_ROLE = "background_worker"

_FORBIDDEN_IMPORT_SYMBOLS = frozenset(
    {
        "TaskId",
        "RunId",
        "AttemptId",
        "ExecutionId",
        "ProblemLifecycleEngine",
        "DeterministicProblemGroupingStrategy",
        "InMemoryProblemPersistence",
        "NoOpObservabilityExporter",
    },
)
_FORBIDDEN_IMPORT_PREFIXES = (
    "intergrax.integrations.providers.message_bus.kafka",
    "intergrax.runtime.diagnostics.problem_lifecycle",
    "intergrax.runtime.diagnostics.deterministic_problem_grouping",
    "intergrax.runtime.observability.otlp_exporter",
    "intergrax.runtime.observability.sentry_export_wiring",
    "intergrax.runtime.observability.elasticsearch_export_wiring",
)
_FORBIDDEN_SOURCE_TOKENS = (
    "OtlpObservabilityExporter",
    "build_otlp_observability_exporter",
    "sentry_sdk",
)


def _settings(monkeypatch: pytest.MonkeyPatch) -> LocalWorkspaceBackendSettings:
    monkeypatch.setenv("INTERGRAX_HARNESS_API_KEY", "test-lkw-worker-bootstrap-r4-key")
    monkeypatch.setenv("LOCAL_WORKSPACE_ENABLE_MESSAGE_BUS", "true")
    monkeypatch.setenv("LOCAL_WORKSPACE_ENABLE_KAFKA_MESSAGE_BUS", "true")
    monkeypatch.setenv("LOCAL_WORKSPACE_ENABLE_REDIS", "true")
    monkeypatch.setenv(
        "INTERGRAX_PROBLEM_LIST_CURSOR_SECRET",
        "unit-test-lkw-worker-bootstrap-r4-cursor-secret",
    )
    return LocalWorkspaceBackendSettings.from_env()


@dataclass
class _WorkerBootstrapDiagnosticHarness:
    orchestrator: DiagnosticOrchestrator
    persistence: InMemoryProblemPersistence
    read_service: DiagnosticReadService
    occurrence_persistence: ProblemOccurrencePersistence
    tenant_binding: HostedDiagnosticTenantBinding
    publisher: HostedApplicationEventPublisher
    published_events: list[HostedApplicationEvent]


def _build_worker_bootstrap_diagnostic_harness(
    *,
    orchestrator: DiagnosticOrchestrator | None = None,
    tenant_id: str = "local_workspace.product",
) -> _WorkerBootstrapDiagnosticHarness:
    built_orchestrator, persistence, read_service, occurrence_persistence = (
        build_diagnostic_orchestrator_stack_for_tests()
    )
    resolved_orchestrator = orchestrator or built_orchestrator
    exporter = InMemoryObservabilityExporter()
    tenant_binding = HostedDiagnosticTenantBinding(tenant_id=tenant_id)
    published_events: list[HostedApplicationEvent] = []

    class _RecordingDiagnosticPublisher(HostedApplicationDiagnosticEventPublisher):
        async def publish(self, event: HostedApplicationEvent) -> None:
            published_events.append(event)
            await super().publish(event)

    publisher = _RecordingDiagnosticPublisher(
        observability_publisher=ObservabilityHostedApplicationEventPublisher(
            exporter,
            policy=ObservabilityExportPolicy(enabled=True),
        ),
        tenant_binding=tenant_binding,
        orchestrator=resolved_orchestrator,
    )
    return _WorkerBootstrapDiagnosticHarness(
        orchestrator=resolved_orchestrator,
        persistence=persistence,
        read_service=read_service,
        occurrence_persistence=occurrence_persistence,
        tenant_binding=tenant_binding,
        publisher=publisher,
        published_events=published_events,
    )


def _failure_events(events: list[HostedApplicationEvent]) -> list[HostedApplicationEvent]:
    return [
        event
        for event in events
        if event.event_type is HostedApplicationEventType.APPLICATION_FAILED
    ]


def _activated_projection(monkeypatch: pytest.MonkeyPatch):
    settings = _settings(monkeypatch)
    environment_profile = build_local_workspace_environment_profile(settings)
    _, projection = activate_local_workspace_reference_production_authority(
        settings,
        environment_profile=environment_profile,
    )
    return settings, environment_profile, projection


def _canonical_document_store(
    settings: LocalWorkspaceBackendSettings,
):
    return resolve_lkw_runtime_document_store(settings)


@pytest.mark.asyncio
async def test_worker_bootstrap_reuses_same_context_for_construction_and_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, environment_profile, projection = _activated_projection(monkeypatch)
    harness = _build_worker_bootstrap_diagnostic_harness(
        tenant_id=environment_profile.profile_id,
    )
    bootstrap_context = HostedProcessBootstrapContext.create(
        application_id=LOCAL_WORKSPACE_APPLICATION_MANIFEST.app_id,
        process_role=_PROCESS_ROLE,
    )
    worker = MagicMock()
    worker.start = MagicMock(side_effect=RuntimeError("startup failed"))

    with patch(
        "local_workspace_application.host.background_worker_main.build_local_workspace_background_worker_wiring",
        return_value=LocalWorkspaceBackgroundWorkerWiring(
            runtime=MagicMock(),
            host_execution=MagicMock(),
            registry=MagicMock(),
            kv_store=MagicMock(),
            idempotency_store=None,
            worker=worker,
        ),
    ):
        with pytest.raises(RuntimeError, match="startup failed"):
            await _run_guarded_worker_bootstrap(
                bootstrap_context=bootstrap_context,
                event_publisher=harness.publisher,
                settings=settings,
                registry_projection=projection,
                document_store=_canonical_document_store(settings),
            )

    failed_events = _failure_events(harness.published_events)
    assert len(failed_events) == 1
    assert failed_events[0].instance_id == bootstrap_context.instance_id
    assert failed_events[0].payload.get("phase") == HostedProcessBootstrapPhase.STARTUP.value


@pytest.mark.asyncio
async def test_worker_construction_failure_emits_application_failed_with_worker_construction_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, environment_profile, projection = _activated_projection(monkeypatch)
    harness = _build_worker_bootstrap_diagnostic_harness(
        tenant_id=environment_profile.profile_id,
    )
    bootstrap_context = HostedProcessBootstrapContext.create(
        application_id=LOCAL_WORKSPACE_APPLICATION_MANIFEST.app_id,
        process_role=_PROCESS_ROLE,
    )
    composition_error = TypeError("create_kafka_worker composition failure")

    with patch(
        "local_workspace_application.host.background_worker_main.build_local_workspace_background_worker_wiring",
        side_effect=composition_error,
    ):
        with pytest.raises(TypeError, match="create_kafka_worker composition failure"):
            await _run_guarded_worker_bootstrap(
                bootstrap_context=bootstrap_context,
                event_publisher=harness.publisher,
                settings=settings,
                registry_projection=projection,
                document_store=_canonical_document_store(settings),
            )

    failed_events = _failure_events(harness.published_events)
    assert len(failed_events) == 1
    event = failed_events[0]
    assert event.application_id == LOCAL_WORKSPACE_APPLICATION_MANIFEST.app_id
    assert event.instance_id == bootstrap_context.instance_id
    assert event.lifecycle_state is HostedApplicationLifecycleState.FAILED
    assert event.payload == {
        "phase": HostedProcessBootstrapPhase.WORKER_CONSTRUCTION.value,
        "reason_code": BOOTSTRAP_UNHANDLED_EXCEPTION_REASON_CODE,
        "exception_type": "TypeError",
        "process_role": _PROCESS_ROLE,
    }
    problems = query_all_problems_for_tenant(
        harness.persistence,
        environment_profile.profile_id,
    )
    assert len(problems) == 1


@pytest.mark.asyncio
async def test_worker_startup_failure_emits_application_failed_with_startup_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, environment_profile, projection = _activated_projection(monkeypatch)
    harness = _build_worker_bootstrap_diagnostic_harness(
        tenant_id=environment_profile.profile_id,
    )
    bootstrap_context = HostedProcessBootstrapContext.create(
        application_id=LOCAL_WORKSPACE_APPLICATION_MANIFEST.app_id,
        process_role=_PROCESS_ROLE,
    )
    startup_error = RuntimeError("worker start failed")
    worker = MagicMock()
    worker.start = MagicMock(side_effect=startup_error)

    with patch(
        "local_workspace_application.host.background_worker_main.build_local_workspace_background_worker_wiring",
        return_value=LocalWorkspaceBackgroundWorkerWiring(
            runtime=MagicMock(),
            host_execution=MagicMock(),
            registry=MagicMock(),
            kv_store=MagicMock(),
            idempotency_store=None,
            worker=worker,
        ),
    ):
        with pytest.raises(RuntimeError, match="worker start failed"):
            await _run_guarded_worker_bootstrap(
                bootstrap_context=bootstrap_context,
                event_publisher=harness.publisher,
                settings=settings,
                registry_projection=projection,
                document_store=_canonical_document_store(settings),
            )

    failed_events = _failure_events(harness.published_events)
    assert len(failed_events) == 1
    assert failed_events[0].payload.get("phase") == HostedProcessBootstrapPhase.STARTUP.value
    assert failed_events[0].payload.get("process_role") == _PROCESS_ROLE


@pytest.mark.asyncio
async def test_worker_bootstrap_uses_manifest_application_id_and_product_tenant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, environment_profile, projection = _activated_projection(monkeypatch)
    harness = _build_worker_bootstrap_diagnostic_harness(
        tenant_id=environment_profile.profile_id,
    )
    bootstrap_context = HostedProcessBootstrapContext.create(
        application_id=LOCAL_WORKSPACE_APPLICATION_MANIFEST.app_id,
        process_role=_PROCESS_ROLE,
    )

    with patch(
        "local_workspace_application.host.background_worker_main.build_local_workspace_background_worker_wiring",
        side_effect=ValueError("composition failed"),
    ):
        with pytest.raises(ValueError, match="composition failed"):
            await _run_guarded_worker_bootstrap(
                bootstrap_context=bootstrap_context,
                event_publisher=harness.publisher,
                settings=settings,
                registry_projection=projection,
                document_store=_canonical_document_store(settings),
            )

    failed_events = _failure_events(harness.published_events)
    assert failed_events[0].application_id == LOCAL_WORKSPACE_APPLICATION_MANIFEST.app_id
    problems = query_all_problems_for_tenant(
        harness.persistence,
        environment_profile.profile_id,
    )
    assert problems


@pytest.mark.asyncio
async def test_worker_bootstrap_preserves_original_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, environment_profile, projection = _activated_projection(monkeypatch)
    harness = _build_worker_bootstrap_diagnostic_harness(
        tenant_id=environment_profile.profile_id,
    )
    bootstrap_context = HostedProcessBootstrapContext.create(
        application_id=LOCAL_WORKSPACE_APPLICATION_MANIFEST.app_id,
        process_role=_PROCESS_ROLE,
    )
    original = OSError("bootstrap configuration failed")

    with patch(
        "local_workspace_application.host.background_worker_main.build_local_workspace_background_worker_wiring",
        side_effect=original,
    ):
        with pytest.raises(OSError) as exc_info:
            await _run_guarded_worker_bootstrap(
                bootstrap_context=bootstrap_context,
                event_publisher=harness.publisher,
                settings=settings,
                registry_projection=projection,
                document_store=_canonical_document_store(settings),
            )

    assert exc_info.value is original


@pytest.mark.asyncio
async def test_worker_bootstrap_diagnostic_projection_failure_does_not_replace_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, environment_profile, projection = _activated_projection(monkeypatch)
    harness = _build_worker_bootstrap_diagnostic_harness(
        tenant_id=environment_profile.profile_id,
    )
    bootstrap_context = HostedProcessBootstrapContext.create(
        application_id=LOCAL_WORKSPACE_APPLICATION_MANIFEST.app_id,
        process_role=_PROCESS_ROLE,
    )
    original = ValueError("classified bootstrap failure")

    with (
        patch.object(
            harness.orchestrator,
            "run",
            side_effect=RuntimeError("projection failed"),
        ),
        patch(
            "local_workspace_application.host.background_worker_main.build_local_workspace_background_worker_wiring",
            side_effect=original,
        ),
    ):
        with pytest.raises(ValueError) as exc_info:
            await _run_guarded_worker_bootstrap(
                bootstrap_context=bootstrap_context,
                event_publisher=harness.publisher,
                settings=settings,
                registry_projection=projection,
                document_store=_canonical_document_store(settings),
            )

    assert exc_info.value is original
    assert len(_failure_events(harness.published_events)) == 1


@pytest.mark.asyncio
async def test_successful_worker_bootstrap_emits_no_failure_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, environment_profile, projection = _activated_projection(monkeypatch)
    harness = _build_worker_bootstrap_diagnostic_harness(
        tenant_id=environment_profile.profile_id,
    )
    bootstrap_context = HostedProcessBootstrapContext.create(
        application_id=LOCAL_WORKSPACE_APPLICATION_MANIFEST.app_id,
        process_role=_PROCESS_ROLE,
    )
    worker = MagicMock()
    worker.start = MagicMock()

    with patch(
        "local_workspace_application.host.background_worker_main.build_local_workspace_background_worker_wiring",
        return_value=LocalWorkspaceBackgroundWorkerWiring(
            runtime=MagicMock(),
            host_execution=MagicMock(),
            registry=MagicMock(),
            kv_store=MagicMock(),
            idempotency_store=None,
            worker=worker,
        ),
    ):
        await _run_guarded_worker_bootstrap(
            bootstrap_context=bootstrap_context,
            event_publisher=harness.publisher,
            settings=settings,
            registry_projection=projection,
            document_store=_canonical_document_store(settings),
        )

    assert _failure_events(harness.published_events) == []
    worker.start.assert_called_once()


def test_worker_main_has_no_execution_identity_imports() -> None:
    source = _WORKER_MAIN_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(_WORKER_MAIN_PATH))
    rel = _WORKER_MAIN_PATH.relative_to(_REPO_ROOT).as_posix()
    violations: list[str] = []
    for token in _FORBIDDEN_SOURCE_TOKENS:
        if token in source:
            violations.append(f"{rel} references forbidden token {token}")
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for prefix in _FORBIDDEN_IMPORT_PREFIXES:
                if node.module.startswith(prefix):
                    violations.append(f"{rel}:{node.lineno} imports {node.module}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(_FORBIDDEN_IMPORT_PREFIXES):
                    violations.append(f"{rel}:{node.lineno} imports {alias.name}")
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_IMPORT_SYMBOLS:
            violations.append(f"{rel}:{node.lineno} references {node.id}")
        if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_IMPORT_SYMBOLS:
            violations.append(f"{rel}:{node.lineno} references .{node.attr}")
    assert violations == []


def test_worker_main_uses_canonical_diagnostic_composition_surface() -> None:
    source = _WORKER_MAIN_PATH.read_text(encoding="utf-8")
    assert "build_hosted_application_diagnostic_event_publisher" in source
    assert "build_diagnostic_orchestrator" in source
    assert "resolve_local_workspace_observability_exporter" in source
    assert "build_local_workspace_worker_bootstrap_diagnostics" in source
    assert "resolve_lkw_runtime_document_store" in source
    assert "run_guarded_hosted_process_bootstrap" in source
    assert "HostedProcessBootstrapPhase.WORKER_CONSTRUCTION" in source
    assert "HostedProcessBootstrapPhase.STARTUP" in source
    assert "LOCAL_WORKSPACE_APPLICATION_MANIFEST.app_id" in source
    assert "WorkerDiagnosticPublisher" not in source
    assert "LkwWorkerDiagnosticPublisher" not in source


class _OrderRecordingObservabilityExporter(InMemoryObservabilityExporter):
    def __init__(self, order: list[str]) -> None:
        super().__init__()
        self._order = order

    async def export(self, envelope: ObservabilityExportEnvelope) -> None:
        self._order.append("observability")
        await super().export(envelope)


class _OrderRecordingDiagnosticOrchestrator:
    def __init__(self, order: list[str], orchestrator: DiagnosticOrchestrator) -> None:
        self._order = order
        self._orchestrator = orchestrator

    def run(self, request: DiagnosticOrchestrationRequest) -> DiagnosticOrchestrationResult:
        self._order.append("diagnostics")
        return self._orchestrator.run(request)


def test_worker_bootstrap_diagnostics_passes_explicit_observability_exporter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, environment_profile, projection = _activated_projection(monkeypatch)
    document_store = _canonical_document_store(settings)
    recording_exporter = InMemoryObservabilityExporter()
    captured: list[ObservabilityExporter | None] = []

    def _capture_build(
        *,
        tenant_binding: HostedDiagnosticTenantBinding,
        orchestrator: DiagnosticOrchestrator,
        observability_exporter: ObservabilityExporter | None = None,
        observability_policy: ObservabilityExportPolicy | None = None,
    ) -> HostedApplicationEventPublisher:
        captured.append(observability_exporter)
        return build_hosted_application_diagnostic_event_publisher(
            tenant_binding=tenant_binding,
            orchestrator=orchestrator,
            observability_exporter=observability_exporter,
            observability_policy=observability_policy,
        )

    monkeypatch.setattr(
        "local_workspace_application.host.background_worker_main.build_hosted_application_diagnostic_event_publisher",
        _capture_build,
    )
    monkeypatch.setattr(
        "local_workspace_application.host.background_worker_main.resolve_local_workspace_observability_exporter",
        lambda _settings: recording_exporter,
    )

    bootstrap = build_local_workspace_worker_bootstrap_diagnostics(
        registry_projection=projection,
        settings=settings,
        environment_profile=environment_profile,
        document_store=document_store,
    )

    assert captured == [recording_exporter]
    assert isinstance(bootstrap.event_publisher, HostedApplicationDiagnosticEventPublisher)


@pytest.mark.asyncio
async def test_worker_bootstrap_observability_export_precedes_diagnostic_orchestration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, environment_profile, projection = _activated_projection(monkeypatch)
    document_store = _canonical_document_store(settings)
    order: list[str] = []
    built_orchestrator, _, _, _ = build_diagnostic_orchestrator_stack_for_tests()
    recording_orchestrator = _OrderRecordingDiagnosticOrchestrator(order, built_orchestrator)
    recording_exporter = _OrderRecordingObservabilityExporter(order)

    monkeypatch.setattr(
        "local_workspace_application.host.background_worker_main.build_diagnostic_orchestrator",
        lambda _dependencies: recording_orchestrator,
    )
    monkeypatch.setattr(
        "local_workspace_application.host.background_worker_main.resolve_local_workspace_observability_exporter",
        lambda _settings: recording_exporter,
    )

    publisher = build_local_workspace_worker_bootstrap_diagnostics(
        registry_projection=projection,
        settings=settings,
        environment_profile=environment_profile,
        document_store=document_store,
    ).event_publisher
    bootstrap_context = HostedProcessBootstrapContext.create(
        application_id=LOCAL_WORKSPACE_APPLICATION_MANIFEST.app_id,
        process_role=_PROCESS_ROLE,
    )

    with patch(
        "local_workspace_application.host.background_worker_main.build_local_workspace_background_worker_wiring",
        side_effect=TypeError("create_kafka_worker composition failure"),
    ):
        with pytest.raises(TypeError, match="create_kafka_worker composition failure"):
            await _run_guarded_worker_bootstrap(
                bootstrap_context=bootstrap_context,
                event_publisher=publisher,
                settings=settings,
                registry_projection=projection,
                document_store=document_store,
            )

    assert order[:2] == ["observability", "diagnostics"]
    assert len(recording_exporter.envelopes) == 1


def test_worker_bootstrap_diagnostic_and_worker_runtime_share_canonical_document_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, environment_profile, projection = _activated_projection(monkeypatch)
    resolved_once = _canonical_document_store(settings)
    resolved_again = _canonical_document_store(settings)
    assert resolved_once is not resolved_again

    diagnostic_runtime = _build_worker_diagnostic_runtime(
        registry_projection=projection,
        settings=settings,
        document_store=resolved_once,
    )
    worker_runtime = _build_worker_diagnostic_runtime(
        registry_projection=projection,
        settings=settings,
        document_store=resolved_once,
    )

    diagnostic_read = build_diagnostic_read_service(
        resolve_host_diagnostic_read_dependencies(diagnostic_runtime),
    )
    worker_read = build_diagnostic_read_service(
        resolve_host_diagnostic_read_dependencies(worker_runtime),
    )

    assert diagnostic_read is not worker_read
    assert diagnostic_runtime is not worker_runtime
    assert environment_profile.profile_id


@pytest.mark.asyncio
async def test_worker_bootstrap_b6_failure_problem_visible_via_worker_read_side(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, environment_profile, projection = _activated_projection(monkeypatch)
    document_store = _canonical_document_store(settings)
    recording_exporter = InMemoryObservabilityExporter()
    monkeypatch.setattr(
        "local_workspace_application.host.background_worker_main.resolve_local_workspace_observability_exporter",
        lambda _settings: recording_exporter,
    )
    publisher = build_local_workspace_worker_bootstrap_diagnostics(
        registry_projection=projection,
        settings=settings,
        environment_profile=environment_profile,
        document_store=document_store,
    ).event_publisher
    bootstrap_context = HostedProcessBootstrapContext.create(
        application_id=LOCAL_WORKSPACE_APPLICATION_MANIFEST.app_id,
        process_role=_PROCESS_ROLE,
    )

    with patch(
        "local_workspace_application.host.background_worker_main.build_local_workspace_background_worker_wiring",
        side_effect=TypeError("create_kafka_worker composition failure"),
    ):
        with pytest.raises(TypeError, match="create_kafka_worker composition failure"):
            await _run_guarded_worker_bootstrap(
                bootstrap_context=bootstrap_context,
                event_publisher=publisher,
                settings=settings,
                registry_projection=projection,
                document_store=document_store,
            )

    assert len(recording_exporter.envelopes) == 1
    worker_runtime = _build_worker_diagnostic_runtime(
        registry_projection=projection,
        settings=settings,
        document_store=document_store,
    )
    read_service = build_diagnostic_read_service(
        resolve_host_diagnostic_read_dependencies(worker_runtime),
    )
    listed = read_service.list_problems(tenant_id=environment_profile.profile_id)
    assert listed.total_count == 1


def test_worker_main_publisher_factory_is_host_diag_3() -> None:
    publisher = build_hosted_application_diagnostic_event_publisher(
        tenant_binding=HostedDiagnosticTenantBinding(tenant_id="local_workspace.product"),
        orchestrator=build_diagnostic_orchestrator_stack_for_tests()[0],
        observability_exporter=InMemoryObservabilityExporter(),
    )
    assert isinstance(publisher, HostedApplicationDiagnosticEventPublisher)
