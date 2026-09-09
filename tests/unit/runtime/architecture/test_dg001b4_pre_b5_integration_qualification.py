# © Artur Czarnecki. All rights reserved.

"""DG-001B4 — worker pre-B5 HostedBootstrapFailureRecord integration qualification."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from intergrax.hosting import (
    BOOTSTRAP_UNHANDLED_EXCEPTION_REASON_CODE,
    BootstrapReadinessLevel,
    BootstrapSurfaceKind,
    HostedApplicationEventType,
    HostedBootstrapFailureProducer,
    HostedProcessBootstrapContext,
    HostedProcessBootstrapPhase,
    LoggingBootstrapFailureReporter,
)
from intergrax.hosting.bootstrap_failure import BOOTSTRAP_FAILURE_RECORD_SCHEMA_ID
from intergrax.runtime.diagnostics.in_memory_problem_persistence import InMemoryProblemPersistence
from local_workspace_application.host.background_worker_factory import (
    LocalWorkspaceBackgroundWorkerWiring,
)
from local_workspace_application.host.background_worker_main import (
    _run_guarded_worker_bootstrap,
    build_local_workspace_worker_bootstrap_diagnostics,
    main as worker_main,
)
from local_workspace_application.manifest import LOCAL_WORKSPACE_APPLICATION_MANIFEST
from scripts.proof.dg001b4_pre_b5_qualification_contracts import (
    controlled_failing_worker_bootstrap_diagnostics_segment,
    qualification_secret_sentinel,
)
from scripts.proof.dg001b4_pre_b5_qualification_support import (
    RecordingBootstrapFailureReporter,
    build_production_equivalent_bootstrap_failure_producer,
    run_canonical_worker_pre_b5_guarded_bootstrap,
)
from tests.unit.applications.local_workspace_application.test_lkw_background_worker_bootstrap_conformance import (
    _activated_projection,
    _build_worker_bootstrap_diagnostic_harness,
    _canonical_document_store,
    _failure_events,
)
from tests.unit.runtime.diagnostics.problem_persistence_test_support import (
    query_all_problems_for_tenant,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_PROCESS_ROLE = "background_worker"


@dataclass
class _FailingBootstrapFailureReporter:
    def report(self, record: object) -> None:
        raise RuntimeError("reporter failed")


def _configure_worker_bootstrap_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERGRAX_HARNESS_API_KEY", "test-dg001b4-pre-b5-key")
    monkeypatch.setenv("LOCAL_WORKSPACE_ENABLE_MESSAGE_BUS", "true")
    monkeypatch.setenv("LOCAL_WORKSPACE_ENABLE_KAFKA_MESSAGE_BUS", "true")
    monkeypatch.setenv("LOCAL_WORKSPACE_ENABLE_REDIS", "true")
    monkeypatch.setenv(
        "INTERGRAX_PROBLEM_LIST_CURSOR_SECRET",
        "unit-test-dg001b4-pre-b5-cursor-secret",
    )


def _controlled_pre_b5_segment(monkeypatch: pytest.MonkeyPatch):
    settings, environment_profile, projection = _activated_projection(monkeypatch)
    document_store = _canonical_document_store(settings)
    failing_segment = controlled_failing_worker_bootstrap_diagnostics_segment()

    def _segment() -> object:
        return failing_segment(
            registry_projection=projection,
            settings=settings,
            environment_profile=environment_profile,
            document_store=document_store,
        )

    return settings, environment_profile, projection, document_store, _segment


def test_scenario_a_pre_b5_failure_emits_record_and_preserves_primary_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_worker_bootstrap_env(monkeypatch)
    _, _, _, _, segment = _controlled_pre_b5_segment(monkeypatch)
    reporter = RecordingBootstrapFailureReporter(records=[])
    persistence = InMemoryProblemPersistence()

    with pytest.raises(RuntimeError, match=qualification_secret_sentinel()) as exc_info:
        run_canonical_worker_pre_b5_guarded_bootstrap(
            diagnostics_segment=segment,
            extra_reporters=[reporter],
        )

    assert len(reporter.records) == 1
    record = reporter.records[0]
    assert record.schema_id == BOOTSTRAP_FAILURE_RECORD_SCHEMA_ID
    assert record.readiness_at_failure is BootstrapReadinessLevel.B3_TENANT_BINDING
    assert record.stage is HostedProcessBootstrapPhase.DEPENDENCY_RESOLUTION
    assert record.surface_kind is BootstrapSurfaceKind.WORKER_BACKGROUND
    assert record.failure_facts.exception_type == "RuntimeError"
    assert record.failure_facts.reason_code == BOOTSTRAP_UNHANDLED_EXCEPTION_REASON_CODE
    assert type(exc_info.value).__name__ == record.failure_facts.exception_type
    assert qualification_secret_sentinel() in str(exc_info.value)
    assert (
        len(query_all_problems_for_tenant(persistence, tenant_id="local_workspace.product")) == 0
    )


def test_scenario_b_pre_b5_identity_boundary_has_attempt_id_without_fabricated_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_worker_bootstrap_env(monkeypatch)
    _, _, _, _, segment = _controlled_pre_b5_segment(monkeypatch)
    reporter = RecordingBootstrapFailureReporter(records=[])

    with pytest.raises(RuntimeError, match=qualification_secret_sentinel()):
        run_canonical_worker_pre_b5_guarded_bootstrap(
            diagnostics_segment=segment,
            extra_reporters=[reporter],
        )

    record = reporter.records[0]
    assert record.bootstrap_attempt_id.startswith("bootstrap-attempt-")
    assert record.identity.bootstrap_attempt_id == record.bootstrap_attempt_id
    assert record.identity.application_id == LOCAL_WORKSPACE_APPLICATION_MANIFEST.app_id
    assert record.identity.process_role == _PROCESS_ROLE
    assert record.identity.instance_id is None
    assert record.identity.diagnostic_tenant_id is None


def test_scenario_c_reporter_failure_does_not_mask_primary_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_worker_bootstrap_env(monkeypatch)
    _, _, _, _, segment = _controlled_pre_b5_segment(monkeypatch)
    recording = RecordingBootstrapFailureReporter(records=[])

    with pytest.raises(RuntimeError, match=qualification_secret_sentinel()):
        run_canonical_worker_pre_b5_guarded_bootstrap(
            diagnostics_segment=segment,
            extra_reporters=[recording, _FailingBootstrapFailureReporter()],
        )

    assert len(recording.records) == 1
    assert recording.records[0].failure_facts.exception_type == "RuntimeError"


def test_scenario_a_worker_main_entrypoint_uses_production_producer_and_guarded_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_worker_bootstrap_env(monkeypatch)
    recording = RecordingBootstrapFailureReporter(records=[])
    producer = build_production_equivalent_bootstrap_failure_producer(extra_reporters=[recording])
    failing_segment = controlled_failing_worker_bootstrap_diagnostics_segment()
    monkeypatch.setattr(
        "local_workspace_application.host.background_worker_main._BOOTSTRAP_FAILURE_PRODUCER",
        producer,
    )
    monkeypatch.setattr(
        "local_workspace_application.host.background_worker_main.build_local_workspace_worker_bootstrap_diagnostics",
        failing_segment,
    )

    with pytest.raises(RuntimeError, match=qualification_secret_sentinel()):
        worker_main()

    assert len(recording.records) == 1
    assert recording.records[0].surface_kind is BootstrapSurfaceKind.WORKER_BACKGROUND


@pytest.mark.asyncio
async def test_scenario_d_post_b5_boundary_preserves_host_diag3_application_failed_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_worker_bootstrap_env(monkeypatch)
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
        side_effect=TypeError("create_kafka_worker composition failure"),
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
    assert failed_events[0].event_type is HostedApplicationEventType.APPLICATION_FAILED
    assert failed_events[0].payload.get("phase") == HostedProcessBootstrapPhase.WORKER_CONSTRUCTION.value


def test_pre_b5_success_path_reaches_post_b5_guard_without_bootstrap_failure_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_worker_bootstrap_env(monkeypatch)
    settings, environment_profile, projection = _activated_projection(monkeypatch)
    document_store = _canonical_document_store(settings)
    reporter = RecordingBootstrapFailureReporter(records=[])
    producer = HostedBootstrapFailureProducer(
        reporters=[LoggingBootstrapFailureReporter(), reporter],
    )
    monkeypatch.setattr(
        "local_workspace_application.host.background_worker_main._BOOTSTRAP_FAILURE_PRODUCER",
        producer,
    )

    bootstrap_diagnostics = build_local_workspace_worker_bootstrap_diagnostics(
        registry_projection=projection,
        settings=settings,
        environment_profile=environment_profile,
        document_store=document_store,
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
        import asyncio

        asyncio.run(
            _run_guarded_worker_bootstrap(
                bootstrap_context=bootstrap_context,
                event_publisher=bootstrap_diagnostics.event_publisher,
                settings=settings,
                registry_projection=projection,
                document_store=document_store,
            ),
        )

    assert reporter.records == []
    worker.start.assert_called_once()
