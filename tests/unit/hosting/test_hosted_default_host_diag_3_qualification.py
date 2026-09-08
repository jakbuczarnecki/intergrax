# © Artur Czarnecki. All rights reserved.

"""DG-001A — default hosted runner vs canonical HOST-DIAG-3 composition qualification."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from intergrax.applications._shared.hosted_application_diagnostic_wiring import (
    HostedApplicationDiagnosticEventPublisher,
    HostedDiagnosticTenantBinding,
    build_hosted_application_diagnostic_event_publisher,
)
from intergrax.hosting import (
    HostedApplicationProfile,
    InstancePolicy,
    RestartPolicy,
    run_hosted_application,
)
from intergrax.hosting.contracts.context import (
    HostedApplicationPaths,
    HostedApplicationProcessIdentity,
)
from intergrax.hosting.contracts.policies import InstanceExclusivityMode
from intergrax.hosting.eventing import ObservabilityHostedApplicationEventPublisher
from intergrax.hosting.runner import _RunnerFactories, _default_runner_factories
from intergrax.runtime.diagnostics.diagnostic_subject import DiagnosticSubjectKind
from intergrax.runtime.observability.export_boundary import ExportRecordKind, InMemoryObservabilityExporter
from tests.unit.applications._shared.test_hosted_application_diagnostic_integration import (
    _build_diagnostic_test_stack,
)
from tests.unit.hosting.engine._fakes import (
    FakeRuntime,
    FixedClock,
    NoopLogger,
    runtime_factory,
)
from tests.unit.hosting.test_runner import _RecordingSignalAdapter
from tests.unit.runtime.diagnostics.problem_persistence_test_support import (
    query_all_occurrences_for_problem,
    query_all_problems_for_tenant,
)

pytestmark = pytest.mark.unit

_TENANT_ID = "dg001a-qualification"
_APP_ID = "dg001a_hosted_default_app"


def _failing_runtime() -> object:
    return runtime_factory(FakeRuntime(fail_start=True))


def _profile_with_failing_runtime() -> HostedApplicationProfile:
    return HostedApplicationProfile(
        application_id=_APP_ID,
        application_factory=_failing_runtime,
        application_factory_id="tests.dg001a.hosted_default_host_diag_3",
        restart=RestartPolicy.never(),
        instance=InstancePolicy(exclusivity_mode=InstanceExclusivityMode.MULTI_INSTANCE),
    )


def _test_runner_factories(tmp_path: Path) -> _RunnerFactories:
    return _RunnerFactories(
        create_paths=lambda definition: HostedApplicationPaths(
            data_home=(tmp_path / "data" / definition.application_id).resolve(),
            run_directory=(tmp_path / "run").resolve(),
        ),
        create_clock=FixedClock,
        create_monotonic_clock=lambda: __import__(
            "intergrax.hosting.shutdown",
            fromlist=["SystemMonotonicClock"],
        ).SystemMonotonicClock(),
        create_logger=lambda _application_id: NoopLogger(),
        create_event_publisher=_default_runner_factories().create_event_publisher,
        create_process_identity=lambda clock: HostedApplicationProcessIdentity(
            process_id=5150,
            started_at=clock.now(),
        ),
        create_instance_guard=lambda definition, paths, process_identity, clock: __import__(
            "intergrax.hosting.runner",
            fromlist=["_NonExclusiveInstanceGuard"],
        )._NonExclusiveInstanceGuard(clock=clock),
        create_signal_adapter=lambda control: _RecordingSignalAdapter(),
        instance_id_generator=(lambda: "dg001a-instance-001"),
    )


def test_default_runner_factory_is_observability_only() -> None:
    factories = _default_runner_factories()
    publisher = factories.create_event_publisher()
    assert isinstance(publisher, ObservabilityHostedApplicationEventPublisher)
    assert not isinstance(publisher, HostedApplicationDiagnosticEventPublisher)


def test_run_hosted_application_exposes_event_publisher_factory_override() -> None:
    signature = inspect.signature(run_hosted_application)
    assert "event_publisher_factory" in signature.parameters


def test_default_runner_failure_has_no_central_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = _build_diagnostic_test_stack()
    monkeypatch.setattr(
        "intergrax.hosting.runner._default_runner_factories",
        lambda: _test_runner_factories(tmp_path),
    )

    run_hosted_application(_profile_with_failing_runtime())

    assert stack.read_service.list_problems(tenant_id=_TENANT_ID).total_count == 0


def test_canonical_event_publisher_factory_wires_host_diag_3_through_supervisor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = _build_diagnostic_test_stack()
    exporter = InMemoryObservabilityExporter()
    tenant_binding = HostedDiagnosticTenantBinding(tenant_id=_TENANT_ID)

    def event_publisher_factory() -> ObservabilityHostedApplicationEventPublisher:
        return build_hosted_application_diagnostic_event_publisher(
            tenant_binding=tenant_binding,
            orchestrator=stack.orchestrator,
            observability_exporter=exporter,
        )

    monkeypatch.setattr(
        "intergrax.hosting.runner._default_runner_factories",
        lambda: _test_runner_factories(tmp_path),
    )

    run_hosted_application(
        _profile_with_failing_runtime(),
        event_publisher_factory=event_publisher_factory,
    )

    platform_exports = [
        envelope
        for envelope in exporter.envelopes
        if envelope.record_kind is ExportRecordKind.PLATFORM_SIGNAL
    ]
    assert len(platform_exports) >= 1

    problems = stack.read_service.list_problems(tenant_id=_TENANT_ID)
    assert problems.total_count == 1
    stored = query_all_problems_for_tenant(stack.problem_persistence, _TENANT_ID)[0]
    occurrences = query_all_occurrences_for_problem(
        stack.occurrence_persistence,
        tenant_id=_TENANT_ID,
        problem_id=stored.problem_id,
    )
    app_ref = occurrences[0].subject_ref.application_instance()
    assert app_ref is not None
    assert app_ref.kind is DiagnosticSubjectKind.APPLICATION_INSTANCE
    assert app_ref.application_id == _APP_ID
    assert app_ref.instance_id == "dg001a-instance-001"
    assert app_ref.tenant_id == _TENANT_ID
    assert occurrences[0].subject_ref.execution() is None
