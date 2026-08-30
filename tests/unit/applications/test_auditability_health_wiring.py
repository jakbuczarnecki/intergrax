# © Artur Czarnecki. All rights reserved.

"""DIAG-FOUNDATION-2 — auditability health projection and observability integration."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.applications._shared.auditability_health_wiring import (
    HostAuditabilityHealthFacts,
    assert_host_auditability_health_valid,
    project_auditability_health_snapshot,
    project_conservative_auditability_health_facts,
    project_host_auditability_health_facts,
    project_host_auditability_health_facts_from_runtime,
)
from intergrax.applications._shared.diagnostic_assembly_resolver import (
    DiagnosticReadiness,
    DiagnosticWiring,
    resolve_central_diagnostics_required,
)
from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications._shared.harness_registry_authority import RegistryAssemblyMode
from intergrax.applications._shared.health_dashboard_wiring import (
    resolve_health_dashboard_wiring,
    resolve_health_dashboard_wiring_from_runtime,
)
from intergrax.applications._shared.observability_assembly_resolver import ObservabilityAssemblyError
from intergrax.applications._shared.product_observability_dashboard_wiring import (
    _build_diagnostic_operations_pane,
    resolve_product_observability_dashboard_wiring,
)
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    DiagnosticPosture,
    DiagnosticProfile,
    ObservabilityProfile,
)
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from echo.echo_agent import EchoAgent
from intergrax.contracts.execution_mode import ExecutionMode
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.observability.auditability_health import AuditabilityDiagnosticReadiness
from intergrax.runtime.registry.agent_registry import AgentRegistry
from tests.unit.applications.test_product_observability_dashboard_wiring import (
    _product_env,
    _read_service_with_problems,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TENANT = "auditability-health-host"


def _attached_product_facts(*, read_side_ready: bool = True) -> HostAuditabilityHealthFacts:
    env = _product_env()
    return project_host_auditability_health_facts(
        env=env,
        diagnostic_wiring=DiagnosticWiring(required=True, attached=True),
        runtime_event_persistence_available=True,
        diagnostic_read_side_ready=read_side_ready,
    )


def _unavailable_required_product_facts(
    *,
    read_side_ready: bool = False,
) -> HostAuditabilityHealthFacts:
    env = _product_env()
    return project_host_auditability_health_facts(
        env=env,
        diagnostic_wiring=DiagnosticWiring(required=True, attached=False),
        runtime_event_persistence_available=False,
        diagnostic_read_side_ready=read_side_ready,
    )


def _lab_optional_unavailable_facts() -> HostAuditabilityHealthFacts:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="auditability.lab.optional")
    return project_host_auditability_health_facts(
        env=env,
        diagnostic_wiring=DiagnosticWiring(required=False, attached=False),
        runtime_event_persistence_available=False,
        diagnostic_read_side_ready=False,
    )


def _lab_attached_facts() -> HostAuditabilityHealthFacts:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="auditability.lab.attached")
    env.diagnostic_profile = DiagnosticProfile(posture=DiagnosticPosture.REQUIRED)
    return project_host_auditability_health_facts(
        env=env,
        diagnostic_wiring=DiagnosticWiring(required=True, attached=True),
        runtime_event_persistence_available=True,
        diagnostic_read_side_ready=False,
    )


@pytest.fixture
def _stub_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    from testing_support.builder import MeteringFakeLLMAdapter

    adapter = MeteringFakeLLMAdapter()

    def _resolve(env: object, agent_override: object | None = None, **_: object) -> object:
        del env
        return agent_override or adapter

    monkeypatch.setattr(
        "intergrax.applications._shared.llm_resolver.resolve_llm_adapter",
        _resolve,
    )


def test_case_a_product_attached_runtime_event_persistence_auditability_ready() -> None:
    snapshot = project_auditability_health_snapshot(_attached_product_facts())
    assert snapshot.diagnostics_required is True
    assert snapshot.diagnostics_attached is True
    assert snapshot.runtime_event_persistence_available is True
    assert snapshot.diagnostic_readiness is AuditabilityDiagnosticReadiness.ATTACHED
    assert snapshot.auditability_ready is True


def test_case_b_product_required_unavailable_not_ready_and_gate_fails() -> None:
    facts = _unavailable_required_product_facts()
    snapshot = project_auditability_health_snapshot(facts)
    assert snapshot.auditability_ready is False
    wiring = resolve_health_dashboard_wiring(_product_env(), auditability_facts=facts)
    assert wiring.contract is not None
    assert wiring.contract.auditability.auditability_ready is False
    with pytest.raises(ObservabilityAssemblyError, match="auditability is not ready"):
        assert_host_auditability_health_valid(facts, _product_env())


def test_case_c_lab_optional_unavailable_allowed_explicit_payload() -> None:
    facts = _lab_optional_unavailable_facts()
    snapshot = project_auditability_health_snapshot(facts)
    assert snapshot.diagnostics_required is False
    assert snapshot.diagnostics_attached is False
    assert snapshot.diagnostic_readiness is AuditabilityDiagnosticReadiness.NOT_REQUIRED_UNAVAILABLE
    assert snapshot.auditability_ready is True
    assert_host_auditability_health_valid(facts, ApplicationEnvironmentProfile.lab_defaults())


def test_case_d_lab_attached_reports_ready() -> None:
    snapshot = project_auditability_health_snapshot(_lab_attached_facts())
    assert snapshot.diagnostics_required is True
    assert snapshot.diagnostics_attached is True
    assert snapshot.auditability_ready is True


def test_case_e_diagnostics_pane_ready_with_read_service() -> None:
    env = _product_env()
    service = _read_service_with_problems(tenant_id=_TENANT, open_count=0, resolved_count=0)
    pane = _build_diagnostic_operations_pane(
        env,
        service,
        auditability_facts=_attached_product_facts(),
    )
    assert pane.ready is True


def test_case_f_required_read_unavailable_pane_and_health_not_ready() -> None:
    env = _product_env()
    facts = _unavailable_required_product_facts(read_side_ready=False)
    pane = _build_diagnostic_operations_pane(env, None, auditability_facts=facts)
    assert pane.ready is False
    assert pane.problem_count == 0
    wiring = resolve_product_observability_dashboard_wiring(
        env,
        repo_root=_REPO_ROOT,
        diagnostic_read_service=None,
        auditability_facts=facts,
    )
    assert wiring.dashboard is not None
    assert wiring.dashboard.diagnostics.ready is False
    assert wiring.dashboard.health.auditability.auditability_ready is False


def test_case_g_runtime_event_persistence_unavailable_blocks_required_auditability() -> None:
    env = _product_env()
    facts = project_host_auditability_health_facts(
        env=env,
        diagnostic_wiring=DiagnosticWiring(required=True, attached=True),
        runtime_event_persistence_available=False,
        diagnostic_read_side_ready=True,
    )
    snapshot = project_auditability_health_snapshot(facts)
    assert snapshot.auditability_ready is False


def test_conservative_env_only_projection_never_claims_attached_for_product() -> None:
    env = _product_env()
    facts = project_conservative_auditability_health_facts(env)
    snapshot = project_auditability_health_snapshot(facts)
    assert snapshot.diagnostics_required is True
    assert snapshot.diagnostics_attached is False
    assert snapshot.auditability_ready is False


def _df2_product_gate_environment(profile_id: str) -> ApplicationEnvironmentProfile:
    environment = ApplicationEnvironmentProfile.product_defaults(profile_id=profile_id)
    environment.execution_mode = ExecutionMode.BALANCED
    environment.observability_profile = environment.observability_profile.model_copy(
        update={"otel_enabled": False},
    )
    return environment


def test_runtime_health_projection_uses_diagnostic_wiring(
    tmp_path: Path,
    _stub_llm: None,
) -> None:
    environment = _df2_product_gate_environment("auditability.runtime")
    manifest = ApplicationManifest.lab(
        app_id="auditability_runtime",
        name="Auditability Runtime Host",
        route_prefix="/v1/auditability_runtime",
        env_prefix="AUDITABILITY_RUNTIME_",
        agents=[AgentBinding.mount(EchoAgent, capabilities=["echo.basic"])],
        environment=environment,
    )
    registry = AgentRegistry()
    registry.register(EchoAgent())
    runtime = build_harness_host_runtime(
        manifest,
        environment,
        registry=registry,
        registry_assembly_mode=RegistryAssemblyMode.MANIFEST_DEVELOPMENT,
        document_store=InMemoryDocumentStore(),
        trace_db_path=tmp_path / "trace.db",
        runtime_events_db_path=tmp_path / "events.db",
    )
    wiring = resolve_health_dashboard_wiring_from_runtime(
        runtime,
        diagnostic_read_side_ready=True,
    )
    assert wiring.contract is not None
    auditability = wiring.contract.auditability
    assert auditability.diagnostics_required is True
    assert auditability.diagnostics_attached is runtime.diagnostic_wiring.attached
    assert auditability.auditability_ready is True
    assert runtime.diagnostic_wiring.readiness is DiagnosticReadiness.ATTACHED


def test_harness_observability_gate_fails_when_required_diagnostics_removed() -> None:
    from scripts.maintenance import check_harness_observability_wiring as gate

    facts = _unavailable_required_product_facts()
    env = _product_env()
    with pytest.raises(ObservabilityAssemblyError):
        gate.assert_host_auditability_health_valid(facts, env)


def test_product_dashboard_exposes_auditability_health() -> None:
    env = _product_env()
    service = _read_service_with_problems(tenant_id=_TENANT, open_count=1, resolved_count=0)
    wiring = resolve_product_observability_dashboard_wiring(
        env,
        repo_root=_REPO_ROOT,
        diagnostic_read_service=service,
        auditability_facts=_attached_product_facts(),
    )
    assert wiring.dashboard is not None
    auditability = wiring.dashboard.health.auditability
    assert auditability.diagnostics_required is True
    assert auditability.diagnostics_attached is True
    assert auditability.auditability_ready is True
    assert wiring.dashboard.diagnostics.ready is True


def test_lab_runtime_optional_diagnostics_explicit_payload(
    tmp_path: Path,
    _stub_llm: None,
) -> None:
    from lab_application.host.settings import LabApplicationSettings
    from lab_application.manifest import build_lab_manifest

    settings = LabApplicationSettings.from_env()
    manifest = build_lab_manifest(settings)
    env = manifest.environment
    assert env is not None
    runtime = build_harness_host_runtime(manifest, env, settings=settings)
    facts = project_host_auditability_health_facts_from_runtime(
        runtime,
        diagnostic_read_side_ready=False,
    )
    snapshot = project_auditability_health_snapshot(facts)
    assert resolve_central_diagnostics_required(env) is False
    assert snapshot.diagnostics_required is False
    assert snapshot.auditability_ready is True
