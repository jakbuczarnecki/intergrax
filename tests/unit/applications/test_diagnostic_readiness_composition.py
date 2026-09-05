# © Artur Czarnecki. All rights reserved.

"""DIAG-FOUNDATION-1 — diagnostic readiness composition contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from echo.echo_agent import EchoAgent
from intergrax.applications._shared.diagnostic_assembly_resolver import (
    DiagnosticAssemblyError,
    DiagnosticReadiness,
    resolve_central_diagnostics_required,
)
from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications._shared.harness_registry_authority import RegistryAssemblyMode
from intergrax.applications._shared.scenario_runtime_baseline import (
    ScenarioRuntimeBuildError,
    build_scenario_runtime_from_environment,
)
from intergrax.applications._shared.scenario_runtime_profiles import (
    ScenarioRuntimeMode,
    build_scenario_lab_runtime,
    build_scenario_production_runtime,
    cleanup_scenario_runtime_workspace,
)
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    DiagnosticPosture,
    DiagnosticProfile,
    ObservabilityProfile,
)
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.applications._shared.harness_host_runtime_compat import resolve_harness_host_nexus_loop_legacy
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.registry.agent_registry import AgentRegistry
from lab_application.host.settings import LabApplicationSettings
from lab_application.manifest import build_lab_manifest

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]

_TENANT = "diag-readiness-tenant"


def _echo_registry() -> AgentRegistry:
    registry = AgentRegistry()
    registry.register(EchoAgent())
    return registry


def _scenario_manifest(app_id: str = "diag_readiness_test") -> ApplicationManifest:
    return ApplicationManifest.lab(
        app_id=app_id,
        name="Diagnostic Readiness Test",
        route_prefix="/v1/diag_readiness",
        env_prefix="DIAG_READINESS_",
        agents=[AgentBinding.mount(EchoAgent, contract_id="echo", capabilities=["echo.basic"])],
    )


def _production_attached_environment(profile_id: str) -> ApplicationEnvironmentProfile:
    environment = ApplicationEnvironmentProfile.lab_defaults(profile_id=profile_id)
    environment.execution_mode = ExecutionMode.STRICT
    return environment


def _product_host_environment() -> ApplicationEnvironmentProfile:
    environment = ApplicationEnvironmentProfile.lab_defaults(profile_id="diag.product.host")
    environment.application_profile = ApplicationProfile.PRODUCT
    environment.observability_profile = ObservabilityProfile(
        trace_sqlite_enabled=True,
        otel_enabled=False,
        metrics_plugins_enabled=True,
    )
    return environment


def _product_manifest() -> ApplicationManifest:
    env = _product_host_environment()
    return ApplicationManifest.lab(
        app_id="diag_product_host",
        name="Diagnostic Product Host",
        route_prefix="/v1/diag_product",
        env_prefix="DIAG_PRODUCT_",
        agents=[AgentBinding.mount(EchoAgent, contract_id="echo", capabilities=["echo.basic"])],
        environment=env,
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


def test_resolve_central_diagnostics_required_product_and_production_attached() -> None:
    lab_env = ApplicationEnvironmentProfile.lab_defaults(profile_id="diag.lab")
    product_env = ApplicationEnvironmentProfile.product_defaults(profile_id="diag.product")
    assert resolve_central_diagnostics_required(lab_env) is False
    assert resolve_central_diagnostics_required(product_env) is True
    assert (
        resolve_central_diagnostics_required(
            lab_env,
            scenario_runtime_mode=ScenarioRuntimeMode.PRODUCTION_ATTACHED,
        )
        is True
    )


def test_resolve_central_diagnostics_required_monotonic_posture() -> None:
    """Profile posture may strengthen LAB but never downgrade hard requirements."""
    not_required = DiagnosticProfile(posture=DiagnosticPosture.NOT_REQUIRED)
    required = DiagnosticProfile(posture=DiagnosticPosture.REQUIRED)

    lab_env = ApplicationEnvironmentProfile.lab_defaults(profile_id="diag.lab.mono")
    lab_env.diagnostic_profile = not_required
    assert resolve_central_diagnostics_required(lab_env) is False

    lab_env.diagnostic_profile = required
    assert resolve_central_diagnostics_required(lab_env) is True

    product_env = ApplicationEnvironmentProfile.product_defaults(profile_id="diag.product.mono")
    product_env.diagnostic_profile = not_required
    assert resolve_central_diagnostics_required(product_env) is True

    production_lab_env = ApplicationEnvironmentProfile.lab_defaults(profile_id="diag.prod_att.mono")
    production_lab_env.diagnostic_profile = not_required
    assert (
        resolve_central_diagnostics_required(
            production_lab_env,
            scenario_runtime_mode=ScenarioRuntimeMode.PRODUCTION_ATTACHED,
        )
        is True
    )


def test_product_host_fails_without_document_store(
    tmp_path: Path,
    _stub_llm: None,
) -> None:
    manifest = _product_manifest()
    env = manifest.environment
    assert env is not None
    with pytest.raises(DiagnosticAssemblyError, match="document store"):
        build_harness_host_runtime(
            manifest,
            env,
            registry=_echo_registry(),
            registry_assembly_mode=RegistryAssemblyMode.MANIFEST_DEVELOPMENT,
            trace_db_path=tmp_path / "trace.db",
            runtime_events_db_path=tmp_path / "events.db",
            document_store=None,
        )


def test_product_host_fails_without_runtime_event_persistence(
    tmp_path: Path,
    _stub_llm: None,
) -> None:
    manifest = _product_manifest()
    env = manifest.environment
    assert env is not None
    env.observability_profile = env.observability_profile.model_copy(
        update={"trace_sqlite_enabled": False},
    )
    with pytest.raises(DiagnosticAssemblyError, match="RuntimeEvent persistence"):
        build_harness_host_runtime(
            manifest,
            env,
            registry=_echo_registry(),
            registry_assembly_mode=RegistryAssemblyMode.MANIFEST_DEVELOPMENT,
            document_store=InMemoryDocumentStore(),
            use_in_memory_trace=True,
            runtime_events_db_path=None,
        )


def test_product_host_attaches_diagnostics_with_prerequisites(
    tmp_path: Path,
    _stub_llm: None,
) -> None:
    manifest = _product_manifest()
    env = manifest.environment
    assert env is not None
    runtime = build_harness_host_runtime(
        manifest,
        env,
        registry=_echo_registry(),
        registry_assembly_mode=RegistryAssemblyMode.MANIFEST_DEVELOPMENT,
        document_store=InMemoryDocumentStore(),
        trace_db_path=tmp_path / "trace.db",
        runtime_events_db_path=tmp_path / "events.db",
    )
    assert runtime.diagnostic_wiring.required is True
    assert runtime.diagnostic_wiring.attached is True
    assert runtime.diagnostic_wiring.readiness is DiagnosticReadiness.ATTACHED
    assert resolve_harness_host_nexus_loop_legacy(runtime)._terminal_diagnostic_trigger is not None  # noqa: SLF001


def test_production_attached_scenario_fails_without_diagnostic_prerequisites(
    tmp_path: Path,
) -> None:
    environment = _production_attached_environment("diag.production.no_store")
    with pytest.raises(ScenarioRuntimeBuildError, match="central diagnostics are required"):
        build_scenario_production_runtime(
            environment=environment,
            manifest=_scenario_manifest("diag_prod_no_store"),
            registry=_echo_registry(),
            tenant_id=_TENANT,
            runtime_events_db_path=tmp_path / "events.db",
            trace_db_path=tmp_path / "trace.db",
            document_store=None,
        )


def test_production_attached_scenario_attaches_with_prerequisites(tmp_path: Path) -> None:
    environment = _production_attached_environment("diag.production.ok")
    composition = build_scenario_production_runtime(
        environment=environment,
        manifest=_scenario_manifest("diag_prod_ok"),
        registry=_echo_registry(),
        tenant_id=_TENANT,
        runtime_events_db_path=tmp_path / "events.db",
        trace_db_path=tmp_path / "trace.db",
        document_store=InMemoryDocumentStore(),
    )
    assert composition.diagnostic_wiring.required is True
    assert composition.diagnostic_wiring.readiness is DiagnosticReadiness.ATTACHED
    assert composition.has_terminal_diagnostic_trigger is True


def test_lab_with_canonical_prerequisites_attaches_diagnostics() -> None:
    composition = build_scenario_lab_runtime(
        registry=_echo_registry(),
        tenant_id=_TENANT,
    )
    assert composition.diagnostic_wiring.required is False
    assert composition.diagnostic_wiring.attached is True
    assert composition.diagnostic_wiring.readiness is DiagnosticReadiness.ATTACHED
    workspace = composition.workspace
    assert workspace is not None
    cleanup_scenario_runtime_workspace(workspace)


def test_optional_lab_without_prerequisites_reports_unavailable_readiness(
    tmp_path: Path,
) -> None:
    environment = ApplicationEnvironmentProfile.lab_defaults(profile_id="diag.lab.optional")
    composition = build_scenario_runtime_from_environment(
        environment=environment,
        registry=_echo_registry(),
        tenant_id=_TENANT,
        manifest=_scenario_manifest("diag_lab_optional"),
        runtime_events_db_path=tmp_path / "events.db",
        trace_db_path=tmp_path / "trace.db",
        document_store=None,
        runtime_mode=ScenarioRuntimeMode.LAB,
    )
    assert composition.diagnostic_wiring.required is False
    assert composition.diagnostic_wiring.attached is False
    assert composition.diagnostic_wiring.readiness is DiagnosticReadiness.NOT_REQUIRED_UNAVAILABLE


def test_lab_host_not_required_without_prerequisites() -> None:
    settings = LabApplicationSettings.from_env()
    manifest = build_lab_manifest(settings)
    env = manifest.environment
    assert env is not None
    runtime = build_harness_host_runtime(manifest, env, settings=settings)
    assert runtime.diagnostic_wiring.required is False


def test_explicit_diagnostic_posture_required_on_lab_profile(tmp_path: Path) -> None:
    environment = ApplicationEnvironmentProfile.lab_defaults(profile_id="diag.lab.explicit")
    environment.diagnostic_profile = DiagnosticProfile(posture=DiagnosticPosture.REQUIRED)
    with pytest.raises(ScenarioRuntimeBuildError, match="central diagnostics are required"):
        build_scenario_runtime_from_environment(
            environment=environment,
            registry=_echo_registry(),
            tenant_id=_TENANT,
            manifest=_scenario_manifest("diag_lab_explicit"),
            runtime_events_db_path=tmp_path / "events.db",
            trace_db_path=tmp_path / "trace.db",
            document_store=None,
            runtime_mode=ScenarioRuntimeMode.LAB,
        )
