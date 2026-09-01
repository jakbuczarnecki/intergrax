# © Artur Czarnecki. All rights reserved.

"""DIAG-PLATFORM-D — platform diagnostic adoption conformance gate."""

from __future__ import annotations

from pathlib import Path

import pytest

from echo.echo_agent import EchoAgent
from intergrax.applications._shared.application_runtime_graph import list_application_projects
from intergrax.applications._shared.diagnostic_assembly_resolver import (
    DiagnosticAssemblyError,
    DiagnosticReadiness,
)
from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications._shared.harness_registry_authority import RegistryAssemblyMode
from intergrax.applications._shared.scenario_runtime_baseline import ScenarioRuntimeBuildError
from intergrax.applications._shared.scenario_runtime_profiles import build_scenario_production_runtime
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.registry.agent_registry import AgentRegistry
from scripts.gates.check_application_production_gates import check_no_ad_hoc_nexus_in_factories
from scripts.maintenance.check_harness_registry_resolution import check_host_wiring_adoption
from scripts.proof.create_scenario_proof import (
    ScenarioDesignRequest,
    create_scenario_design_package,
    validate_scenario_slug,
)
from scripts.proof.scenario_architecture_conformance import (
    ScenarioArchitectureConformanceError,
    assert_all_initialized_scenario_architectures,
    discover_initialized_scenario_slugs,
    validate_scenario_application_architecture,
)
from scripts.proof.scenario_lifecycle import (
    ScenarioGapDecisionStatus,
    ScenarioGateStatus,
    ScenarioImplementationStatus,
    ScenarioLifecycle,
    ScenarioLifecycleMetadata,
    write_scenario_spec_frontmatter,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.ci_smoke]

REPO_ROOT = Path(__file__).resolve().parents[4]
_TENANT = "diag-platform-gate"


def _echo_registry() -> AgentRegistry:
    registry = AgentRegistry()
    registry.register(EchoAgent())
    return registry


def _product_manifest(profile_id: str = "diag.platform.product") -> ApplicationManifest:
    environment = ApplicationEnvironmentProfile.product_defaults(profile_id=profile_id)
    environment.execution_mode = ExecutionMode.BALANCED
    environment.observability_profile = environment.observability_profile.model_copy(
        update={"otel_enabled": False},
    )
    return ApplicationManifest.lab(
        app_id="diag_platform_product",
        name="Diag Platform Product",
        route_prefix="/v1/diag_platform",
        env_prefix="DIAG_PLATFORM_",
        agents=[AgentBinding.mount(EchoAgent, capabilities=["echo.basic"])],
        environment=environment,
    )


def _scenario_manifest(app_id: str = "diag_platform_scenario") -> ApplicationManifest:
    return ApplicationManifest.lab(
        app_id=app_id,
        name="Diag Platform Scenario",
        route_prefix="/v1/diag_platform_scenario",
        env_prefix="DIAG_PLATFORM_SCENARIO_",
        agents=[AgentBinding.mount(EchoAgent, capabilities=["echo.basic"])],
    )


def _production_attached_environment(profile_id: str) -> ApplicationEnvironmentProfile:
    environment = ApplicationEnvironmentProfile.lab_defaults(profile_id=profile_id)
    environment.execution_mode = ExecutionMode.STRICT
    return environment


def _initialized_metadata(slug: str) -> ScenarioLifecycleMetadata:
    return ScenarioLifecycleMetadata(
        scenario_slug=slug,
        lifecycle=ScenarioLifecycle.IMPLEMENTATION_INITIALIZED,
        implementation_status=ScenarioImplementationStatus.INITIALIZED,
        intergrax_fit=ScenarioGateStatus.COMPLETED,
        gap_decision=ScenarioGapDecisionStatus.RESOLVED,
        observability_contract=ScenarioGateStatus.COMPLETED,
        application_vs_proof_ownership=ScenarioGateStatus.COMPLETED,
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


def test_all_application_factories_use_harness_host_runtime_spine() -> None:
    violations = check_no_ad_hoc_nexus_in_factories()
    assert violations == []


def test_all_application_host_wiring_uses_environment_wiring_spine() -> None:
    violations = check_host_wiring_adoption()
    assert violations == []


def test_adoption_guard_covers_non_suffix_application_names(tmp_path: Path) -> None:
    """Regression: discovery must not require *_application directory suffix."""
    app_dir = tmp_path / "applications" / "x7"
    host_dir = app_dir / "host"
    host_dir.mkdir(parents=True)
    (app_dir / "manifest.py").write_text("# contract\n", encoding="utf-8")
    (host_dir / "factory.py").write_text(
        "def build_factory():\n    build_harness_host_runtime()\n",
        encoding="utf-8",
    )
    (host_dir / "wiring.py").write_text(
        "def wire_host():\n    wire_application_environment()\n",
        encoding="utf-8",
    )
    assert list_application_projects(tmp_path) == ["x7"]
    assert check_no_ad_hoc_nexus_in_factories(repo_root=tmp_path) == []
    assert check_host_wiring_adoption(repo_root=tmp_path) == []


def test_destructive_host_wiring_missing_spine_detected_for_manifest_app(
    tmp_path: Path,
) -> None:
    """Regression: wiring guard must apply to every manifest-discovered application."""
    app_dir = tmp_path / "applications" / "x7"
    host_dir = app_dir / "host"
    host_dir.mkdir(parents=True)
    (app_dir / "manifest.py").write_text("# contract\n", encoding="utf-8")
    (host_dir / "wiring.py").write_text(
        "def wire_host():\n    pass\n",
        encoding="utf-8",
    )
    violations = check_host_wiring_adoption(repo_root=tmp_path)
    assert len(violations) == 1
    assert violations[0].startswith("applications/x7/host/wiring.py:")
    assert "wire_application_environment" in violations[0]
    assert "build_harness_host_runtime" in violations[0]


def test_all_initialized_scenarios_pass_architecture_conformance() -> None:
    slugs = discover_initialized_scenario_slugs(REPO_ROOT)
    assert "ai_incident_investigation" in slugs
    assert_all_initialized_scenario_architectures(REPO_ROOT)


def test_product_host_runtime_attaches_required_diagnostics(
    tmp_path: Path,
    _stub_llm: None,
) -> None:
    manifest = _product_manifest("diag.platform.attach")
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


def test_destructive_case_a_production_scenario_without_problem_persistence_fails(
    tmp_path: Path,
) -> None:
    environment = _production_attached_environment("diag.platform.case.a")
    with pytest.raises(ScenarioRuntimeBuildError, match="central diagnostics are required"):
        build_scenario_production_runtime(
            environment=environment,
            manifest=_scenario_manifest("diag_platform_no_store"),
            registry=_echo_registry(),
            tenant_id=_TENANT,
            runtime_events_db_path=tmp_path / "events.db",
            trace_db_path=tmp_path / "trace.db",
            document_store=None,
        )


def test_destructive_case_b_scenario_local_diagnostic_orchestrator_rejected(
    tmp_path: Path,
) -> None:
    slug = "diag_platform_forbidden_orchestrator"
    package = create_scenario_design_package(
        ScenarioDesignRequest(
            slug=validate_scenario_slug(slug),
            title="Forbidden Orchestrator",
            repo_root=tmp_path,
        ),
    )
    write_scenario_spec_frontmatter(package.scenario_spec_path, _initialized_metadata(slug))
    application_dir = package.package_root / "application"
    application_dir.mkdir(parents=True, exist_ok=True)
    (application_dir / "__init__.py").write_text("", encoding="utf-8")
    (application_dir / "runtime.py").write_text(
        (
            "from intergrax.runtime.diagnostics.diagnostic_orchestrator import DiagnosticOrchestrator\n"
            "from intergrax.applications._shared.scenario_runtime_profiles import "
            "build_scenario_lab_runtime\n\n"
            "def build_runtime():\n"
            "    return build_scenario_lab_runtime(registry=None, tenant_id='t', scenario_slug='x')\n"
        ),
        encoding="utf-8",
    )
    report = validate_scenario_application_architecture(
        repo_root=tmp_path,
        scenario_slug=slug,
        package_root=package.package_root,
    )
    assert not report.ok


def test_destructive_case_c_manual_nexus_bypass_detected_by_production_gate(
    tmp_path: Path,
) -> None:
    applications_root = tmp_path / "applications"
    app_dir = applications_root / "bypass_demo"
    factory_dir = app_dir / "host"
    factory_dir.mkdir(parents=True)
    (app_dir / "manifest.py").write_text("# contract\n", encoding="utf-8")
    (factory_dir / "factory.py").write_text(
        "from intergrax.runtime.nexus.nexus_loop import NexusLoop\n\n"
        "def create_app():\n"
        "    return NexusLoop()\n",
        encoding="utf-8",
    )
    (applications_root / "bypass_demo" / "pyproject.toml").write_text(
        (
            "[project]\n"
            "name = \"intergrax-bypass-demo\"\n"
            "version = \"0.1.0\"\n"
            "requires-python = \">=3.12,<3.13\"\n"
            "dependencies = []\n"
        ),
        encoding="utf-8",
    )
    violations = check_no_ad_hoc_nexus_in_factories(repo_root=tmp_path)
    assert len(violations) == 2
    assert any("bypass_demo" in item for item in violations)
    assert any("build_harness_host_runtime" in item for item in violations)
    assert any("NexusLoop" in item for item in violations)


def test_destructive_product_without_runtime_events_fails_closed(
    tmp_path: Path,
    _stub_llm: None,
) -> None:
    manifest = _product_manifest("diag.platform.no.events")
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


def test_destructive_initialized_scenario_graph_executor_rejected_in_temp_repo(
    tmp_path: Path,
) -> None:
    slug = "diag_platform_forbidden_executor"
    package = create_scenario_design_package(
        ScenarioDesignRequest(
            slug=validate_scenario_slug(slug),
            title="Forbidden Executor",
            repo_root=tmp_path,
        ),
    )
    write_scenario_spec_frontmatter(package.scenario_spec_path, _initialized_metadata(slug))
    application_dir = package.package_root / "application"
    application_dir.mkdir(parents=True, exist_ok=True)
    (application_dir / "__init__.py").write_text("", encoding="utf-8")
    (application_dir / "runtime.py").write_text(
        (
            "from intergrax.runtime.nexus.engine.graph_executor import GraphExecutor\n"
            "from intergrax.applications._shared.scenario_runtime_profiles import "
            "build_scenario_lab_runtime\n\n"
            "def build_runtime():\n"
            "    return build_scenario_lab_runtime(registry=None, tenant_id='t', scenario_slug='x')\n"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ScenarioArchitectureConformanceError):
        assert_all_initialized_scenario_architectures(tmp_path)
