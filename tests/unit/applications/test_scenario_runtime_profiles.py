# © Artur Czarnecki. All rights reserved.

"""SCENARIO-PLATFORM-4 — LAB and production-attached scenario runtime profiles."""

from __future__ import annotations

from pathlib import Path

import pytest

from echo.echo_agent import EchoAgent
from intergrax.applications._shared.scenario_runtime_baseline import (
    ScenarioExecutionRequest,
    ScenarioRuntimeBuildError,
    execute_scenario_task,
)
from intergrax.applications._shared.scenario_runtime_profiles import (
    ScenarioRuntimeMode,
    build_scenario_lab_runtime,
    build_scenario_production_runtime,
    cleanup_scenario_runtime_workspace,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.diagnostics.persistence_conformance import sample_problem
from intergrax.runtime.diagnostics.document_store_problem_persistence import wire_problem_persistence
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import TaskState

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]

_TENANT = "scenario-lab-tenant"
_PRODUCTION_TENANT = "production-tenant-real"


def _echo_registry() -> AgentRegistry:
    registry = AgentRegistry()
    registry.register(EchoAgent())
    return registry


def _scenario_manifest(app_id: str = "scenario_profile_test") -> ApplicationManifest:
    return ApplicationManifest.lab(
        app_id=app_id,
        name="Scenario Profile Test",
        route_prefix="/v1/scenario_profile_test",
        env_prefix="SCENARIO_PROFILE_TEST_",
        agents=[AgentBinding.mount(EchoAgent, capabilities=["echo.basic"])],
    )


def _production_attached_environment(profile_id: str) -> ApplicationEnvironmentProfile:
    environment = ApplicationEnvironmentProfile.lab_defaults(profile_id=profile_id)
    environment.execution_mode = ExecutionMode.STRICT
    return environment


@pytest.fixture
def _stub_scenario_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    from testing_support.builder import MeteringFakeLLMAdapter

    adapter = MeteringFakeLLMAdapter()

    def _resolve(env: object, agent_override: object | None = None, **_: object) -> object:
        del env
        return agent_override or adapter

    monkeypatch.setattr(
        "intergrax.applications._shared.llm_resolver.resolve_llm_adapter",
        _resolve,
    )


def test_lab_zero_config_build_succeeds() -> None:
    composition = build_scenario_lab_runtime(
        registry=_echo_registry(),
        tenant_id=_TENANT,
    )
    assert composition.nexus_loop is not None
    assert composition.tenant_id == _TENANT
    assert composition.has_runtime_event_store is True
    assert composition.runtime_mode is ScenarioRuntimeMode.LAB
    assert composition.workspace is not None
    assert composition.workspace.runtime_events_db_path.is_absolute()
    assert composition.workspace.trace_db_path.is_absolute()
    cleanup_scenario_runtime_workspace(composition.workspace)


def test_lab_diagnostics_enabled_by_default() -> None:
    composition = build_scenario_lab_runtime(
        registry=_echo_registry(),
        tenant_id=_TENANT,
    )
    assert composition.has_terminal_diagnostic_trigger is True
    workspace = composition.workspace
    assert workspace is not None
    cleanup_scenario_runtime_workspace(workspace)


@pytest.mark.asyncio
async def test_lab_execution_persists_terminal_runtime_events(
    _stub_scenario_llm: None,
) -> None:
    composition = build_scenario_lab_runtime(
        registry=_echo_registry(),
        tenant_id=_TENANT,
    )
    result = await execute_scenario_task(
        composition,
        ScenarioExecutionRequest(
            tenant_id=_TENANT,
            message="lab profile proof",
            capability="echo.basic",
        ),
    )
    assert result.task_result.state == TaskState.COMPLETED
    store = composition.observability.runtime_event_store
    assert store is not None
    events = store.list_for_task(result.task_id, tenant_id=_TENANT)
    assert any(event.event_type == RuntimeEventType.TASK_COMPLETED for event in events)
    workspace = composition.workspace
    assert workspace is not None
    cleanup_scenario_runtime_workspace(workspace)


@pytest.mark.asyncio
async def test_lab_document_store_supports_problem_persistence(
    _stub_scenario_llm: None,
) -> None:
    composition = build_scenario_lab_runtime(
        registry=_echo_registry(),
        tenant_id=_TENANT,
    )
    wiring_context = composition.env_wiring.build_context.tool_wiring_context
    assert wiring_context is not None
    assert wiring_context.document_store is not None
    persistence = wire_problem_persistence(document_store=wiring_context.document_store)
    record = persistence.create(sample_problem(tenant_id=_TENANT))
    loaded = persistence.get(tenant_id=_TENANT, problem_id=record.problem_id)
    assert loaded is not None
    workspace = composition.workspace
    assert workspace is not None
    cleanup_scenario_runtime_workspace(workspace)


def test_lab_storage_isolation_between_runtimes() -> None:
    composition_a = build_scenario_lab_runtime(
        registry=_echo_registry(),
        tenant_id=_TENANT,
    )
    composition_b = build_scenario_lab_runtime(
        registry=_echo_registry(),
        tenant_id=_TENANT,
    )
    workspace_a = composition_a.workspace
    workspace_b = composition_b.workspace
    assert workspace_a is not None and workspace_b is not None
    assert workspace_a.root != workspace_b.root
    assert workspace_a.runtime_events_db_path != workspace_b.runtime_events_db_path
    cleanup_scenario_runtime_workspace(workspace_a)
    cleanup_scenario_runtime_workspace(workspace_b)


def test_production_fail_closed_without_manifest(tmp_path: Path) -> None:
    environment = _production_attached_environment("scenario.production.fail")
    with pytest.raises(ScenarioRuntimeBuildError, match="ApplicationManifest"):
        build_scenario_production_runtime(
            environment=environment,
            manifest=None,  # type: ignore[arg-type]
            registry=_echo_registry(),
            tenant_id=_PRODUCTION_TENANT,
            runtime_events_db_path=tmp_path / "events.db",
        )


def test_production_fail_closed_without_runtime_storage() -> None:
    environment = _production_attached_environment("scenario.production.no_storage")
    with pytest.raises(ScenarioRuntimeBuildError, match="runtime_events_db_path"):
        build_scenario_production_runtime(
            environment=environment,
            manifest=_scenario_manifest("scenario_prod_no_storage"),
            registry=_echo_registry(),
            tenant_id=_PRODUCTION_TENANT,
            runtime_events_db_path=None,  # type: ignore[arg-type]
        )


def test_production_fail_closed_without_tenant(tmp_path: Path) -> None:
    environment = _production_attached_environment("scenario.production.no_tenant")
    with pytest.raises(ValueError, match="non-empty"):
        build_scenario_production_runtime(
            environment=environment,
            manifest=_scenario_manifest("scenario_prod_no_tenant"),
            registry=_echo_registry(),
            tenant_id="",
            runtime_events_db_path=tmp_path / "events.db",
        )


def test_production_fail_closed_when_diagnostics_required_without_document_store(
    tmp_path: Path,
) -> None:
    environment = _production_attached_environment("scenario.production.no_diag")
    with pytest.raises(ScenarioRuntimeBuildError, match="diagnostics are required"):
        build_scenario_production_runtime(
            environment=environment,
            manifest=_scenario_manifest("scenario_prod_no_diag"),
            registry=_echo_registry(),
            tenant_id=_PRODUCTION_TENANT,
            runtime_events_db_path=tmp_path / "events.db",
            trace_db_path=tmp_path / "trace.db",
            document_store=None,
            diagnostics_required=True,
        )


def test_production_success_with_explicit_configuration(tmp_path: Path) -> None:
    environment = _production_attached_environment("scenario.production.ok")
    composition = build_scenario_production_runtime(
        environment=environment,
        manifest=_scenario_manifest("scenario_prod_ok"),
        registry=_echo_registry(),
        tenant_id=_PRODUCTION_TENANT,
        runtime_events_db_path=tmp_path / "events.db",
        trace_db_path=tmp_path / "trace.db",
        document_store=InMemoryDocumentStore(),
        diagnostics_required=True,
    )
    assert composition.runtime_mode is ScenarioRuntimeMode.PRODUCTION_ATTACHED
    assert composition.has_runtime_event_store is True
    assert composition.has_terminal_diagnostic_trigger is True
    assert composition.workspace is None
