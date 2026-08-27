# © Artur Czarnecki. All rights reserved.

"""SCENARIO-PLATFORM-3A — shared scenario runtime baseline."""

from __future__ import annotations

import ast
from pathlib import Path
import pytest

from echo.echo_agent import EchoAgent
from intergrax.applications._shared.scenario_runtime_baseline import (
    ScenarioExecutionRequest,
    ScenarioRuntimeBuildError,
    build_scenario_runtime_from_environment,
    execute_scenario_task,
    validate_scenario_tenant_id,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.contracts.execution_identity import mint_task_id
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.task.task import TaskState

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]

_TENANT = "scenario-tenant-synthetic"
_FORBIDDEN_IMPORT_MODULES = frozenset(
    {
        "intergrax.applications._shared.harness_host_runtime",
        "intergrax.harness.application_host",
        "platform_proofs",
        "scripts.proof",
    }
)
_FORBIDDEN_SYMBOLS = frozenset(
    {
        "DiagnosticOrchestrator",
        "ProblemGroupingEngine",
        "ProblemLifecycleEngine",
        "ExecutionReconstructor",
        "GraphExecutor",
        "HarnessHostRuntime",
        "ApplicationHost",
        "build_harness_host_runtime",
    }
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _scenario_manifest(app_id: str = "scenario_baseline_test") -> ApplicationManifest:
    return ApplicationManifest.lab(
        app_id=app_id,
        name="Scenario Baseline Test",
        route_prefix="/v1/scenario_baseline_test",
        env_prefix="SCENARIO_BASELINE_TEST_",
        agents=[AgentBinding.mount(EchoAgent, capabilities=["echo.basic"])],
    )


def _echo_registry() -> AgentRegistry:
    registry = AgentRegistry()
    registry.register(EchoAgent())
    return registry


def _build_composition(
    tmp_path: Path,
    *,
    document_store: InMemoryDocumentStore | None = None,
    use_in_memory_trace: bool = True,
    tenant_id: str = _TENANT,
) -> object:
    environment = ApplicationEnvironmentProfile.lab_defaults(profile_id="scenario.baseline.lab")
    return build_scenario_runtime_from_environment(
        environment=environment,
        registry=_echo_registry(),
        tenant_id=tenant_id,
        manifest=_scenario_manifest(),
        runtime_events_db_path=tmp_path / "events.db",
        trace_db_path=tmp_path / "trace.db",
        document_store=document_store,
        use_in_memory_trace=use_in_memory_trace,
    )


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


def test_build_scenario_runtime_returns_nexus_backed_composition(tmp_path: Path) -> None:
    composition = _build_composition(tmp_path)

    assert composition.nexus_loop is not None
    assert composition.tenant_id == _TENANT
    assert composition.env_wiring.build_context.policy_bundle is not None
    assert composition.security_wiring is not None
    assert composition.guardrail_wiring is not None
    assert composition.nexus_loop.policy_engine is not None
    assert composition.has_runtime_event_store is True


def test_validate_scenario_tenant_id_rejects_whitespace_and_empty() -> None:
    with pytest.raises(ValueError, match="whitespace"):
        validate_scenario_tenant_id(" tenant")
    with pytest.raises(ValueError, match="whitespace"):
        validate_scenario_tenant_id("tenant ")
    with pytest.raises(ValueError, match="non-empty"):
        validate_scenario_tenant_id("")


@pytest.mark.asyncio
async def test_execute_scenario_task_identity_and_tenant(
    tmp_path: Path,
    _stub_scenario_llm: None,
) -> None:
    composition = _build_composition(tmp_path)
    explicit_task_id = mint_task_id()
    request = ScenarioExecutionRequest(
        tenant_id=_TENANT,
        user_id="scenario-user",
        message="hello scenario",
        capability="echo.basic",
        task_id=explicit_task_id,
    )

    captured: dict[str, object] = {}
    original_handle = composition.nexus_loop.handle_task

    async def _capture_handle(task: object, *, run_id: object, attempt_id: object = None) -> object:
        captured["run_id"] = run_id
        captured["attempt_id"] = attempt_id
        return await original_handle(task, run_id=run_id, attempt_id=attempt_id)

    composition.nexus_loop.handle_task = _capture_handle  # type: ignore[method-assign]

    result = await execute_scenario_task(composition, request)

    assert result.run_id == captured["run_id"]
    assert captured["attempt_id"] is None
    assert result.task_id == explicit_task_id
    assert result.tenant_id == _TENANT
    assert result.task_result.task_id == str(explicit_task_id)
    assert result.task_result.run_id == str(result.run_id)


@pytest.mark.asyncio
async def test_execute_scenario_task_rejects_tenant_mismatch(tmp_path: Path) -> None:
    composition = _build_composition(tmp_path)
    request = ScenarioExecutionRequest(
        tenant_id="other-tenant",
        message="hello",
    )
    with pytest.raises(ValueError, match="tenant_id"):
        await execute_scenario_task(composition, request)


@pytest.mark.asyncio
async def test_execute_scenario_task_persists_terminal_runtime_events(
    tmp_path: Path,
    _stub_scenario_llm: None,
) -> None:
    composition = _build_composition(tmp_path)
    result = await execute_scenario_task(
        composition,
        ScenarioExecutionRequest(
            tenant_id=_TENANT,
            message="runtime event proof",
            capability="echo.basic",
        ),
    )
    assert result.task_result.state == TaskState.COMPLETED
    store = composition.observability.runtime_event_store
    assert store is not None
    events = store.list_for_task(result.task_id, tenant_id=_TENANT)
    assert events
    assert any(event.event_type == RuntimeEventType.TASK_COMPLETED for event in events)


def test_build_scenario_runtime_attaches_terminal_diagnostic_trigger_with_document_store(
    tmp_path: Path,
) -> None:
    composition = _build_composition(tmp_path, document_store=InMemoryDocumentStore())
    assert composition.has_terminal_diagnostic_trigger is True
    assert composition.nexus_loop._terminal_diagnostic_trigger is not None  # noqa: SLF001


def test_build_scenario_runtime_lab_without_document_store_has_no_diagnostic_trigger(
    tmp_path: Path,
) -> None:
    composition = _build_composition(tmp_path, document_store=None)
    assert composition.has_terminal_diagnostic_trigger is False
    assert composition.has_runtime_event_store is True


@pytest.mark.asyncio
async def test_execute_scenario_task_works_without_document_store(
    tmp_path: Path,
    _stub_scenario_llm: None,
) -> None:
    composition = _build_composition(tmp_path, document_store=None)
    result = await execute_scenario_task(
        composition,
        ScenarioExecutionRequest(
            tenant_id=_TENANT,
            message="lab without diagnostics",
            capability="echo.basic",
        ),
    )
    assert result.task_result.state == TaskState.COMPLETED


def test_build_scenario_runtime_fails_closed_without_runtime_event_store(tmp_path: Path) -> None:
    environment = ApplicationEnvironmentProfile.lab_defaults(profile_id="scenario.no.events")
    environment.observability_profile = environment.observability_profile.model_copy(
        update={"trace_sqlite_enabled": False},
    )
    with pytest.raises(ScenarioRuntimeBuildError, match="RuntimeEvent persistence"):
        build_scenario_runtime_from_environment(
            environment=environment,
            registry=_echo_registry(),
            tenant_id=_TENANT,
            manifest=_scenario_manifest("scenario_no_events"),
            use_in_memory_trace=True,
            runtime_events_db_path=None,
        )


def test_scenario_runtime_baseline_architecture_gate() -> None:
    path = _repo_root() / "intergrax/applications/_shared/scenario_runtime_baseline.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    rel = path.relative_to(_repo_root()).as_posix()
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in _FORBIDDEN_IMPORT_MODULES:
            violations.append(f"{rel}:{node.lineno} imports from {node.module}")
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_SYMBOLS:
            violations.append(f"{rel}:{node.lineno} references {node.id}")
        if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_SYMBOLS:
            violations.append(f"{rel}:{node.lineno} references .{node.attr}")
    assert violations == []


def test_scenario_runtime_baseline_does_not_bind_active_identity() -> None:
    path = _repo_root() / "intergrax/applications/_shared/scenario_runtime_baseline.py"
    source = path.read_text(encoding="utf-8")
    assert "bind_active_execution_identity" not in source
    assert "mint_attempt_id" not in source
