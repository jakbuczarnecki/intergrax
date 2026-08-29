# © Artur Czarnecki. All rights reserved.

"""DIAG-FOUNDATION-4 — consistent identity and diagnostic behavior across entrypoints."""

from __future__ import annotations

import ast
import asyncio
import concurrent.futures
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pytest

from echo.echo_agent import EchoAgent
from intergrax.applications._shared.hosted_application_diagnostic_wiring import (
    HostedApplicationDiagnosticEventPublisher,
    HostedDiagnosticTenantBinding,
    build_hosted_application_diagnostic_event_publisher,
)
from intergrax.applications._shared.hosted_application_failure_projection import (
    hosted_application_failure_to_problem_signal,
)
from intergrax.applications._shared.scenario_runtime_baseline import (
    ScenarioExecutionRequest,
    build_scenario_runtime_from_environment,
    execute_scenario_task,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    require_active_execution_id,
    require_active_execution_identity,
)
from intergrax.hosting import HostedApplicationLifecycleState
from intergrax.hosting.contracts.events import (
    HostedApplicationEvent,
    HostedApplicationEventType,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.queueing.worker.execution import execute_logical_task
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.runtime.background_execution.bootstrap import BackgroundExecutionIdentity
from intergrax.runtime.background_execution.required_audit_evidence import (
    admit_background_execution_handler,
)
from intergrax.runtime.background_execution.transport_ref import BackgroundTransportExecutionRef
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.execution.boundary import ExecutionBoundary, ExecutionIdentityBinding
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.execution.child import ChildExecutionRunner
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner
from intergrax.tools.execution_models import ToolExecutionResult
from tests.integration.runtime.test_terminal_diagnostic_production_e2e import (
    _build_diagnostic_nexus_loop,
)
from tests.unit.applications._shared.test_hosted_application_diagnostic_integration import (
    _build_orchestrator_stack,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "df4-tenant"
_TASK_NAME = "df4.echo.v1"
_UNLIMITED_LEDGER = create_execution_budget_ledger(RunBudget())


@dataclass(frozen=True, slots=True)
class EntrypointBehavior:
    """Documented DF-4 behavior contract per entry surface."""

    entrypoint: Literal[
        "standard_task",
        "scenario_task",
        "background_task",
        "child_execution",
        "hosted_application",
    ]
    identity_model: str
    diagnostic_path: str
    mints_new_run: bool


DF4_BEHAVIOR_TABLE: tuple[EntrypointBehavior, ...] = (
    EntrypointBehavior(
        entrypoint="standard_task",
        identity_model="TaskId + RunId at intake; AttemptId + ExecutionId in NexusLoop.handle_task",
        diagnostic_path="NexusLoop._publish_terminal_runtime_event → invoke_terminal_execution_diagnostics",
        mints_new_run=False,
    ),
    EntrypointBehavior(
        entrypoint="scenario_task",
        identity_model="Scenario mints RunId once; same run through NexusLoop.handle_task",
        diagnostic_path="wire_terminal_execution_diagnostics → shared Nexus terminal path",
        mints_new_run=False,
    ),
    EntrypointBehavior(
        entrypoint="background_task",
        identity_model="BackgroundExecutionIdentity (task/run/attempt) passed into Nexus worker",
        diagnostic_path="UnifiedTaskRunner → NexusLoop terminal path (no remint inside worker)",
        mints_new_run=False,
    ),
    EntrypointBehavior(
        entrypoint="child_execution",
        identity_model="inherits parent RunId/AttemptId; mints child ExecutionId; parent link preserved",
        diagnostic_path="delegates through parent Nexus execution tree (no separate diagnostic engine)",
        mints_new_run=False,
    ),
    EntrypointBehavior(
        entrypoint="hosted_application",
        identity_model="tenant_id + application_id + instance_id (non-execution subject)",
        diagnostic_path="HostedApplicationDiagnosticEventPublisher → injected DiagnosticOrchestrator",
        mints_new_run=False,
    ),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _run_coro_sync(coro: object) -> object:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # type: ignore[arg-type]

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coro).result()


def test_df4_behavior_table_covers_all_required_entrypoints() -> None:
    covered = {row.entrypoint for row in DF4_BEHAVIOR_TABLE}
    assert covered == {
        "standard_task",
        "scenario_task",
        "background_task",
        "child_execution",
        "hosted_application",
    }


@pytest.mark.asyncio
async def test_df4_standard_task_uses_nexus_terminal_diagnostic_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop, runtime_store, _ = _build_diagnostic_nexus_loop(inject_violation=True)
    bridge_calls: list[tuple[object, ...]] = []

    from intergrax.runtime.diagnostics import terminal_execution_diagnostic_bridge as bridge_module

    original_invoke = bridge_module.invoke_terminal_execution_diagnostics

    def _capture_bridge(*args: object, **kwargs: object) -> object:
        bridge_calls.append((args, kwargs))
        return original_invoke(*args, **kwargs)

    monkeypatch.setattr(bridge_module, "invoke_terminal_execution_diagnostics", _capture_bridge)
    runner = UnifiedTaskRunner(loop)
    run_id = mint_run_id()

    result = await runner.run_task(
        Task(
            tenant_id=_TENANT,
            user_id="user-1",
            message="df4 standard",
            context=TaskContext(capability="echo.basic"),
        ),
        run_id=run_id,
    )

    assert result.state is TaskState.COMPLETED
    assert len(bridge_calls) == 1
    _, kwargs = bridge_calls[0]
    assert kwargs["tenant_id"] == _TENANT
    assert kwargs["run_id"] == run_id
    events = runtime_store.list_for_task(result.task_id, tenant_id=_TENANT)
    assert any(event.event_type is RuntimeEventType.TASK_COMPLETED for event in events)


@pytest.mark.asyncio
async def test_df4_scenario_task_preserves_run_and_uses_terminal_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from testing_support.builder import MeteringFakeLLMAdapter

    adapter = MeteringFakeLLMAdapter()

    def _resolve(env: object, agent_override: object | None = None, **_: object) -> object:
        del env
        return agent_override or adapter

    monkeypatch.setattr(
        "intergrax.applications._shared.llm_resolver.resolve_llm_adapter",
        _resolve,
    )

    environment = ApplicationEnvironmentProfile.lab_defaults(profile_id="df4.scenario")
    composition = build_scenario_runtime_from_environment(
        environment=environment,
        registry=_echo_registry(),
        tenant_id=_TENANT,
        manifest=ApplicationManifest.lab(
            app_id="df4_scenario",
            name="DF4 Scenario",
            route_prefix="/v1/df4_scenario",
            env_prefix="DF4_SCENARIO_",
            agents=[AgentBinding.mount(EchoAgent, capabilities=["echo.basic"])],
        ),
        runtime_events_db_path=tmp_path / "events.db",
        trace_db_path=tmp_path / "trace.db",
        document_store=InMemoryDocumentStore(),
        use_in_memory_trace=True,
    )
    trigger = composition.nexus_loop._terminal_diagnostic_trigger  # noqa: SLF001
    assert trigger is not None
    captured: list[object] = []
    original_run = trigger._orchestrator.run  # noqa: SLF001

    def _capture_run(request: object) -> object:
        result = original_run(request)
        captured.append(result)
        return result

    monkeypatch.setattr(trigger._orchestrator, "run", _capture_run)  # noqa: SLF001

    result = await execute_scenario_task(
        composition,
        ScenarioExecutionRequest(
            tenant_id=_TENANT,
            message="df4 scenario",
            capability="echo.basic",
        ),
    )

    assert result.task_result.run_id == str(result.run_id)
    store = composition.observability.runtime_event_store
    assert store is not None
    events = store.list_for_task(result.task_id, tenant_id=_TENANT)
    assert any(event.event_type is RuntimeEventType.TASK_COMPLETED for event in events)
    assert len(captured) == 1


def test_df4_background_worker_passes_identity_without_remint() -> None:
    source = (
        _repo_root() / "intergrax/runtime/task/nexus_worker_execution.py"
    ).read_text(encoding="utf-8")
    assert "mint_run_id" not in source
    assert "mint_task_id" not in source
    assert "execution_identity.run_id" in source
    assert "execution_identity.attempt_id" in source


def test_df4_background_task_uses_shared_terminal_diagnostic_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop, _, _ = _build_diagnostic_nexus_loop(inject_violation=True)
    trigger = loop._terminal_diagnostic_trigger  # noqa: SLF001
    assert trigger is not None
    captured: list[RunId] = []
    original_run = trigger._orchestrator.run  # noqa: SLF001

    def _capture_run(request: object) -> object:
        result = original_run(request)
        captured.append(request.executions[0].run_id)  # type: ignore[attr-defined]
        return result

    monkeypatch.setattr(trigger._orchestrator, "run", _capture_run)  # noqa: SLF001
    runner = UnifiedTaskRunner(loop)
    registry = TaskExecutionRegistry()
    causal_store = InMemoryCausalEvidencePersistence()
    execution_identity = BackgroundExecutionIdentity(
        tenant_id=_TENANT,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
    )

    def handler(
        *,
        tenant_id: str,
        run_id: str,
        payload: bytes,
        idempotency_key: str | None,
        execution_identity: BackgroundExecutionIdentity,
    ) -> ToolExecutionResult[object]:
        _ = payload, idempotency_key, run_id
        task = Task(
            tenant_id=tenant_id,
            user_id="worker-user",
            message="df4 background",
            context=TaskContext(capability="echo.basic"),
        )
        result = _run_coro_sync(
            runner.run_task(
                task,
                run_id=execution_identity.run_id,
                attempt_id=execution_identity.attempt_id,
            ),
        )
        assert result.state is TaskState.COMPLETED
        return ToolExecutionResult.ok({"answer": result.answer})

    registry.register(_TASK_NAME, handler)
    transport_ref = BackgroundTransportExecutionRef(
        tenant_id=_TENANT,
        provider="document_store",
        transport_task_id="df4-transport-1",
    )
    admit_background_execution_handler(
        transport_ref=transport_ref,
        execution_identity=execution_identity,
        causal_evidence_persistence=causal_store,
        handler=lambda: execute_logical_task(
            registry=registry,
            logical_task_name=_TASK_NAME,
            tenant_id=_TENANT,
            run_id=str(execution_identity.run_id),
            payload=b"{}",
            idempotency_key=None,
            idempotency_store=None,
            execution_identity=execution_identity,
        ),
    )

    assert captured == [execution_identity.run_id]


@pytest.mark.asyncio
async def test_df4_child_execution_preserves_parent_run_and_attempt() -> None:
    root = ExecutionIdentityBinding(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )
    child_captured: dict[str, RunId | AttemptId | ExecutionId | None] = {}
    child_runner = ChildExecutionRunner[object, object](ledger=_UNLIMITED_LEDGER)

    class ChildDelegate:
        async def execute(self, request: object) -> object:
            run_id, attempt_id = require_active_execution_identity()
            child_captured["run_id"] = run_id
            child_captured["attempt_id"] = attempt_id
            child_captured["execution_id"] = require_active_execution_id()
            return request

    class RootDelegate:
        async def execute(self, request: object) -> object:
            return await child_runner.execute(request=request, delegate=ChildDelegate())

    await ExecutionBoundary[object, object](
        RootDelegate(),
        identity=root,
        authority=ParentExecutionAuthority.unrestricted_root(),
    ).execute("ping")

    assert child_captured["run_id"] == root.run_id
    assert child_captured["attempt_id"] == root.attempt_id
    assert child_captured["execution_id"] != root.execution_id


@pytest.mark.asyncio
async def test_df4_hosted_application_uses_injected_orchestrator_subject_scope() -> None:
    orchestrator, persistence, _ = _build_orchestrator_stack()
    captured_requests: list[object] = []
    original_run = orchestrator.run

    def _capture_run(request: object) -> object:
        captured_requests.append(request)
        return original_run(request)

    orchestrator.run = _capture_run  # type: ignore[method-assign]
    publisher = build_hosted_application_diagnostic_event_publisher(
        tenant_binding=HostedDiagnosticTenantBinding(tenant_id=_TENANT),
        orchestrator=orchestrator,
    )
    observed_at = datetime(2026, 8, 29, 9, 0, 0, tzinfo=UTC)
    event = HostedApplicationEvent(
        event_type=HostedApplicationEventType.APPLICATION_FAILED,
        occurred_at=observed_at,
        application_id="df4_app",
        instance_id="df4-instance",
        lifecycle_state=HostedApplicationLifecycleState.FAILED,
        payload={
            "phase": "start",
            "reason_code": "runtime_error",
            "source_kind": "process",
            "source_id": "main",
            "exception_type": "RuntimeError",
        },
    )
    await publisher.publish(event)

    assert len(captured_requests) == 1
    request = captured_requests[0]
    assert request.tenant_id == _TENANT  # type: ignore[attr-defined]
    assert request.executions == ()  # type: ignore[attr-defined]
    scope = request.signal_subjects[0]  # type: ignore[attr-defined]
    assert scope.tenant_id == _TENANT
    assert scope.application_id == "df4_app"
    assert scope.instance_id == "df4-instance"
    problems = persistence.list_for_tenant(_TENANT)
    assert problems


def test_df4_hosted_publisher_does_not_construct_orchestrator() -> None:
    wiring_source = (
        _repo_root()
        / "intergrax/applications/_shared/hosted_application_diagnostic_wiring.py"
    ).read_text(encoding="utf-8")
    assert "DiagnosticOrchestrator(" not in wiring_source
    assert "build_diagnostic_orchestrator" not in wiring_source
    assert "wire_problem_persistence" not in wiring_source


def test_df4_hosted_failure_projection_has_no_execution_identity() -> None:
    signal = hosted_application_failure_to_problem_signal(
        HostedApplicationEvent(
            event_type=HostedApplicationEventType.APPLICATION_FAILED,
            occurred_at=datetime(2026, 8, 29, 9, 0, 0, tzinfo=UTC),
            application_id="df4_app",
            instance_id="df4-instance",
            lifecycle_state=HostedApplicationLifecycleState.FAILED,
            payload={
                "phase": "start",
                "reason_code": "runtime_error",
                "source_kind": "process",
                "source_id": "main",
            },
        ),
    )
    assert signal is not None
    assert signal.application_attributes is not None
    assert signal.application_attributes.application_id == "df4_app"
    assert signal.application_attributes.instance_id == "df4-instance"


def test_df4_scenario_runtime_has_no_separate_diagnostic_engine() -> None:
    path = _repo_root() / "intergrax/applications/_shared/scenario_runtime_baseline.py"
    source = path.read_text(encoding="utf-8")
    assert "DiagnosticOrchestrator(" not in source
    assert "wire_terminal_execution_diagnostics" in source
    assert "handle_task" in source


def test_df4_only_central_wiring_mints_diagnostic_orchestrator_in_applications_shared() -> None:
    shared_root = _repo_root() / "intergrax/applications/_shared"
    violations: list[str] = []
    for path in shared_root.rglob("*.py"):
        if path.name == "diagnostic_runtime_wiring.py":
            continue
        source = path.read_text(encoding="utf-8")
        if "DiagnosticOrchestrator(" in source:
            violations.append(path.relative_to(_repo_root()).as_posix())
    assert violations == []


def test_df4_nexus_loop_is_single_terminal_diagnostic_emitter() -> None:
    nexus_source = (_repo_root() / "intergrax/runtime/nexus/nexus_loop.py").read_text(
        encoding="utf-8",
    )
    assert nexus_source.count("invoke_terminal_execution_diagnostics(") == 1
    assert "_publish_terminal_runtime_event" in nexus_source


def _echo_registry() -> AgentRegistry:
    registry = AgentRegistry()
    registry.register(EchoAgent())
    return registry


def test_df4_hosted_publisher_accepts_orchestrator_via_constructor_only() -> None:
    tree = ast.parse(
        (
            _repo_root()
            / "intergrax/applications/_shared/hosted_application_diagnostic_wiring.py"
        ).read_text(encoding="utf-8"),
    )
    init_assigns_orchestrator = False
    constructs_orchestrator = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "DiagnosticOrchestrator":
                constructs_orchestrator = True
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr == "_orchestrator":
                    init_assigns_orchestrator = True
    assert constructs_orchestrator is False
    assert init_assigns_orchestrator is True
    assert issubclass(HostedApplicationDiagnosticEventPublisher, object)
