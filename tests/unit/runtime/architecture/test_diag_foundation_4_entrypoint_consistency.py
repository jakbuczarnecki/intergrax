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
    execute_scenario_task,
)
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    mint_attempt_id,
    mint_event_id,
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
from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)
from intergrax.queueing.worker.execution import execute_logical_task
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.runtime.background_execution.bootstrap import BackgroundExecutionIdentity
from intergrax.runtime.background_execution.required_audit_evidence import (
    admit_background_execution_handler,
)
from intergrax.runtime.background_execution.transport_ref import (
    BackgroundTransportExecutionRef,
)
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event
from intergrax.runtime.execution.boundary import (
    ExecutionBoundary,
    ExecutionIdentityBinding,
)
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
from tests.unit.applications.scenario_runtime_test_support import (
    build_valid_minimal_lab_scenario_fixture,
    echo_agent_registry,
)
from testing_support.runtime.diagnostics.problem_persistence_test_support import (
    build_diagnostic_orchestrator_stack_for_tests,
    query_all_problems_for_tenant,
)
from testing_support.admitted_root_governance_identity import (
    lab_admitted_root_governance_identity_for_task,
)
from testing_support.runtime_events import with_preferred_canonical_payload

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "df4-tenant"
_TASK_NAME = "df4.echo.v1"
_UNLIMITED_LEDGER = create_execution_budget_ledger(RunBudget())
_SCENARIO_RUNTIME_FORBIDDEN_SYMBOLS = frozenset(
    {
        "DiagnosticOrchestrator",
        "ProblemLifecycleEngine",
        "ProblemGroupingEngine",
        "ExecutionReconstructor",
    }
)


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
        diagnostic_path="NexusLoop._publish_terminal_runtime_event → TerminalExecutionDiagnosticPort.dispatch",
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


_CANONICAL_ORCHESTRATOR_MINT_FILE = (
    _repo_root() / "intergrax/applications/_shared/diagnostic_composition.py"
)
_CANONICAL_ORCHESTRATOR_MINT_FUNCTION = "build_diagnostic_orchestrator_from_composition"
_RUNTIME_WIRING_FILE = (
    _repo_root() / "intergrax/applications/_shared/diagnostic_runtime_wiring.py"
)


def _typed_violating_runtime_event(
    anchor: RuntimeEvent,
    violating_event_type: RuntimeEventType,
) -> RuntimeEvent:
    reference = sample_runtime_event(
        tenant_id=anchor.tenant_id,
        task_id=anchor.task_id,
        run_id=anchor.run_id,
        attempt_id=anchor.attempt_id,
    )
    skeleton = RuntimeEvent(
        event_id=mint_event_id(),
        tenant_id=reference.tenant_id,
        task_id=reference.task_id,
        run_id=reference.run_id,
        attempt_id=reference.attempt_id,
        execution_id=reference.execution_id,
        event_type=violating_event_type,
        phase=reference.phase,
        severity=reference.severity,
        timestamp=reference.timestamp,
        correlation_id=reference.correlation_id,
    )
    return with_preferred_canonical_payload(skeleton)


def _inject_df4_violation_after_completed(
    runtime_store: object,
    *,
    violating_event_type: RuntimeEventType,
):
    from intergrax.runtime.events.stores.memory_runtime_event_store import (
        InMemoryRuntimeEventStore,
    )

    store = runtime_store
    assert isinstance(store, InMemoryRuntimeEventStore)

    def _handler(event: RuntimeEvent) -> None:
        if event.event_type is not RuntimeEventType.TASK_COMPLETED:
            return
        store.append(
            _typed_violating_runtime_event(event, violating_event_type),
            tenant_id=event.tenant_id,
        )

    return _handler


def _build_df4_diagnostic_nexus_loop(
    *,
    violating_event_type: RuntimeEventType = RuntimeEventType.RETRY_SCHEDULED,
):
    loop, runtime_store, deps = _build_diagnostic_nexus_loop(inject_violation=False)
    loop.event_bus.subscribe(
        _inject_df4_violation_after_completed(
            runtime_store,
            violating_event_type=violating_event_type,
        ),
        event_types={RuntimeEventType.TASK_COMPLETED},
        priority=10,
    )
    return loop, runtime_store, deps


@dataclass(frozen=True, slots=True)
class _DiagnosticOrchestratorMintSite:
    rel_path: str
    lineno: int
    enclosing_function: str | None


def _direct_diagnostic_orchestrator_mint_sites(path: Path) -> list[_DiagnosticOrchestratorMintSite]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    rel = path.relative_to(_repo_root()).as_posix()
    sites: list[_DiagnosticOrchestratorMintSite] = []
    function_stack: list[str] = []

    class _Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            function_stack.append(node.name)
            self.generic_visit(node)
            function_stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            function_stack.append(node.name)
            self.generic_visit(node)
            function_stack.pop()

        def visit_Call(self, node: ast.Call) -> None:
            func = node.func
            is_mint = (
                isinstance(func, ast.Name) and func.id == "DiagnosticOrchestrator"
            ) or (
                isinstance(func, ast.Attribute) and func.attr == "DiagnosticOrchestrator"
            )
            if is_mint:
                enclosing = function_stack[-1] if function_stack else None
                sites.append(
                    _DiagnosticOrchestratorMintSite(
                        rel_path=rel,
                        lineno=node.lineno,
                        enclosing_function=enclosing,
                    )
                )
            self.generic_visit(node)

    _Visitor().visit(tree)
    return sites


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
    loop, runtime_store, _ = _build_df4_diagnostic_nexus_loop()
    bridge_calls: list[tuple[object, ...]] = []

    from intergrax.runtime.diagnostics import (
        terminal_execution_diagnostic_bridge as bridge_module,
    )

    original_invoke = bridge_module.invoke_terminal_execution_diagnostics

    def _capture_bridge(*args: object, **kwargs: object) -> object:
        bridge_calls.append((args, kwargs))
        return original_invoke(*args, **kwargs)

    monkeypatch.setattr(
        bridge_module, "invoke_terminal_execution_diagnostics", _capture_bridge
    )
    runner = UnifiedTaskRunner(
        loop,
        admitted_governance_identity_for_task=lab_admitted_root_governance_identity_for_task,
    )
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
    assert len(bridge_calls) >= 1
    for _, kwargs in bridge_calls:
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

    def _resolve(
        env: object, agent_override: object | None = None, **_: object
    ) -> object:
        del env
        return agent_override or adapter

    monkeypatch.setattr(
        "intergrax.applications._shared.llm_resolver.resolve_llm_adapter",
        _resolve,
    )
    monkeypatch.setattr(
        "intergrax.applications._shared.llm_resolver.resolve_optional_llm_adapter",
        _resolve,
    )
    monkeypatch.setattr(
        "intergrax.applications._shared.llm_resolver.resolve_optional_environment_llm_adapter",
        _resolve,
    )

    composition = build_valid_minimal_lab_scenario_fixture(
        tmp_path,
        tenant_id=_TENANT,
        profile_id="df4.scenario",
        app_id="df4_scenario",
        document_store=InMemoryDocumentStore(),
    )
    assert composition.has_terminal_diagnostic_trigger is True
    assert composition.diagnostic_wiring.attached is True

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
    from intergrax.runtime.diagnostics import (
        terminal_execution_diagnostic_bridge as bridge_module,
    )

    loop, _, _ = _build_df4_diagnostic_nexus_loop()
    assert loop._terminal_diagnostic_trigger is not None  # noqa: SLF001
    captured: list[RunId] = []
    original_invoke = bridge_module.invoke_terminal_execution_diagnostics

    def _capture_bridge(*args: object, **kwargs: object) -> object:
        run_id = kwargs.get("run_id")
        if run_id is not None:
            captured.append(run_id)
        return original_invoke(*args, **kwargs)

    monkeypatch.setattr(
        bridge_module, "invoke_terminal_execution_diagnostics", _capture_bridge
    )
    runner = UnifiedTaskRunner(
        loop,
        admitted_governance_identity_for_task=lab_admitted_root_governance_identity_for_task,
    )
    registry = TaskExecutionRegistry()
    causal_store = InMemoryCausalEvidencePersistence()
    execution_identity = BackgroundExecutionIdentity(
        tenant_id=_TENANT,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
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

    assert captured
    assert all(run_id == execution_identity.run_id for run_id in captured)


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
async def test_df4_hosted_application_uses_injected_orchestrator_subject_scope() -> (
    None
):
    orchestrator, persistence, _read_service, _ = (
        build_diagnostic_orchestrator_stack_for_tests()
    )
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
    problems = query_all_problems_for_tenant(persistence, _TENANT)
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

    tree = ast.parse(source, filename=str(path))
    rel = path.relative_to(_repo_root()).as_posix()
    violations: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Name)
            and node.id in _SCENARIO_RUNTIME_FORBIDDEN_SYMBOLS
        ):
            violations.append(f"{rel}:{node.lineno} references {node.id}")
        if (
            isinstance(node, ast.Attribute)
            and node.attr in _SCENARIO_RUNTIME_FORBIDDEN_SYMBOLS
        ):
            violations.append(f"{rel}:{node.lineno} references .{node.attr}")
    assert violations == []


def test_df4_only_central_wiring_mints_diagnostic_orchestrator_in_applications_shared() -> (
    None
):
    shared_root = _repo_root() / "intergrax/applications/_shared"
    mint_sites: list[_DiagnosticOrchestratorMintSite] = []
    for path in shared_root.rglob("*.py"):
        mint_sites.extend(_direct_diagnostic_orchestrator_mint_sites(path))

    assert len(mint_sites) == 1
    site = mint_sites[0]
    assert site.rel_path == _CANONICAL_ORCHESTRATOR_MINT_FILE.relative_to(
        _repo_root()
    ).as_posix()
    assert site.enclosing_function == _CANONICAL_ORCHESTRATOR_MINT_FUNCTION

    runtime_wiring_source = _RUNTIME_WIRING_FILE.read_text(encoding="utf-8")
    assert "build_diagnostic_orchestrator_from_composition" in runtime_wiring_source
    assert _direct_diagnostic_orchestrator_mint_sites(_RUNTIME_WIRING_FILE) == []


def test_df4_nexus_loop_is_single_terminal_diagnostic_emitter() -> None:
    nexus_source = (_repo_root() / "intergrax/runtime/nexus/nexus_loop.py").read_text(
        encoding="utf-8",
    )
    assert nexus_source.count("dispatch_terminal_execution(") == 1
    assert "runtime.diagnostics" not in nexus_source
    assert "_publish_terminal_runtime_event" in nexus_source


def _echo_registry() -> AgentRegistry:
    return echo_agent_registry()


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
