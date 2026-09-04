# © Artur Czarnecki. All rights reserved.

"""Production adoption E2E for terminal execution diagnostics (ONE-SPINE-3)."""

from __future__ import annotations

import asyncio
import concurrent.futures
from dataclasses import replace

import pytest

from echo.echo_agent import EchoAgent
from intergrax.applications._shared.diagnostic_read_wiring import (
    HostDiagnosticReadDependencies,
    build_diagnostic_read_service,
    resolve_host_diagnostic_read_dependencies,
)
from intergrax.applications._shared.diagnostic_runtime_wiring import (
    build_terminal_execution_diagnostic_trigger,
    resolve_host_diagnostic_runtime_dependencies,
)
from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications._shared.product_observability_dashboard_wiring import (
    _build_diagnostic_operations_pane,
)
from intergrax.applications._shared.harness_host_runtime_compat import (
    resolve_harness_host_nexus_loop_legacy,
)
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    reset_active_execution_identity,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.queueing.worker.execution import execute_logical_task
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.runtime.background_execution.bootstrap import BackgroundExecutionIdentity
from intergrax.runtime.background_execution.required_audit_evidence import (
    admit_background_execution_handler,
)
from intergrax.runtime.background_execution.transport_ref import BackgroundTransportExecutionRef
from intergrax.runtime.diagnostics.in_memory_problem_persistence import InMemoryProblemPersistence
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.observability_wiring import wire_nexus_observability
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner
from intergrax.tools.execution_models import ToolExecutionResult
from governed_contractor_application.host.environment_profile import (
    build_governed_contractor_environment_profile,
)
from governed_contractor_application.host.settings import GovernedContractorBackendSettings
from governed_contractor_application.manifest import build_governed_contractor_manifest
from governed_contractor_application.tests.governed_contractor_ac3_projection import (
    build_governed_contractor_test_registry_projection,
)
from tests.unit.applications.test_product_observability_dashboard_wiring import (
    _product_env,
)

pytestmark = [pytest.mark.integration, pytest.mark.gate]

_TENANT_A = "tenant-terminal-diag-a"
_TENANT_B = "tenant-terminal-diag-b"
_TASK_NAME = "terminal_diag.echo.v1"


def _run_coro_sync(coro: object) -> object:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # type: ignore[arg-type]

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coro).result()


class _FakeEnvWiring:
    def __init__(self, document_store: object) -> None:
        self.build_context = _FakeBuildContext(document_store)


class _FakeBuildContext:
    def __init__(self, document_store: object) -> None:
        self.tool_wiring_context = _FakeToolWiringContext(document_store)


class _FakeToolWiringContext:
    def __init__(self, document_store: object) -> None:
        self.document_store = document_store


class _HarnessRuntimeStub:
    def __init__(self, *, document_store: object, nexus_loop: NexusLoop) -> None:
        self.env_wiring = _FakeEnvWiring(document_store)
        self.observability = type(
            "_Obs",
            (),
            {"runtime_event_store": nexus_loop._runtime_event_store},  # noqa: SLF001
        )()
        self.nexus_loop = nexus_loop


def _inject_violation_after_completed(
    runtime_store: InMemoryRuntimeEventStore,
    *,
    violating_event_type: RuntimeEventType,
):
    def _handler(event: RuntimeEvent) -> None:
        if event.event_type is not RuntimeEventType.TASK_COMPLETED:
            return
        runtime_store.append(
            sample_runtime_event(
                tenant_id=event.tenant_id,
                task_id=event.task_id,
                run_id=event.run_id,
                attempt_id=event.attempt_id,
            ).model_copy(update={"event_type": violating_event_type}),
            tenant_id=event.tenant_id,
        )

    return _handler


def _build_diagnostic_nexus_loop(
    *,
    inject_violation: bool,
    violating_event_type: RuntimeEventType = RuntimeEventType.RETRY_SCHEDULED,
    problem_persistence: object | None = None,
) -> tuple[NexusLoop, InMemoryRuntimeEventStore, object]:
    document_store = InMemoryDocumentStore()
    runtime_store = InMemoryRuntimeEventStore()
    stores = wire_nexus_observability(
        use_in_memory_trace=True,
        runtime_event_store=runtime_store,
    )
    deps = resolve_host_diagnostic_runtime_dependencies(
        env_wiring=_FakeEnvWiring(document_store),
        observability=stores,
    )
    assert deps is not None
    if problem_persistence is not None:
        deps = replace(deps, problem_persistence=problem_persistence)
    trigger = build_terminal_execution_diagnostic_trigger(deps)

    registry = AgentRegistry()
    registry.register(EchoAgent())
    loop = NexusLoop(
        registry,
        trace_store=stores.trace_store,
        runtime_event_store=runtime_store,
    )
    loop.attach_terminal_diagnostic_trigger(trigger)
    if inject_violation:
        loop.event_bus.subscribe(
            _inject_violation_after_completed(
                runtime_store,
                violating_event_type=violating_event_type,
            ),
            event_types={RuntimeEventType.TASK_COMPLETED},
            priority=10,
        )
    return loop, runtime_store, deps.problem_persistence


@pytest.mark.asyncio
async def test_real_nexus_execution_triggers_diagnostics_without_manual_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop, _, _ = _build_diagnostic_nexus_loop(inject_violation=True)
    trigger = loop._terminal_diagnostic_trigger  # noqa: SLF001
    assert trigger is not None
    captured: list[object] = []
    original_run = trigger._orchestrator.run  # noqa: SLF001

    def _capture_run(request: object) -> object:
        result = original_run(request)
        captured.append(result)
        return result

    monkeypatch.setattr(trigger._orchestrator, "run", _capture_run)  # noqa: SLF001
    runner = UnifiedTaskRunner(loop)

    result = await runner.run_task(
        Task(
            tenant_id=_TENANT_A,
            user_id="user-1",
            message="terminal diagnostics",
            context=TaskContext(capability="echo.basic"),
        ),
        run_id=mint_run_id(),
    )

    assert result.state is TaskState.COMPLETED
    assert len(captured) == 1
    orchestration_result = captured[0]
    assert orchestration_result.execution_results[0].assessment.has_findings  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_clean_execution_does_not_create_problem() -> None:
    loop, _, persistence = _build_diagnostic_nexus_loop(inject_violation=False)
    runner = UnifiedTaskRunner(loop)

    result = await runner.run_task(
        Task(
            tenant_id=_TENANT_A,
            user_id="user-1",
            message="clean execution",
            context=TaskContext(capability="echo.basic"),
        ),
        run_id=mint_run_id(),
    )

    assert result.state is TaskState.COMPLETED
    assert query_all_problems_for_tenant(persistence, _TENANT_A) == ()


@pytest.mark.asyncio
async def test_evidence_recording_failure_does_not_change_business_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop, runtime_store, _ = _build_diagnostic_nexus_loop(inject_violation=False)
    trigger = loop._terminal_diagnostic_trigger  # noqa: SLF001
    assert trigger is not None

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("diagnostic persistence failed")

    def _raise_evidence(*_args: object, **_kwargs: object) -> None:
        raise OSError("evidence journal unavailable")

    monkeypatch.setattr(trigger, "trigger_for_terminal_execution", _raise)
    monkeypatch.setattr(
        "intergrax.runtime.diagnostics.terminal_execution_diagnostic_bridge.record_diagnostic_subsystem_failure",
        _raise_evidence,
    )
    runner = UnifiedTaskRunner(loop)
    run_id = mint_run_id()

    result = await runner.run_task(
        Task(
            tenant_id=_TENANT_A,
            user_id="user-1",
            message="evidence failure isolation",
            context=TaskContext(capability="echo.basic"),
        ),
        run_id=run_id,
    )

    assert result.state is TaskState.COMPLETED
    events = runtime_store.list_for_task(result.task_id, tenant_id=_TENANT_A)
    assert any(event.event_type is RuntimeEventType.TASK_COMPLETED for event in events)
    from intergrax.runtime.diagnostics.diagnostic_subsystem_failure_evidence import (
        diagnostic_subsystem_failure_observed_for_run,
    )

    assert not diagnostic_subsystem_failure_observed_for_run(
        runtime_store,
        tenant_id=_TENANT_A,
        run_id=run_id,
    )


@pytest.mark.asyncio
async def test_diagnostic_failure_does_not_change_business_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop, runtime_store, _ = _build_diagnostic_nexus_loop(inject_violation=False)
    trigger = loop._terminal_diagnostic_trigger  # noqa: SLF001
    assert trigger is not None

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("diagnostic persistence failed")

    monkeypatch.setattr(trigger, "trigger_for_terminal_execution", _raise)
    runner = UnifiedTaskRunner(loop)
    run_id = mint_run_id()

    result = await runner.run_task(
        Task(
            tenant_id=_TENANT_A,
            user_id="user-1",
            message="diagnostic failure isolation",
            context=TaskContext(capability="echo.basic"),
        ),
        run_id=run_id,
    )

    assert result.state is TaskState.COMPLETED
    events = runtime_store.list_for_task(result.task_id, tenant_id=_TENANT_A)
    assert any(event.event_type is RuntimeEventType.TASK_COMPLETED for event in events)
    from intergrax.runtime.diagnostics.diagnostic_subsystem_failure_evidence import (
        diagnostic_subsystem_failure_observed_for_run,
        is_diagnostic_subsystem_failure_event,
    )

    failure_events = [event for event in events if is_diagnostic_subsystem_failure_event(event)]
    assert len(failure_events) == 1
    failure = failure_events[0]
    assert failure.tenant_id == _TENANT_A
    assert failure.task_id == result.task_id
    assert failure.run_id == run_id
    assert failure.payload["error_type"] == "RuntimeError"
    assert diagnostic_subsystem_failure_observed_for_run(
        runtime_store,
        tenant_id=_TENANT_A,
        run_id=run_id,
    )


def test_background_execution_inherits_terminal_diagnostic_trigger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop, _, _ = _build_diagnostic_nexus_loop(inject_violation=True)
    trigger = loop._terminal_diagnostic_trigger  # noqa: SLF001
    assert trigger is not None
    captured: list[object] = []
    original_run = trigger._orchestrator.run  # noqa: SLF001

    def _capture_run(request: object) -> object:
        result = original_run(request)
        captured.append(result)
        return result

    monkeypatch.setattr(trigger._orchestrator, "run", _capture_run)  # noqa: SLF001
    runner = UnifiedTaskRunner(loop)
    registry = TaskExecutionRegistry()
    causal_store = InMemoryCausalEvidencePersistence()

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
            message="background echo",
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
        tenant_id=_TENANT_A,
        provider="document_store",
        transport_task_id="transport-task-1",
    )
    execution_identity = BackgroundExecutionIdentity(
        tenant_id=_TENANT_A,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
    )
    admit_background_execution_handler(
        transport_ref=transport_ref,
        execution_identity=execution_identity,
        causal_evidence_persistence=causal_store,
        handler=lambda: execute_logical_task(
            registry=registry,
            logical_task_name=_TASK_NAME,
            tenant_id=_TENANT_A,
            run_id=str(execution_identity.run_id),
            payload=b"{}",
            idempotency_key=None,
            idempotency_store=None,
            execution_identity=execution_identity,
        ),
    )

    assert len(captured) == 1
    assert captured[0].execution_results[0].assessment.has_findings  # type: ignore[attr-defined]


def test_background_execution_records_diagnostic_failure_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop, runtime_store, _ = _build_diagnostic_nexus_loop(inject_violation=False)
    trigger = loop._terminal_diagnostic_trigger  # noqa: SLF001
    assert trigger is not None

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("background diagnostic failed")

    monkeypatch.setattr(trigger, "trigger_for_terminal_execution", _raise)
    runner = UnifiedTaskRunner(loop)
    registry = TaskExecutionRegistry()
    causal_store = InMemoryCausalEvidencePersistence()
    execution_identity = BackgroundExecutionIdentity(
        tenant_id=_TENANT_A,
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
            message="background echo",
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
        tenant_id=_TENANT_A,
        provider="document_store",
        transport_task_id="transport-task-2",
    )
    admit_background_execution_handler(
        transport_ref=transport_ref,
        execution_identity=execution_identity,
        causal_evidence_persistence=causal_store,
        handler=lambda: execute_logical_task(
            registry=registry,
            logical_task_name=_TASK_NAME,
            tenant_id=_TENANT_A,
            run_id=str(execution_identity.run_id),
            payload=b"{}",
            idempotency_key=None,
            idempotency_store=None,
            execution_identity=execution_identity,
        ),
    )

    from intergrax.runtime.diagnostics.diagnostic_subsystem_failure_evidence import (
        diagnostic_subsystem_failure_observed_for_run,
    )

    assert diagnostic_subsystem_failure_observed_for_run(
        runtime_store,
        tenant_id=_TENANT_A,
        run_id=execution_identity.run_id,
    )


@pytest.mark.asyncio
async def test_separate_terminal_executions_reconcile_same_problem() -> None:
    loop, _, persistence = _build_diagnostic_nexus_loop(inject_violation=True)
    runner = UnifiedTaskRunner(loop)
    read_deps = HostDiagnosticReadDependencies(
        problem_persistence=persistence,
        runtime_event_persistence=loop._runtime_event_store,  # noqa: SLF001
        causal_evidence_persistence=InMemoryCausalEvidencePersistence(),
    )
    read_service = build_diagnostic_read_service(read_deps)

    await runner.run_task(
        Task(
            tenant_id=_TENANT_A,
            user_id="user-1",
            message="terminal diagnostics run a",
            context=TaskContext(capability="echo.basic"),
        ),
        run_id=mint_run_id(),
    )
    problems_after_a = query_all_problems_for_tenant(persistence, _TENANT_A)
    assert len(problems_after_a) == 1
    problem_id = problems_after_a[0].problem_id
    assert problems_after_a[0].occurrence_count == 1

    await runner.run_task(
        Task(
            tenant_id=_TENANT_A,
            user_id="user-1",
            message="terminal diagnostics run b",
            context=TaskContext(capability="echo.basic"),
        ),
        run_id=mint_run_id(),
    )
    problems_after_b = query_all_problems_for_tenant(persistence, _TENANT_A)
    assert len(problems_after_b) == 1
    assert problems_after_b[0].problem_id == problem_id
    assert problems_after_b[0].occurrence_count == 2

    listed = read_service.list_problems(tenant_id=_TENANT_A)
    assert listed.total_count == 1
    assert listed.problems[0].problem_id == problem_id
    assert listed.problems[0].occurrence_count == 2


@pytest.mark.asyncio
async def test_different_terminal_signatures_create_distinct_problems() -> None:
    shared_persistence = InMemoryProblemPersistence()
    loop_retry, _, _ = _build_diagnostic_nexus_loop(
        inject_violation=True,
        violating_event_type=RuntimeEventType.RETRY_SCHEDULED,
        problem_persistence=shared_persistence,
    )
    runner_retry = UnifiedTaskRunner(loop_retry)

    await runner_retry.run_task(
        Task(
            tenant_id=_TENANT_A,
            user_id="user-1",
            message="retry anomaly",
            context=TaskContext(capability="echo.basic"),
        ),
        run_id=mint_run_id(),
    )

    loop_failed, _, _ = _build_diagnostic_nexus_loop(
        inject_violation=True,
        violating_event_type=RuntimeEventType.TASK_FAILED,
        problem_persistence=shared_persistence,
    )
    runner_failed = UnifiedTaskRunner(loop_failed)

    await runner_failed.run_task(
        Task(
            tenant_id=_TENANT_A,
            user_id="user-1",
            message="failed anomaly",
            context=TaskContext(capability="echo.basic"),
        ),
        run_id=mint_run_id(),
    )

    problems = query_all_problems_for_tenant(shared_persistence, _TENANT_A)
    assert len(problems) == 2
    assert problems[0].problem_id != problems[1].problem_id


@pytest.mark.asyncio
async def test_replay_terminal_trigger_does_not_duplicate_failure_evidence() -> None:
    loop, runtime_store, _ = _build_diagnostic_nexus_loop(inject_violation=False)
    trigger = loop._terminal_diagnostic_trigger  # noqa: SLF001
    assert trigger is not None
    runner = UnifiedTaskRunner(loop)
    task = Task(
        tenant_id=_TENANT_A,
        user_id="user-1",
        message="replay failure evidence",
        context=TaskContext(capability="echo.basic"),
    )
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("diagnostic replay failed")

    trigger.trigger_for_terminal_execution = _raise  # type: ignore[method-assign]

    await runner.run_task(task, run_id=run_id, attempt_id=attempt_id)
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=mint_execution_id(),
    )
    try:
        await loop._publish_terminal_runtime_event(task)  # noqa: SLF001
        await loop._publish_terminal_runtime_event(task)  # noqa: SLF001
    finally:
        reset_active_execution_identity(token)

    from intergrax.runtime.diagnostics.diagnostic_subsystem_failure_evidence import (
        is_diagnostic_subsystem_failure_event,
    )

    failure_events = [
        event
        for event in runtime_store.list_for_run(run_id, tenant_id=_TENANT_A)
        if is_diagnostic_subsystem_failure_event(event)
    ]
    assert len(failure_events) == 1


@pytest.mark.asyncio
async def test_replay_terminal_trigger_does_not_duplicate_occurrence() -> None:
    loop, _, persistence = _build_diagnostic_nexus_loop(inject_violation=True)
    runner = UnifiedTaskRunner(loop)
    task = Task(
        tenant_id=_TENANT_A,
        user_id="user-1",
        message="replay",
        context=TaskContext(capability="echo.basic"),
    )
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()

    await runner.run_task(task, run_id=run_id, attempt_id=attempt_id)
    problems_before_replay = query_all_problems_for_tenant(persistence, _TENANT_A)
    assert len(problems_before_replay) == 1
    assert problems_before_replay[0].occurrence_count == 1
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=mint_execution_id(),
    )
    try:
        await loop._publish_terminal_runtime_event(task)  # noqa: SLF001
        await loop._publish_terminal_runtime_event(task)  # noqa: SLF001
    finally:
        reset_active_execution_identity(token)

    problems_after_replay = query_all_problems_for_tenant(persistence, _TENANT_A)
    assert len(problems_after_replay) == 1
    assert problems_after_replay[0].occurrence_count == 1


@pytest.mark.asyncio
async def test_tenant_isolation_for_terminal_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop, _, _ = _build_diagnostic_nexus_loop(inject_violation=True)
    trigger = loop._terminal_diagnostic_trigger  # noqa: SLF001
    assert trigger is not None
    captured_tenants: list[str] = []
    original_run = trigger._orchestrator.run  # noqa: SLF001

    def _capture_run(request: object) -> object:
        captured_tenants.append(request.tenant_id)  # type: ignore[attr-defined]
        return original_run(request)

    monkeypatch.setattr(trigger._orchestrator, "run", _capture_run)  # noqa: SLF001
    runner = UnifiedTaskRunner(loop)

    await runner.run_task(
        Task(
            tenant_id=_TENANT_A,
            user_id="user-a",
            message="tenant a",
            context=TaskContext(capability="echo.basic"),
        ),
        run_id=mint_run_id(),
    )
    await runner.run_task(
        Task(
            tenant_id=_TENANT_B,
            user_id="user-b",
            message="tenant b",
            context=TaskContext(capability="echo.basic"),
        ),
        run_id=mint_run_id(),
    )

    assert captured_tenants == [_TENANT_A, _TENANT_B]


def test_harness_host_runtime_wires_terminal_diagnostic_trigger(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: object,
) -> None:
    from testing_support.builder import MeteringFakeLLMAdapter

    adapter = MeteringFakeLLMAdapter()

    def _resolve(
        env: object,
        agent_override: object | None = None,
        **_: object,
    ) -> object:
        del env
        if agent_override is not None:
            return agent_override
        return adapter

    monkeypatch.setattr(
        "intergrax.applications._shared.llm_resolver.resolve_llm_adapter",
        _resolve,
    )

    document_store = InMemoryDocumentStore()
    settings = GovernedContractorBackendSettings.from_env()
    manifest = build_governed_contractor_manifest()
    env = manifest.environment or build_governed_contractor_environment_profile(settings)
    runtime = build_harness_host_runtime(
        manifest,
        env,
        settings=settings,
        registry_projection=build_governed_contractor_test_registry_projection(),
        document_store=document_store,
        trace_db_path=tmp_path / "trace.db",  # type: ignore[operator]
        runtime_events_db_path=tmp_path / "events.db",  # type: ignore[operator]
    )

    assert resolve_harness_host_nexus_loop_legacy(runtime)._terminal_diagnostic_trigger is not None  # noqa: SLF001


def test_dashboard_sees_problem_on_shared_persistence_after_runtime_trigger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from testing_support.builder import MeteringFakeLLMAdapter

    adapter = MeteringFakeLLMAdapter()

    def _resolve(
        env: object,
        agent_override: object | None = None,
        **_: object,
    ) -> object:
        del env
        if agent_override is not None:
            return agent_override
        return adapter

    monkeypatch.setattr(
        "intergrax.applications._shared.llm_resolver.resolve_llm_adapter",
        _resolve,
    )

    loop, _, persistence = _build_diagnostic_nexus_loop(inject_violation=True)
    env = _product_env()
    tenant_id = env.profile_id
    deps = HostDiagnosticReadDependencies(
        problem_persistence=persistence,
        runtime_event_persistence=loop._runtime_event_store,  # noqa: SLF001
        causal_evidence_persistence=InMemoryCausalEvidencePersistence(),
    )
    read_service = build_diagnostic_read_service(deps)
    runner = UnifiedTaskRunner(loop)
    _run_coro_sync(
        runner.run_task(
            Task(
                tenant_id=tenant_id,
                user_id="user-1",
                message="shared persistence proof",
                context=TaskContext(capability="echo.basic"),
            ),
            run_id=mint_run_id(),
        ),
    )
    _run_coro_sync(
        runner.run_task(
            Task(
                tenant_id=tenant_id,
                user_id="user-1",
                message="shared persistence proof recurrence",
                context=TaskContext(capability="echo.basic"),
            ),
            run_id=mint_run_id(),
        ),
    )

    problems = read_service.list_problems(tenant_id=tenant_id)
    assert problems.total_count == 1
    assert problems.problems[0].occurrence_count == 2

    pane = _build_diagnostic_operations_pane(env, read_service)
    assert pane is not None
    assert pane.ready is True
    assert pane.problem_count == 1
    assert pane.open_problem_count == 1
