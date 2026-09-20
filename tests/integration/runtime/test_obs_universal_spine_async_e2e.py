# © Artur Czarnecki. All rights reserved.

"""OBS-UNIVERSAL-SPINE-E2E — async worker ingress through canonical diagnostics."""

from __future__ import annotations

import asyncio
import concurrent.futures

import pytest

from intergrax.applications._shared.diagnostic_read_wiring import build_diagnostic_read_service
from intergrax.applications._shared.diagnostic_runtime_wiring import (
    resolve_host_diagnostic_runtime_dependencies,
)
from intergrax.contracts.execution_identity import mint_attempt_id, mint_execution_id, mint_run_id, mint_task_id
from intergrax.queueing.worker.execution import execute_logical_task
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.runtime.background_execution.bootstrap import BackgroundExecutionIdentity
from intergrax.runtime.background_execution.required_audit_evidence import admit_background_execution_handler
from intergrax.runtime.background_execution.transport_ref import BackgroundTransportExecutionRef
from intergrax.runtime.diagnostics.persistence_conformance import query_all_problems_for_tenant
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.observability.reconstruction import ExecutionReconstructor
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner
from intergrax.runtime.tools.in_memory_idempotency_store import InMemoryIdempotencyStore
from intergrax.tools.execution_models import ToolExecutionResult
from testing_support.obs_universal_spine.diagnostic_execution_stack import (
    build_diagnostic_nexus_loop,
    build_obs_spine_unified_task_runner,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gate,
    pytest.mark.obs_coverage_p1,
]

_TENANT_A = "tenant-obs-spine-async-a"
_TENANT_B = "tenant-obs-spine-async-b"
_TASK_NAME = "obs_spine.async.echo.v1"


def _run_coro_sync(coro: object) -> object:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # type: ignore[arg-type]
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coro).result()


def _register_echo_handler(
    runner: UnifiedTaskRunner,
    registry: TaskExecutionRegistry,
) -> list[str]:
    completed_task_ids: list[str] = []

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
            message="async spine echo",
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
        completed_task_ids.append(result.task_id)
        return ToolExecutionResult.ok({"answer": result.answer})

    registry.register(_TASK_NAME, handler)
    return completed_task_ids


def _worker_invoke(
    *,
    registry: TaskExecutionRegistry,
    execution_identity: BackgroundExecutionIdentity,
    causal_store: InMemoryCausalEvidencePersistence,
    idempotency_store: InMemoryIdempotencyStore | None,
    idempotency_key: str | None,
) -> None:
    transport_ref = BackgroundTransportExecutionRef(
        tenant_id=execution_identity.tenant_id,
        provider="in_process_transport",
        transport_task_id=f"transport-{idempotency_key or 'direct'}",
    )
    admit_background_execution_handler(
        transport_ref=transport_ref,
        execution_identity=execution_identity,
        causal_evidence_persistence=causal_store,
        handler=lambda: execute_logical_task(
            registry=registry,
            logical_task_name=_TASK_NAME,
            tenant_id=execution_identity.tenant_id,
            run_id=str(execution_identity.run_id),
            payload=b"{}",
            idempotency_key=idempotency_key,
            idempotency_store=idempotency_store,
            execution_identity=execution_identity,
        ),
    )


def test_async_worker_success_no_false_problem() -> None:
    loop, runtime_store, read_deps = build_diagnostic_nexus_loop(inject_violation=False)
    runner = build_obs_spine_unified_task_runner(loop)
    registry = TaskExecutionRegistry()
    completed_task_ids = _register_echo_handler(runner, registry)
    causal_store = InMemoryCausalEvidencePersistence()
    execution_identity = BackgroundExecutionIdentity(
        tenant_id=_TENANT_A,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )

    _worker_invoke(
        registry=registry,
        execution_identity=execution_identity,
        causal_store=causal_store,
        idempotency_store=None,
        idempotency_key=None,
    )

    assert len(completed_task_ids) == 1
    events = runtime_store.list_for_run(execution_identity.run_id, tenant_id=_TENANT_A)
    assert any(event.event_type is RuntimeEventType.TASK_COMPLETED for event in events)
    reconstructor = ExecutionReconstructor(
        runtime_events=runtime_store,
        causal_evidence=causal_store,
    )
    reconstruction = reconstructor.reconstruct_execution(
        _TENANT_A,
        completed_task_ids[0],
        execution_identity.run_id,
    )
    assert reconstruction.has_runtime_events
    assert reconstruction.is_runtime_history_complete
    assert query_all_problems_for_tenant(read_deps.problem_persistence, _TENANT_A) == ()
    read_service = build_diagnostic_read_service(read_deps)
    assert read_service.list_problems(tenant_id=_TENANT_A).total_count == 0


def test_async_worker_failure_creates_problem_and_read_model() -> None:
    loop, runtime_store, read_deps = build_diagnostic_nexus_loop(inject_violation=True)
    runner = build_obs_spine_unified_task_runner(loop)
    registry = TaskExecutionRegistry()
    completed_task_ids = _register_echo_handler(runner, registry)
    causal_store = InMemoryCausalEvidencePersistence()
    execution_identity = BackgroundExecutionIdentity(
        tenant_id=_TENANT_A,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )

    _worker_invoke(
        registry=registry,
        execution_identity=execution_identity,
        causal_store=causal_store,
        idempotency_store=None,
        idempotency_key=None,
    )

    assert len(completed_task_ids) == 1
    problems = query_all_problems_for_tenant(read_deps.problem_persistence, _TENANT_A)
    assert len(problems) == 1
    read_service = build_diagnostic_read_service(read_deps)
    detail = read_service.get_problem(tenant_id=_TENANT_A, problem_id=problems[0].problem_id)
    assert detail is not None
    assert detail.tenant_id == _TENANT_A
    events = runtime_store.list_for_run(execution_identity.run_id, tenant_id=_TENANT_A)
    assert any(event.event_type is RuntimeEventType.TASK_COMPLETED for event in events)


def test_async_duplicate_delivery_idempotent_single_execution() -> None:
    loop, runtime_store, read_deps = build_diagnostic_nexus_loop(inject_violation=False)
    runner = build_obs_spine_unified_task_runner(loop)
    registry = TaskExecutionRegistry()
    run_count = 0
    completed_task_ids: list[str] = []

    def handler(
        *,
        tenant_id: str,
        run_id: str,
        payload: bytes,
        idempotency_key: str | None,
        execution_identity: BackgroundExecutionIdentity,
    ) -> ToolExecutionResult[object]:
        nonlocal run_count
        _ = payload, idempotency_key, run_id
        run_count += 1
        task = Task(
            tenant_id=tenant_id,
            user_id="worker-user",
            message="idempotent async",
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
        completed_task_ids.append(result.task_id)
        return ToolExecutionResult.ok({"answer": result.answer})

    registry.register(_TASK_NAME, handler)
    causal_store = InMemoryCausalEvidencePersistence()
    idempotency_store = InMemoryIdempotencyStore()
    execution_identity = BackgroundExecutionIdentity(
        tenant_id=_TENANT_A,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )
    idem_key = "delivery-1"

    _worker_invoke(
        registry=registry,
        execution_identity=execution_identity,
        causal_store=causal_store,
        idempotency_store=idempotency_store,
        idempotency_key=idem_key,
    )
    _worker_invoke(
        registry=registry,
        execution_identity=execution_identity,
        causal_store=causal_store,
        idempotency_store=idempotency_store,
        idempotency_key=idem_key,
    )

    assert run_count == 1
    assert len(completed_task_ids) == 1
    completed_events = [
        event
        for event in runtime_store.list_for_task(completed_task_ids[0], tenant_id=_TENANT_A)
        if event.event_type is RuntimeEventType.TASK_COMPLETED
    ]
    assert len(completed_events) >= 1
    assert query_all_problems_for_tenant(read_deps.problem_persistence, _TENANT_A) == ()


def test_async_tenant_isolation_for_diagnostic_state() -> None:
    loop, _, read_deps = build_diagnostic_nexus_loop(inject_violation=True)
    runner = build_obs_spine_unified_task_runner(loop)
    registry = TaskExecutionRegistry()
    _register_echo_handler(runner, registry)
    causal_store = InMemoryCausalEvidencePersistence()

    for tenant_id in (_TENANT_A, _TENANT_B):
        execution_identity = BackgroundExecutionIdentity(
            tenant_id=tenant_id,
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
        )
        _worker_invoke(
            registry=registry,
            execution_identity=execution_identity,
            causal_store=causal_store,
            idempotency_store=None,
            idempotency_key=None,
        )

    assert len(query_all_problems_for_tenant(read_deps.problem_persistence, _TENANT_A)) == 1
    assert len(query_all_problems_for_tenant(read_deps.problem_persistence, _TENANT_B)) == 1
    read_service = build_diagnostic_read_service(read_deps)
    assert read_service.list_problems(tenant_id=_TENANT_A).total_count == 1
    assert read_service.list_problems(tenant_id=_TENANT_B).total_count == 1
    assert (
        read_service.list_problems(tenant_id=_TENANT_A).problems[0].problem_id
        != read_service.list_problems(tenant_id=_TENANT_B).problems[0].problem_id
    )


def test_async_worker_natural_failure_creates_problem_and_terminal_event() -> None:
    from intergrax.agents.harness_reference_agent import HarnessReferenceAgent
    from intergrax.contracts.agent_contract_meta import AgentContract
    from intergrax.contracts.agent_step import AgentStep, StepOutput
    from intergrax.contracts.capability import CapabilityMatchResult
    from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
    from intergrax.runtime.nexus.config import RuntimeConfig
    from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
    from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
    from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager

    class _FailingAgent(HarnessReferenceAgent):
        def get_contract(self) -> AgentContract:
            return AgentContract(
                id="failing",
                name="Failing",
                description="fails in run_step",
                capabilities=["obs_spine.fail"],
                max_steps=1,
            )

        def can_handle(self, task_context: object) -> CapabilityMatchResult:
            return CapabilityMatchResult(
                matched=True,
                agent_id="failing",
                matched_capabilities=["obs_spine.fail"],
                score=1.0,
            )

        def build_context(self, request: RuntimeRequest) -> RuntimeContext:
            config = RuntimeConfig(
                llm_adapter=FakeLLMAdapter(fixed_text="ok"),
                enable_rag=False,
                production_mode=False,
                tenant_id=request.tenant_id,
            )
            return RuntimeContext.build(
                config=config,
                session_manager=build_in_memory_session_manager(),
            )

        def get_steps(self) -> list[AgentStep]:
            return [AgentStep(step_id="fail", step_name="fail", step_index=0)]

        async def run_step(self, step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
            raise RuntimeError("natural worker failure")

    loop, runtime_store, read_deps = build_diagnostic_nexus_loop(
        inject_violation=False,
        primary_agent=_FailingAgent(),
    )
    runner = build_obs_spine_unified_task_runner(loop)
    registry = TaskExecutionRegistry()
    completed_task_ids: list[str] = []

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
            message="natural failure",
            context=TaskContext(capability="obs_spine.fail"),
        )
        try:
            result = _run_coro_sync(
                runner.run_task(
                    task,
                    run_id=execution_identity.run_id,
                    attempt_id=execution_identity.attempt_id,
                ),
            )
        except RuntimeError:
            events = runtime_store.list_for_run(
                execution_identity.run_id,
                tenant_id=tenant_id,
            )
            assert any(event.event_type is RuntimeEventType.TASK_FAILED for event in events)
            return ToolExecutionResult.fail("natural_failure", "natural worker failure")
        completed_task_ids.append(result.task_id)
        assert result.state is TaskState.FAILED
        return ToolExecutionResult.ok({"state": result.state.value})

    registry.register(_TASK_NAME, handler)
    causal_store = InMemoryCausalEvidencePersistence()
    execution_identity = BackgroundExecutionIdentity(
        tenant_id=_TENANT_A,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )
    transport_ref = BackgroundTransportExecutionRef(
        tenant_id=execution_identity.tenant_id,
        provider="in_process_transport",
        transport_task_id="transport-natural-failure",
    )
    admit_background_execution_handler(
        transport_ref=transport_ref,
        execution_identity=execution_identity,
        causal_evidence_persistence=causal_store,
        handler=lambda: execute_logical_task(
            registry=registry,
            logical_task_name=_TASK_NAME,
            tenant_id=execution_identity.tenant_id,
            run_id=str(execution_identity.run_id),
            payload=b"{}",
            idempotency_key=None,
            idempotency_store=None,
            execution_identity=execution_identity,
        ),
    )

    events = runtime_store.list_for_run(execution_identity.run_id, tenant_id=_TENANT_A)
    assert any(event.event_type is RuntimeEventType.TASK_FAILED for event in events)
    problems = query_all_problems_for_tenant(read_deps.problem_persistence, _TENANT_A)
    if problems:
        read_service = build_diagnostic_read_service(read_deps)
        assert read_service.get_problem(tenant_id=_TENANT_A, problem_id=problems[0].problem_id) is not None


def test_async_worker_shared_harness_host_composition_root(tmp_path) -> None:
    from echo.echo_agent import EchoAgent
    from intergrax.applications._shared.harness_host_composition import resolve_harness_host_nexus_loop
    from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
    from intergrax.applications.contracts.environment_profile import (
        ApplicationEnvironmentProfile,
        DiagnosticPosture,
    )
    from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
    from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore

    manifest = ApplicationManifest.lab(
        app_id="obs_spine_async_harness",
        name="OBS Spine Async Harness",
        route_prefix="/v1/obs_spine_async",
        env_prefix="OBS_SPINE_ASYNC_",
        agents=[AgentBinding.mount(EchoAgent, contract_id="echo", capabilities=["echo.basic"])],
    )
    environment = ApplicationEnvironmentProfile.lab_defaults(profile_id="obs_spine_async.lab")
    environment = environment.model_copy(
        update={
            "diagnostic_profile": environment.diagnostic_profile.model_copy(
                update={"posture": DiagnosticPosture.REQUIRED},
            ),
        },
    )
    host_runtime = build_harness_host_runtime(
        manifest,
        environment,
        use_in_memory_trace=False,
        document_store=InMemoryDocumentStore(),
        runtime_events_db_path=tmp_path / "harness_events.db",
        trace_db_path=tmp_path / "harness_trace.db",
    )
    assert host_runtime.diagnostic_wiring.attached is True
    nexus_loop = resolve_harness_host_nexus_loop(host_runtime)
    runtime_store = host_runtime.observability.runtime_event_store
    assert runtime_store is not None
    read_deps = resolve_host_diagnostic_runtime_dependencies(
        env_wiring=host_runtime.env_wiring,
        observability=host_runtime.observability,
    )
    assert read_deps is not None

    runner = build_obs_spine_unified_task_runner(nexus_loop)
    registry = TaskExecutionRegistry()
    completed_task_ids = _register_echo_handler(runner, registry)
    causal_store = InMemoryCausalEvidencePersistence()
    execution_identity = BackgroundExecutionIdentity(
        tenant_id=_TENANT_A,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )
    _worker_invoke(
        registry=registry,
        execution_identity=execution_identity,
        causal_store=causal_store,
        idempotency_store=None,
        idempotency_key="harness-root",
    )
    assert len(completed_task_ids) == 1
    assert completed_task_ids[0] != str(execution_identity.task_id)
    events = runtime_store.list_for_run(execution_identity.run_id, tenant_id=_TENANT_A)
    completed = [event for event in events if event.event_type is RuntimeEventType.TASK_COMPLETED]
    assert completed
    assert completed[0].run_id == execution_identity.run_id
    assert completed[0].attempt_id == execution_identity.attempt_id
    reconstructor = ExecutionReconstructor(
        runtime_events=runtime_store,
        causal_evidence=causal_store,
    )
    reconstruction = reconstructor.reconstruct_execution(
        _TENANT_A,
        completed_task_ids[0],
        execution_identity.run_id,
    )
    assert reconstruction.has_runtime_events
    assert query_all_problems_for_tenant(read_deps.problem_persistence, _TENANT_A) == ()


def test_async_worker_identity_correlation_matrix() -> None:
    loop, runtime_store, _read_deps = build_diagnostic_nexus_loop(inject_violation=False)
    runner = build_obs_spine_unified_task_runner(loop)
    registry = TaskExecutionRegistry()
    completed_task_ids = _register_echo_handler(runner, registry)
    causal_store = InMemoryCausalEvidencePersistence()
    execution_identity = BackgroundExecutionIdentity(
        tenant_id=_TENANT_A,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )
    transport_ref = BackgroundTransportExecutionRef(
        tenant_id=execution_identity.tenant_id,
        provider="in_process_transport",
        transport_task_id="transport-identity-matrix",
    )
    _worker_invoke(
        registry=registry,
        execution_identity=execution_identity,
        causal_store=causal_store,
        idempotency_store=None,
        idempotency_key="identity-matrix",
    )
    assert transport_ref.transport_task_id == "transport-identity-matrix"
    assert len(completed_task_ids) == 1
    events = runtime_store.list_for_run(execution_identity.run_id, tenant_id=_TENANT_A)
    terminal = next(event for event in events if event.event_type is RuntimeEventType.TASK_COMPLETED)
    assert terminal.run_id == execution_identity.run_id
    assert terminal.attempt_id == execution_identity.attempt_id
    assert terminal.task_id == completed_task_ids[0]
