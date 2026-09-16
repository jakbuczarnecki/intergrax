# © Artur Czarnecki. All rights reserved.

"""OBS-UNIVERSAL-SPINE-E2E — async worker ingress through canonical diagnostics."""

from __future__ import annotations

import asyncio
import concurrent.futures

import pytest

from intergrax.applications._shared.diagnostic_read_wiring import build_diagnostic_read_service
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
from testing_support.obs_universal_spine.diagnostic_execution_stack import build_diagnostic_nexus_loop

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
    runner = UnifiedTaskRunner(loop)
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
    runner = UnifiedTaskRunner(loop)
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
    runner = UnifiedTaskRunner(loop)
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
    runner = UnifiedTaskRunner(loop)
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
