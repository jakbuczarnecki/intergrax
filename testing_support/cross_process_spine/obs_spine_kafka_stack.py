# © Artur Czarnecki. All rights reserved.

"""Canonical OBS spine stack wiring for cross-process Kafka worker qualification."""

from __future__ import annotations

import asyncio
import concurrent.futures
from dataclasses import dataclass
from pathlib import Path

from echo.echo_agent import EchoAgent
from intergrax.applications._shared.diagnostic_read_wiring import HostDiagnosticReadDependencies
from intergrax.contracts.execution_identity import mint_attempt_id, mint_execution_id
from intergrax.distributed.providers.sqlite_kv_store import SqliteDistributedKVStore
from intergrax.queueing.worker.execution import execute_logical_task
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.runtime.background_execution.bootstrap import BackgroundExecutionIdentity
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.stores.sqlite_runtime_event_store import SQLiteRuntimeEventStore
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.observability.reconstruction import ExecutionReconstructor
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from intergrax.runtime.tools.in_memory_idempotency_store import InMemoryIdempotencyStore
from pydantic import BaseModel
from intergrax.tools.execution_models import ToolExecutionResult
from testing_support.cross_process_spine.durable_document_store import SqliteFileDocumentStore
from testing_support.obs_universal_spine.diagnostic_execution_stack import (
    build_diagnostic_nexus_loop,
    build_obs_spine_unified_task_runner,
)
from testing_support.runtime.diagnostics.problem_persistence_test_support import (
    TEST_DOCUMENT_STORE_CURSOR_SECRET,
    document_store_problem_persistence_for_tests,
    document_store_occurrence_persistence_for_tests,
)

_OBS_SPINE_TASK = "obs_spine.cross_process.kafka.v1"


class _KafkaEchoOutput(BaseModel):
    answer: str


def _run_coro_sync(coro: object) -> object:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # type: ignore[arg-type]
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coro).result()


@dataclass(frozen=True, slots=True)
class ObsSpineKafkaWorkerStack:
    registry: TaskExecutionRegistry
    kv_store: SqliteDistributedKVStore
    runtime_event_store: SQLiteRuntimeEventStore
    read_deps: HostDiagnosticReadDependencies
    causal_store: InMemoryCausalEvidencePersistence
    idempotency_store: InMemoryIdempotencyStore
    handler_invocation_counter: list[int]
    completed_task_ids: list[str]


def build_obs_spine_kafka_worker_stack(
    *,
    work_dir: Path,
    inject_violation: bool,
) -> ObsSpineKafkaWorkerStack:
    work_dir.mkdir(parents=True, exist_ok=True)
    document_store = SqliteFileDocumentStore(
        work_dir / "problems.docstore.sqlite",
        cursor_secret=TEST_DOCUMENT_STORE_CURSOR_SECRET,
    )
    problem_persistence = document_store_problem_persistence_for_tests(document_store)
    occurrence_persistence = document_store_occurrence_persistence_for_tests(document_store)
    runtime_store = SQLiteRuntimeEventStore(db_path=work_dir / "runtime_events.sqlite")
    loop, _, read_deps = build_diagnostic_nexus_loop(
        inject_violation=inject_violation,
        runtime_event_store=runtime_store,
        problem_persistence=problem_persistence,
        occurrence_persistence=occurrence_persistence,
        document_store=document_store,
        primary_agent=EchoAgent(),
    )
    runner = build_obs_spine_unified_task_runner(loop)
    registry = TaskExecutionRegistry()
    counter: list[int] = [0]
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
        counter[0] += 1
        task = Task(
            tenant_id=tenant_id,
            user_id="cross-process-worker",
            message="obs spine kafka cross-process",
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
        return ToolExecutionResult.ok(_KafkaEchoOutput(answer=str(result.answer)))

    registry.register(_OBS_SPINE_TASK, handler)
    return ObsSpineKafkaWorkerStack(
        registry=registry,
        kv_store=SqliteDistributedKVStore(work_dir / "worker_kv.sqlite"),
        runtime_event_store=runtime_store,
        read_deps=read_deps,
        causal_store=InMemoryCausalEvidencePersistence(),
        idempotency_store=InMemoryIdempotencyStore(),
        handler_invocation_counter=counter,
        completed_task_ids=completed_task_ids,
    )


def fresh_diagnostic_read_from_durable_backing(
    *,
    work_dir: Path,
    tenant_id: str,
    task_id: str,
    run_id: str,
) -> tuple[int, tuple[str, ...], bool, bool]:
    from intergrax.applications._shared.diagnostic_read_wiring import build_diagnostic_read_service

    document_store = SqliteFileDocumentStore(
        work_dir / "problems.docstore.sqlite",
        cursor_secret=TEST_DOCUMENT_STORE_CURSOR_SECRET,
    )
    problem_persistence = document_store_problem_persistence_for_tests(document_store)
    occurrence_persistence = document_store_occurrence_persistence_for_tests(document_store)
    runtime_store = SQLiteRuntimeEventStore(db_path=work_dir / "runtime_events.sqlite")
    loop, _, read_deps = build_diagnostic_nexus_loop(
        inject_violation=False,
        runtime_event_store=runtime_store,
        problem_persistence=problem_persistence,
        occurrence_persistence=occurrence_persistence,
        document_store=document_store,
    )
    _ = loop
    read_service = build_diagnostic_read_service(read_deps)
    problems = read_service.list_problems(tenant_id=tenant_id)
    reconstructor = ExecutionReconstructor(
        runtime_events=runtime_store,
        causal_evidence=InMemoryCausalEvidencePersistence(),
    )
    reconstruction = reconstructor.reconstruct_execution(tenant_id, task_id, run_id)
    problem_ids = tuple(problem.problem_id for problem in problems.problems)
    return (
        problems.total_count,
        problem_ids,
        reconstruction.has_runtime_events,
        reconstruction.is_runtime_history_complete,
    )


def terminal_event_type_for_run(
    *,
    runtime_store: SQLiteRuntimeEventStore,
    tenant_id: str,
    run_id: str,
) -> RuntimeEventType | None:
    terminal = {
        RuntimeEventType.TASK_COMPLETED,
        RuntimeEventType.TASK_FAILED,
    }
    for event in runtime_store.list_for_run(run_id, tenant_id=tenant_id):
        if event.event_type in terminal:
            return event.event_type
    return None


def mint_cross_process_execution_identity(
    *,
    tenant_id: str,
    run_id: str,
) -> BackgroundExecutionIdentity:
    return BackgroundExecutionIdentity(
        tenant_id=tenant_id,
        task_id="pending-transport",
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )


__all__ = [
    "ObsSpineKafkaWorkerStack",
    "build_obs_spine_kafka_worker_stack",
    "execute_logical_task",
    "fresh_diagnostic_read_from_durable_backing",
    "mint_cross_process_execution_identity",
    "terminal_event_type_for_run",
    "_OBS_SPINE_TASK",
]
