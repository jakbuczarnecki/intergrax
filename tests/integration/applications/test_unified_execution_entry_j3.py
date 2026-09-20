# © Artur Czarnecki. All rights reserved.

from __future__ import annotations
from intergrax.utils import attribute_access

import asyncio

import pytest
from echo.echo_agent import EchoAgent
from fastapi import BackgroundTasks

from intergrax.fastapi_core.context import RequestContext
from intergrax.fastapi_core.execution.models import ExecutionRequest
from intergrax.fastapi_core.runs.default_service import DefaultRunService
from intergrax.fastapi_core.runs.models import CreateRunRequest, RunStatus
from intergrax.queueing.providers.celery.celery_task_queue import CeleryTaskQueue
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.queued_host_task_execution_adapter import QueuedHostTaskExecutionAdapter
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from intergrax.runtime.task.task_run_bridge import task_to_execution_payload
from intergrax.runtime.task.worker_bootstrap import create_nexus_celery_worker_app
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from tests.unit.queueing.worker.dispatcher_test_kv import DispatcherTestKVStore
from intergrax.runtime.task.worker_payload import (
    decode_execution_request,
    encode_execution_request,
)
from testing_support.admitted_root_governance_identity import (
    lab_admitted_root_governance_identity_for_task,
)
from testing_support.builder import DummyRunStore

pytestmark = [pytest.mark.integration, pytest.mark.gate]


def _echo_celery_stack(*, wait_for_result: bool = True):
    registry = AgentRegistry()
    registry.register(EchoAgent())
    app = create_nexus_celery_worker_app(
        app_name="test_nexus_worker_j3",
        broker_url="memory://",
        backend_url="cache+memory://",
        agent_registry=registry,
        admit_root_governance_identity=lab_admitted_root_governance_identity_for_task,
        task_always_eager=True,
        kv_store=DispatcherTestKVStore(),
        causal_evidence_persistence=InMemoryCausalEvidencePersistence(),
        production_mode=False,
    )
    queue = CeleryTaskQueue(app)
    store = DummyRunStore()
    service = DefaultRunService(store, execution_adapter=None)
    adapter = QueuedHostTaskExecutionAdapter(
        queue,
        service,
        wait_for_result=wait_for_result,
    )
    service._execution_adapter = adapter
    return service, store, app


def test_worker_payload_roundtrip_execution_request() -> None:
    request = ExecutionRequest(
        run_id="run_payload_j3",
        tenant_id="t1",
        user_id="u1",
        input_payload=task_to_execution_payload(
            Task(
                tenant_id="t1",
                user_id="u1",
                message="payload roundtrip",
                context=TaskContext(capability="echo.basic"),
            )
        ),
    )
    restored = decode_execution_request(encode_execution_request(request))
    assert restored.run_id == request.run_id
    assert restored.input_payload["task"]["message"] == "payload roundtrip"


def test_queued_host_task_execution_adapter_runs_echo_via_celery_eager() -> None:
    service, store, _ = _echo_celery_stack()
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="worker queue echo",
        context=TaskContext(capability="echo.basic"),
    )
    context = RequestContext(
        request_id="req-j3",
        tenant_id="t1",
        user_id="u1",
        auth=None,
        path="/runs",
        method="POST",
    )
    background_tasks = BackgroundTasks()
    run = service.create_run(
        context,
        background_tasks,
        create_request=CreateRunRequest(payload=task_to_execution_payload(task)),
    )

    for bg_task in background_tasks.tasks:
        result = bg_task.func(*bg_task.args, **bg_task.kwargs)
        if asyncio.iscoroutine(result):
            asyncio.run(result)

    final = store.get(run.run_id)
    assert final.status == RunStatus.COMPLETED
    assert final.result_payload is not None
    assert "worker queue echo" in final.result_payload["answer"]
    assert final.result_payload["state"] == TaskState.COMPLETED.value


def test_worker_checkpoint_resume_via_queue_payload(tmp_path) -> None:
    from intergrax.contracts.execution_identity import mint_run_id
    from intergrax.contracts.human_approver import local_development_approver_evidence
    from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
    from intergrax.runtime.execution.continuation.persistence import (
        ExecutionContinuationDurableBacking,
        backing_execution_continuation_state_store,
    )
    from intergrax.runtime.nexus.nexus_loop import NexusLoop
    from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner
    from intergrax.runtime.task.task_contract import (
        TaskExecutionOptions,
        TaskHumanInput,
        TaskLongRunningOptions,
    )
    from testing_support.admitted_root_governance_identity import (
        lab_admitted_root_governance_identity_for_task,
    )
    from testing_support.nexus_hitl_test_agent import (
        NexusBasicHitlTestAgent,
        prepare_nexus_hitl_resume_task,
    )

    class _HitlAgent(NexusBasicHitlTestAgent):
        def __init__(self) -> None:
            super().__init__(
                track_step_runs=True,
                human_request_id="hr_worker_j3",
                step_summary="review",
            )

    NexusBasicHitlTestAgent.step_run_count = 0
    registry = AgentRegistry()
    registry.register(_HitlAgent())
    checkpoint_store = SQLiteTaskCheckpointStore(db_path=tmp_path / "worker_ckpt.db")
    continuation_backing = ExecutionContinuationDurableBacking()
    continuation_store = backing_execution_continuation_state_store(continuation_backing)

    setup_loop = NexusLoop(
        registry,
        checkpoint_store=checkpoint_store,
        execution_continuation_state_store=continuation_store,
    )
    runner = UnifiedTaskRunner(
        setup_loop,
        admitted_governance_identity_for_task=lab_admitted_root_governance_identity_for_task,
    )
    run_id = mint_run_id()
    paused = asyncio.run(
        runner.run_task(
            Task(
                tenant_id="t1",
                user_id="u1",
                message="worker resume",
                context=TaskContext(capability="hitl.basic"),
                options=TaskExecutionOptions(
                    long_running=TaskLongRunningOptions(enabled=True),
                ),
            ),
            run_id=run_id,
        )
    )
    token = paused.summary.resume_token
    assert token
    assert paused.state == TaskState.WAITING_FOR_HUMAN
    assert NexusBasicHitlTestAgent.step_run_count == 1
    # Worker B simulation: class-level counter must not leak worker A step executions.
    NexusBasicHitlTestAgent.step_run_count = 0

    app = create_nexus_celery_worker_app(
        app_name="test_nexus_worker_resume_j3",
        broker_url="memory://",
        backend_url="cache+memory://",
        agent_registry=registry,
        checkpoint_store=checkpoint_store,
        execution_continuation_state_store=continuation_store,
        admit_root_governance_identity=lab_admitted_root_governance_identity_for_task,
        task_always_eager=True,
        kv_store=DispatcherTestKVStore(),
        causal_evidence_persistence=InMemoryCausalEvidencePersistence(),
        production_mode=False,
    )
    queue = CeleryTaskQueue(app)
    store = DummyRunStore()
    service = DefaultRunService(store, execution_adapter=None)
    adapter = QueuedHostTaskExecutionAdapter(queue, service, wait_for_result=True)
    service._execution_adapter = adapter

    resumed_task = Task(
        task_id=paused.task_id,
        tenant_id="t1",
        user_id="u1",
        message="worker resume",
        context=TaskContext(capability="hitl.basic"),
        options=TaskExecutionOptions(
            long_running=TaskLongRunningOptions(enabled=True, resume_token=token),
            human=TaskHumanInput(response_text="approve", verdict="approve"),
        ),
        metadata={"human_approved": True, "resume_token": token},
    )
    loaded = checkpoint_store.get_latest(paused.task_id, "t1")
    assert loaded is not None
    prepare_nexus_hitl_resume_task(
        resumed_task,
        loaded=loaded,
        run_id=run_id,
        human_approved=True,
    )
    context = RequestContext(
        request_id="req-j3-resume",
        tenant_id="t1",
        user_id="u1",
        auth=None,
        path="/runs",
        method="POST",
    )
    background_tasks = BackgroundTasks()
    run = service.create_run(
        context,
        background_tasks,
        create_request=CreateRunRequest(payload=task_to_execution_payload(resumed_task)),
    )
    for bg_task in background_tasks.tasks:
        result = bg_task.func(*bg_task.args, **bg_task.kwargs)
        if asyncio.iscoroutine(result):
            asyncio.run(result)

    final = store.get(run.run_id)
    assert final.status == RunStatus.COMPLETED
    assert final.result_payload is not None
    assert final.result_payload["state"] == TaskState.COMPLETED.value
    assert NexusBasicHitlTestAgent.step_run_count == 1
