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
from intergrax.runtime.task.queued_nexus_execution_adapter import QueuedNexusExecutionAdapter
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from intergrax.runtime.task.task_run_bridge import task_to_execution_payload
from intergrax.runtime.task.worker_bootstrap import create_nexus_celery_worker_app
from intergrax.runtime.task.worker_payload import (
    decode_execution_request,
    encode_execution_request,
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
        task_always_eager=True,
    )
    queue = CeleryTaskQueue(app)
    store = DummyRunStore()
    service = DefaultRunService(store, execution_adapter=None)
    adapter = QueuedNexusExecutionAdapter(
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


def test_queued_nexus_execution_adapter_runs_echo_via_celery_eager() -> None:
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
    from intergrax.agents.agent_contract import Agent
    from intergrax.contracts.agent_contract_meta import AgentContract
    from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType, HumanRequest
    from intergrax.contracts.agent_step import AgentStep, StepOutput
    from intergrax.contracts.capability import CapabilityMatchResult
    from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
    from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
    from intergrax.runtime.nexus.config import RuntimeConfig
    from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
    from intergrax.runtime.nexus.nexus_loop import NexusLoop
    from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
    from intergrax.runtime.task.task_contract import (
        TaskExecutionOptions,
        TaskHumanInput,
        TaskLongRunningOptions,
    )
    from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager

    class _HitlAgent(Agent):
        runs = 0

        def get_contract(self) -> AgentContract:
            return AgentContract(
                id="hitl",
                name="HITL",
                description="hitl",
                capabilities=["hitl.basic"],
                max_steps=2,
            )

        def can_handle(self, task_context: object) -> CapabilityMatchResult:
            cap = attribute_access.optional(task_context, "capability", None)
            if cap in (None, "hitl.basic"):
                return CapabilityMatchResult(
                    matched=True,
                    agent_id="hitl",
                    matched_capabilities=["hitl.basic"],
                    score=1.0,
                )
            return CapabilityMatchResult(matched=False, rationale="no")

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

        def get_steps(self, context: RuntimeContext) -> list[AgentStep]:
            _ = context
            return [AgentStep(step_id="review", step_name="review", step_index=0)]

        async def run_step(self, step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
            _HitlAgent.runs += 1
            return StepOutput(step_id=step.step_id, summary="review")

        def decide_after_step(
            self,
            step: AgentStep,
            output: StepOutput | None,
            ctx: RuntimeExecutionContext,
        ) -> AgentDecision:
            _ = step, output
            if ctx.request and ctx.request.metadata.get("human_approved"):
                return AgentDecision(type=AgentDecisionType.COMPLETE, reason="approved")
            return AgentDecision(
                type=AgentDecisionType.REQUEST_HUMAN,
                reason="approval required",
                human_request=HumanRequest(
                    request_id="hr_worker_j3",
                    prompt="Approve?",
                    options=["approve", "reject"],
                ),
            )

    _HitlAgent.runs = 0
    registry = AgentRegistry()
    registry.register(_HitlAgent())
    checkpoint_store = SQLiteTaskCheckpointStore(db_path=tmp_path / "worker_ckpt.db")

    setup_loop = NexusLoop(registry, checkpoint_store=checkpoint_store)
    paused = asyncio.run(
        setup_loop.handle_task(
            Task(
                tenant_id="t1",
                user_id="u1",
                message="worker resume",
                context=TaskContext(capability="hitl.basic"),
                options=TaskExecutionOptions(
                    long_running=TaskLongRunningOptions(enabled=True),
                ),
            )
        )
    )
    token = paused.summary.resume_token
    assert token
    assert paused.state == TaskState.WAITING_FOR_HUMAN

    app = create_nexus_celery_worker_app(
        app_name="test_nexus_worker_resume_j3",
        broker_url="memory://",
        backend_url="cache+memory://",
        agent_registry=registry,
        checkpoint_store=checkpoint_store,
        task_always_eager=True,
    )
    queue = CeleryTaskQueue(app)
    store = DummyRunStore()
    service = DefaultRunService(store, execution_adapter=None)
    adapter = QueuedNexusExecutionAdapter(queue, service, wait_for_result=True)
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
    assert _HitlAgent.runs == 1
