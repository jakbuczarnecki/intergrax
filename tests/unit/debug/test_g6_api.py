# © Artur Czarnecki. All rights reserved.

import pytest
from fastapi.testclient import TestClient

from intergrax.agents.agent_contract import Agent
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType, HumanRequest
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.debug.app import create_debug_app
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.long_running.coordinator import LongRunningCoordinator
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from intergrax.runtime.task.task_contract import TaskExecutionOptions, TaskLongRunningOptions
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager

pytestmark = pytest.mark.unit


class _HitlAgent(Agent):
    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="hitl",
            name="HITL Agent",
            description="requests human approval once",
            capabilities=["hitl.basic"],
            max_steps=2,
        )

    def can_handle(self, task_context: object) -> CapabilityMatchResult:
        capability = getattr(task_context, "capability", None)
        if capability in (None, "hitl.basic"):
            return CapabilityMatchResult(
                matched=True,
                agent_id="hitl",
                matched_capabilities=["hitl.basic"],
                score=1.0,
            )
        return CapabilityMatchResult(matched=False, rationale="capability not supported")

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
        return StepOutput(step_id=step.step_id, summary="pending review")

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
                request_id="hr_hitl_1",
                prompt="Approve this action?",
                options=["approve", "reject"],
            ),
        )


@pytest.fixture
def event_store():
    store = InMemoryRuntimeEventStore()
    store.append(
        RuntimeEvent(
            tenant_id="t1",
            task_id="task_events_1",
            run_id="task_events_1",
            event_type=RuntimeEventType.HUMAN_APPROVAL_REQUESTED,
            phase=ExecutionPhase.HUMAN_APPROVAL,
            payload={"human_request": {"urgency": "high"}},
        ),
        tenant_id="t1",
    )
    return store


@pytest.fixture
def checkpoint_store(tmp_path):
    return SQLiteTaskCheckpointStore(db_path=tmp_path / "ckpt.db")


@pytest.fixture
def events_client(event_store):
    app = create_debug_app(runtime_event_store=event_store)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def checkpoints_client(checkpoint_store):
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="paused",
        context=TaskContext(capability="hitl.basic"),
        options=TaskExecutionOptions(long_running=TaskLongRunningOptions(enabled=True)),
    )
    LongRunningCoordinator.persist_checkpoint(
        task,
        checkpoint_store,
        progress_message="awaiting human input",
    )
    app = create_debug_app(checkpoint_store=checkpoint_store)
    with TestClient(app) as client:
        yield client, task


@pytest.mark.gate
@pytest.mark.no_ci
def test_debug_api_runtime_events(events_client: TestClient):
    response = events_client.get(
        "/debug/tasks/task_events_1/events",
        params={"tenant": "t1"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert payload["events"][0]["event_type"] == "human_approval_requested"


@pytest.mark.gate
@pytest.mark.no_ci
def test_debug_api_runtime_events_not_configured():
    app = create_debug_app()
    with TestClient(app) as client:
        response = client.get("/debug/tasks/x/events", params={"tenant": "t1"})
    assert response.status_code == 503


@pytest.mark.gate
@pytest.mark.no_ci
def test_debug_api_checkpoints(checkpoints_client):
    client, task = checkpoints_client
    response = client.get(
        f"/debug/tasks/{task.task_id}/checkpoints",
        params={"tenant": "t1"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert payload["checkpoints"][0]["resume_token"]


@pytest.mark.asyncio
@pytest.mark.gate
@pytest.mark.no_ci
async def test_debug_api_human_response_resume(tmp_path):
    registry = AgentRegistry()
    registry.register(_HitlAgent())
    checkpoint_store = SQLiteTaskCheckpointStore(db_path=tmp_path / "ckpt.db")
    event_store = InMemoryRuntimeEventStore()

    loop = NexusLoop(
        registry,
        checkpoint_store=checkpoint_store,
        runtime_event_store=event_store,
    )
    paused = await loop.handle_task(
        Task(
            tenant_id="t1",
            user_id="u1",
            message="sensitive action",
            context=TaskContext(capability="hitl.basic"),
            options=TaskExecutionOptions(
                long_running=TaskLongRunningOptions(enabled=True),
            ),
        )
    )
    assert paused.state == TaskState.WAITING_FOR_HUMAN
    assert checkpoint_store.get_latest(paused.task_id, "t1") is not None

    app = create_debug_app(
        registry=registry,
        checkpoint_store=checkpoint_store,
        runtime_event_store=event_store,
    )
    with TestClient(app) as client:
        approved = client.post(
            f"/debug/tasks/{paused.task_id}/human-response",
            params={"tenant": "t1"},
            json={"response": "approve"},
        )
        assert approved.status_code == 200
        body = approved.json()
        assert body["state"] == "completed"

        not_configured = create_debug_app()
        with TestClient(not_configured) as bare_client:
            denied = bare_client.post(
                f"/debug/tasks/{paused.task_id}/human-response",
                params={"tenant": "t1"},
                json={"response": "approve"},
            )
            assert denied.status_code == 503
