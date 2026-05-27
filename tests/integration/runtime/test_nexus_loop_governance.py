# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.agents.agent_contract import Agent
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType, HumanRequest
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.human.pause import GOVERNANCE_HUMAN_REQUEST_KEY, HumanPauseCoordinator
from intergrax.runtime.human.store import SQLiteHumanDecisionStore
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from intergrax.runtime.events.runtime_event import RuntimeEventType
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager


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


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gate
async def test_nexus_loop_pauses_for_human_request():
    registry = AgentRegistry()
    registry.register(_HitlAgent())
    loop = NexusLoop(registry)

    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="sensitive action",
        context=TaskContext(capability="hitl.basic"),
    )

    result = await loop.handle_task(task)

    assert result.state == TaskState.WAITING_FOR_HUMAN
    assert result.metadata.get("governance_human_request") is not None
    assert any(
        e.event_type == RuntimeEventType.HUMAN_APPROVAL_REQUESTED
        for e in loop.event_bus.history
    )


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gate
async def test_nexus_loop_resumes_after_human_approval():
    registry = AgentRegistry()
    registry.register(_HitlAgent())
    loop = NexusLoop(registry)

    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="sensitive action",
        context=TaskContext(capability="hitl.basic"),
        metadata={"human_approved": True},
    )

    result = await loop.handle_task(task)

    assert result.state == TaskState.COMPLETED
    assert GOVERNANCE_HUMAN_REQUEST_KEY not in result.metadata or not result.metadata.get(
        GOVERNANCE_HUMAN_REQUEST_KEY
    )


@pytest.mark.unit
@pytest.mark.gate
def test_human_pause_coordinator_records_response():
    task = Task(tenant_id="t1", user_id="u1", message="x")
    HumanPauseCoordinator.record_human_response(task, "approve")
    assert HumanPauseCoordinator.is_resumed(task) is True
    assert HumanPauseCoordinator.verdict_from_task(task) == HumanResponseVerdict.APPROVE


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gate
async def test_nexus_loop_fails_on_human_rejection(tmp_path):
    registry = AgentRegistry()
    registry.register(_HitlAgent())
    store = SQLiteHumanDecisionStore(db_path=tmp_path / "human.db")
    loop = NexusLoop(registry, human_decision_store=store)

    paused = await loop.handle_task(
        Task(
            tenant_id="t1",
            user_id="u1",
            message="sensitive action",
            context=TaskContext(capability="hitl.basic"),
        )
    )
    assert paused.state == TaskState.WAITING_FOR_HUMAN

    rejected = await loop.handle_task(
        Task(
            tenant_id="t1",
            user_id="u1",
            message="sensitive action",
            context=TaskContext(capability="hitl.basic"),
            task_id=paused.task_id,
            metadata={"human_response": "reject"},
        )
    )
    assert rejected.state == TaskState.FAILED
    assert "human rejected" in (rejected.metadata.get("validation_errors") or [""])[0]
    decisions = store.list_for_task(paused.task_id, "t1")
    assert len(decisions) == 1
    assert decisions[0].verdict == HumanResponseVerdict.REJECT


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gate
async def test_nexus_loop_escalates_and_persists(tmp_path):
    registry = AgentRegistry()
    registry.register(_HitlAgent())
    store = SQLiteHumanDecisionStore(db_path=tmp_path / "human.db")
    loop = NexusLoop(registry, human_decision_store=store)

    task_id = "task_escalate_1"
    await loop.handle_task(
        Task(
            tenant_id="t1",
            user_id="u1",
            message="sensitive action",
            context=TaskContext(capability="hitl.basic"),
            task_id=task_id,
        )
    )

    escalated = await loop.handle_task(
        Task(
            tenant_id="t1",
            user_id="u1",
            message="sensitive action",
            context=TaskContext(capability="hitl.basic"),
            task_id=task_id,
            metadata={"human_response": "escalate"},
        )
    )
    assert escalated.state == TaskState.WAITING_FOR_HUMAN
    assert escalated.metadata.get("escalation_level") == 1
    assert any(
        e.event_type == RuntimeEventType.INTERRUPT_ESCALATED
        for e in loop.event_bus.history
    )
    records = store.list_for_task(task_id, "t1")
    assert len(records) == 1
    assert records[0].verdict == HumanResponseVerdict.ESCALATE
