# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.agents.agent_contract import Agent
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType, HumanRequest
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState
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
async def test_nexus_loop_persists_runtime_events_via_injected_store():
    store = InMemoryRuntimeEventStore()
    registry = AgentRegistry()
    registry.register(_HitlAgent())
    loop = NexusLoop(registry, runtime_event_store=store)

    paused = await loop.handle_task(
        Task(
            tenant_id="t1",
            user_id="u1",
            message="sensitive action",
            context=TaskContext(capability="hitl.basic"),
        )
    )

    assert paused.state == TaskState.WAITING_FOR_HUMAN
    persisted = store.list_for_task(paused.task_id, tenant_id="t1")
    assert persisted
    assert all(event.tenant_id == "t1" for event in persisted)
    assert any(
        event.event_type == RuntimeEventType.HUMAN_APPROVAL_REQUESTED for event in persisted
    )


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gate
async def test_nexus_loop_persists_runtime_events_via_sqlite_path(tmp_path):
    registry = AgentRegistry()
    registry.register(_HitlAgent())
    loop = NexusLoop(
        registry,
        runtime_events_db_path=tmp_path / "runtime_events.db",
    )

    paused = await loop.handle_task(
        Task(
            tenant_id="t1",
            user_id="u1",
            message="sensitive action",
            context=TaskContext(capability="hitl.basic"),
        )
    )

    assert loop.runtime_event_store is not None
    persisted = loop.runtime_event_store.list_for_task(paused.task_id, tenant_id="t1")
    assert persisted
    assert any(
        event.event_type == RuntimeEventType.HUMAN_APPROVAL_REQUESTED for event in persisted
    )
