# © Artur Czarnecki. All rights reserved.

import pytest

from dataclasses import replace

from intergrax.agents.agent_contract import Agent
from intergrax.agents.uaep import UAEPExecutor
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.memory_write_policy import MemoryWritePolicy
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.task_memory import InMemoryTaskMemoryStore
from testing_support.builder import (
    FakeLLMAdapter,
    build_in_memory_session_manager,
    build_runtime_request_for_tests,
    canonical_execution_identity_scope,
)


def _verified_canonical_identity(*, tenant_id: str = "t1", user_id: str = "u1") -> RequestIdentity:
    return RequestIdentity(
        tenant_id=tenant_id,
        user_id=user_id,
        principal_type=PrincipalType.USER,
        auth_subject=user_id,
    )


class _MemoryUaepAgent(Agent):
    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="memory-agent",
            name="Memory Agent",
            description="uses ctx.memory_view",
            capabilities=["memory.demo"],
            max_steps=2,
        )

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        config = RuntimeConfig(
            llm_adapter=FakeLLMAdapter(fixed_text="ok"),
            enable_rag=False,
            production_mode=False,
            tenant_id=request.tenant_id,
            enable_user_longterm_memory=False,
        )
        return RuntimeContext.build(
            config=config,
            session_manager=build_in_memory_session_manager(),
        )

    def get_steps(self, context: RuntimeContext) -> list[AgentStep]:
        _ = context
        return [AgentStep(step_id="persist", step_name="persist", step_index=0)]

    async def run_step(self, step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
        _ = step
        assert ctx.memory_view is not None
        await ctx.memory_view.write(
            "vendor_report",
            "draft",
            {"subject": "Acme Q1"},
            policy=MemoryWritePolicy.REPLACE,
        )
        loaded = await ctx.memory_view.read("vendor_report", "draft")
        return StepOutput(step_id=step.step_id, summary=str(loaded))

    def decide_after_step(
        self,
        step: AgentStep,
        output: StepOutput | None,
        ctx: RuntimeExecutionContext,
    ) -> AgentDecision:
        _ = step, output, ctx
        return AgentDecision(type=AgentDecisionType.COMPLETE, reason="done")


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gate
async def test_uaep_executor_wires_memory_view():
    bus = RuntimeEventBus()
    store = InMemoryTaskMemoryStore()
    executor = UAEPExecutor(event_bus=bus, task_memory_store=store)
    agent = _MemoryUaepAgent()
    request = replace(
        build_runtime_request_for_tests(
            seed="uaep-memory-view-1",
            tenant_id="t1",
            agent_id="memory-agent",
            user_id="u1",
            session_id="s1",
            message="persist memory",
        ),
        canonical_identity=_verified_canonical_identity(tenant_id="t1"),
    )
    task_id = str(request.task_id)

    with canonical_execution_identity_scope(str(request.run_id)):
        answer, validation, _governance = await executor.execute(agent, request)

    assert validation.valid
    assert "Acme Q1" in answer.answer
    memory_events = [
        event for event in bus.history if event.event_type in {
            RuntimeEventType.MEMORY_READ,
            RuntimeEventType.MEMORY_WRITE,
        }
    ]
    assert len(memory_events) >= 2
    persisted = store.get(
        tenant_id="t1",
        task_id=task_id,
        namespace="vendor_report",
        key="draft",
    )
    assert persisted is not None
    assert persisted.value["subject"] == "Acme Q1"


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gate
async def test_uaep_executor_without_store_leaves_memory_view_none():
    bus = RuntimeEventBus()
    executor = UAEPExecutor(event_bus=bus)
    agent = _MemoryUaepAgent()

    class _NoMemoryAgent(_MemoryUaepAgent):
        async def run_step(self, step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
            assert ctx.memory_view is None
            return StepOutput(step_id=step.step_id, summary="no memory")

    request = replace(
        build_runtime_request_for_tests(
            seed="uaep-memory-view-2",
            tenant_id="t1",
            agent_id="memory-agent",
            user_id="u1",
            session_id="s1",
            message="skip",
        ),
        canonical_identity=_verified_canonical_identity(tenant_id="t1"),
    )

    with canonical_execution_identity_scope(str(request.run_id)):
        answer, validation, _governance = await executor.execute(_NoMemoryAgent(), request)
    assert validation.valid
    assert answer.answer == "no memory"
