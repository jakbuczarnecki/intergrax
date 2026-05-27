# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.agents.agent_contract import Agent
from intergrax.agents.uaep import UAEPExecutor, supports_uaep
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer, RuntimeRequest
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager


class _UaepStubAgent(Agent):
    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="uaep-stub",
            name="UAEP Stub",
            description="minimal uaep agent",
            capabilities=["stub.basic"],
            max_steps=3,
        )

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        config = RuntimeConfig(
            llm_adapter=FakeLLMAdapter(fixed_text="stub-ok"),
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
        return [
            AgentStep(step_id="s1", step_name="first", step_index=0),
            AgentStep(step_id="s2", step_name="second", step_index=1),
        ]

    async def run_step(self, step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
        return StepOutput(step_id=step.step_id, summary=f"out:{step.step_name}")

    def decide_after_step(
        self,
        step: AgentStep,
        output: StepOutput | None,
        ctx: RuntimeExecutionContext,
    ) -> AgentDecision:
        _ = output, ctx
        if step.step_id == "s2":
            return AgentDecision(type=AgentDecisionType.COMPLETE, reason="done")
        return AgentDecision(type=AgentDecisionType.CONTINUE)


@pytest.mark.unit
@pytest.mark.gate
def test_supports_uaep_detects_step_protocol():
    assert supports_uaep(_UaepStubAgent()) is True


@pytest.mark.unit
def test_supports_uaep_false_for_legacy_agent():
    class _Legacy(Agent):
        def build_context(self, request: RuntimeRequest) -> RuntimeContext:
            return _UaepStubAgent().build_context(request)

    assert supports_uaep(_Legacy()) is False


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.gate
async def test_uaep_executor_runs_runtime_controlled_steps():
    bus = RuntimeEventBus()
    executor = UAEPExecutor(event_bus=bus)
    agent = _UaepStubAgent()
    request = RuntimeRequest(
        tenant_id="t1",
        user_id="u1",
        session_id="s1",
        agent_id="uaep-stub",
        message="hi",
        metadata={"run_id": "run_uaep_1", "task_id": "task_uaep_1"},
    )

    answer, validation, _context, _governance = await executor.execute(agent, request)

    assert validation.valid
    assert answer.answer == "out:second"
    step_events = [e for e in bus.history if e.event_type == RuntimeEventType.STEP_STARTED]
    assert len(step_events) == 2
    assert bus.history[-1].event_type in {
        RuntimeEventType.VALIDATION_PASSED,
        RuntimeEventType.VALIDATION_FAILED,
    }
