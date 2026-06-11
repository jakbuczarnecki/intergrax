# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.agents.agent_contract import Agent
from intergrax.agents.agent_engine import AgentEngine
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.agent_execution_result import AgentExecutionStatus
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from testing_support.builder import (
    FakeLLMAdapter,
    build_fake_embedding_manager,
    build_in_memory_session_manager,
    build_in_memory_vectorstore_manager,
)


class _ContractUaepAgent(Agent):
    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="test",
            name="Test Agent",
            description="gate baseline agent",
            capabilities=["test.cap"],
        )

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        config = RuntimeConfig(
            llm_adapter=FakeLLMAdapter(),
            embedding_manager=build_fake_embedding_manager(),
            vectorstore_manager=build_in_memory_vectorstore_manager(),
            enable_rag=True,
            production_mode=False,
        )
        return RuntimeContext.build(
            config=config,
            session_manager=build_in_memory_session_manager(),
        )

    def get_steps(self, context: RuntimeContext) -> list[AgentStep]:
        _ = context
        return [AgentStep(step_id="s1", step_name="only", step_index=0)]

    async def run_step(self, step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
        _ = ctx
        return StepOutput(step_id=step.step_id, summary="OK")

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
async def test_agent_engine_run_with_result_returns_canonical_shape():
    agent = _ContractUaepAgent()
    engine = AgentEngine({"test": agent})

    request = RuntimeRequest(
        tenant_id="t1",
        user_id="u1",
        session_id="s1",
        agent_id="test",
        message="hello",
    )

    result = await engine.run_with_result(request)

    assert result.agent_id == "test"
    assert result.status == AgentExecutionStatus.COMPLETED
    assert result.summary == "OK"
    assert result.run_id
