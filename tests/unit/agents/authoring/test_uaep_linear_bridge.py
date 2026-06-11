# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.agents.authoring.base import IntergraxAgent
from intergrax.agents.authoring.decorators import step
from intergrax.agents.authoring.uaep_linear_bridge import (
    linear_agent_decide_after_step,
    linear_agent_get_steps,
)
from intergrax.contracts.agent_decision import AgentDecisionType
from intergrax.contracts.agent_step import StepOutput
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager


class _LinearAgent(IntergraxAgent):
    contract_id = "linear-test"
    capabilities = ("test.linear",)

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

    @step("s1")
    async def first(self, ctx: RuntimeExecutionContext) -> StepOutput:
        _ = ctx
        return StepOutput(step_id="s1", summary="one")

    @step("s2")
    async def second(self, ctx: RuntimeExecutionContext) -> StepOutput:
        _ = ctx
        return StepOutput(step_id="s2", summary="two")


@pytest.mark.unit
def test_linear_bridge_get_steps_and_decide() -> None:
    agent = _LinearAgent()
    ctx = agent.build_context(
        RuntimeRequest(
            tenant_id="t",
            user_id="u",
            session_id="s",
            agent_id="linear-test",
            message="hi",
        )
    )
    steps = linear_agent_get_steps(agent, ctx)
    assert [step.step_id for step in steps] == ["s1", "s2"]

    exec_ctx = RuntimeExecutionContext(
        run_id="r1",
        task_id="task-1",
        agent_id="linear-test",
        step_index=0,
        request=RuntimeRequest(
            tenant_id="t",
            user_id="u",
            session_id="s",
            agent_id="linear-test",
            message="hi",
        ),
        domain_context=ctx,
    )
    decision = linear_agent_decide_after_step(
        agent,
        steps[0],
        StepOutput(step_id="s1", summary="one"),
        exec_ctx,
    )
    assert decision.type == AgentDecisionType.CONTINUE
    assert decision.payload.get("next_step_id") == "s2"

    assert "decide_after_step" not in IntergraxAgent.__dict__
