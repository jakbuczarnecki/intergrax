# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import pytest

from intergrax.agents.agent_contract import Agent
from intergrax.contracts.agent_run import AgentRunRequest, AgentRunResult
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.nexus.agents.agent_engine import AgentEngine

from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.config import RuntimeConfig

from testing_support.builder import (
    FakeLLMAdapter,
    build_fake_embedding_manager,
    build_in_memory_session_manager,
    build_in_memory_vectorstore_manager,
    build_runtime_request_for_tests,
)


# ----------------------------------------
# Agent WITHOUT pipeline
# ----------------------------------------
class NoPipelineAgent(Agent):

    async def run(self, request: AgentRunRequest) -> AgentRunResult:
        _ = request
        raise NotImplementedError("NoPipelineAgent is exercised via AgentEngine/UAEP only")

    def get_steps(self) -> list[AgentStep]:
        return [AgentStep(step_id="noop", step_name="noop", step_index=0)]

    async def run_step(self, step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
        _ = ctx
        return StepOutput(step_id=step.step_id, summary="noop")

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        config = RuntimeConfig(
            llm_adapter=FakeLLMAdapter(),
            embedding_manager=build_fake_embedding_manager(),
            vectorstore_manager=build_in_memory_vectorstore_manager(tenant_id="t1"),
            enable_rag=True,
            production_mode=False,
        )

        # intentionally NO config.pipeline

        return RuntimeContext.build(
            config=config,
            session_manager=build_in_memory_session_manager()
        )


# ----------------------------------------
# TEST
# ----------------------------------------
@pytest.mark.asyncio
async def test_agent_engine_without_pipeline_rejects_legacy_path():
    agent = NoPipelineAgent()
    engine = AgentEngine({"test": agent})

    request = build_runtime_request_for_tests(
        seed="no-pipeline-reject",
        tenant_id="t1",
        user_id="u1",
        session_id="s1",
        agent_id="test",
        message="hello",
    )

    with pytest.raises(ValueError, match="ACP-CLOSE-LEG-5"):
        await engine.run(request)