# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import pytest

from intergrax.agents.agent_contract import Agent
from intergrax.agents.agent_engine import AgentEngine

from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest, RuntimeAnswer
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.pipelines.contract import RuntimePipeline
from testing_support.builder import FakeLLMAdapter, build_fake_embedding_manager, build_in_memory_session_manager, build_in_memory_vectorstore_manager


# ----------------------------------------
# Fake Pipeline (minimal, deterministic)
# ----------------------------------------
class FakePipeline(RuntimePipeline):
    async def _inner_run(self, state: RuntimeState) -> RuntimeAnswer:

        answer: str = "OK"

        state.raw_answer = answer
        state.runtime_answer = RuntimeAnswer(
            run_id=state.run_id,
            answer=answer
        )

        return state.runtime_answer


# ----------------------------------------
# Fake Agent
# ----------------------------------------
class FakeAgent(Agent):

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        config = RuntimeConfig(
            llm_adapter=FakeLLMAdapter(),
            embedding_manager=build_fake_embedding_manager(),
            vectorstore_manager=build_in_memory_vectorstore_manager(),
            enable_rag=True,
            production_mode=False,
        )
        config.pipeline = FakePipeline()

        return RuntimeContext.build(
            config=config,
            session_manager=build_in_memory_session_manager()
        )


# ----------------------------------------
# TEST
# ----------------------------------------
@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gate
async def test_agent_engine_runs_pipeline():
    agent = FakeAgent()
    engine = AgentEngine({"test": agent})

    request = RuntimeRequest(
        tenant_id="t1",
        user_id="u1",
        session_id="s1",
        agent_id="test",
        message="hello"
    )

    response = await engine.run(request)

    assert response.answer == "OK"