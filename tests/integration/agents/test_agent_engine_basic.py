# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import pytest

from intergrax.agents.agent_contract import Agent
from intergrax.agents.agent_engine import AgentEngine

from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.config import RuntimeConfig
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager


class FakeAgent(Agent):

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        config = RuntimeConfig(
            llm_adapter=FakeLLMAdapter(),
            enable_rag=False,
            production_mode=False,
        )
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
async def test_agent_engine_rejects_pipeline_only_agent():
    agent = FakeAgent()
    engine = AgentEngine({"test": agent})

    request = RuntimeRequest(
        tenant_id="t1",
        user_id="u1",
        session_id="s1",
        agent_id="test",
        message="hello"
    )

    with pytest.raises(ValueError, match="ACP-CLOSE-LEG-5"):
        await engine.run(request)