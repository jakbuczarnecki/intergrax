# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import pytest

from intergrax.agents.agent_contract import Agent
from intergrax.agents.agent_engine import AgentEngine

from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.config import RuntimeConfig

from tests._support.builder import (
    FakeLLMAdapter,
    build_fake_embedding_manager,
    build_in_memory_session_manager,
    build_in_memory_vectorstore_manager,
)


# ----------------------------------------
# Agent WITHOUT pipeline
# ----------------------------------------
class NoPipelineAgent(Agent):

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
async def test_agent_engine_without_pipeline_uses_default_pipeline():
    agent = NoPipelineAgent()
    engine = AgentEngine({"test": agent})

    request = RuntimeRequest(
        tenant_id="t1",
        user_id="u1",
        session_id="s1",
        agent_id="test",
        message="hello"
    )

    response = await engine.run(request)

    # We expect runtime fallback (NoPlannerPipeline)
    assert response is not None
    assert response.answer is not None