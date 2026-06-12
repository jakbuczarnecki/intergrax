# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.agents.agent_contract import Agent
from intergrax.agents.agent_engine import AgentEngine
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager


class _LegacyPipelineAgent(Agent):
    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="legacy-pipeline",
            name="Legacy",
            description="no uaep",
            capabilities=["legacy.basic"],
        )

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        config = RuntimeConfig(
            llm_adapter=FakeLLMAdapter(fixed_text="legacy-ok"),
            enable_rag=False,
            production_mode=False,
            tenant_id=request.tenant_id,
        )
        return RuntimeContext.build(
            config=config,
            session_manager=build_in_memory_session_manager(),
        )


@pytest.mark.unit
@pytest.mark.gate
async def test_agent_engine_rejects_runtime_engine_fallback() -> None:
    agent = _LegacyPipelineAgent()
    request = RuntimeRequest(
        tenant_id="t1",
        user_id="u1",
        session_id="s1",
        agent_id="legacy-pipeline",
        message="hello",
    )
    with pytest.raises(ValueError, match="ACP-CLOSE-LEG-5"):
        await AgentEngine.run_agent(agent, request)
