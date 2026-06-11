# © Artur Czarnecki. All rights reserved.

from unittest.mock import AsyncMock, patch

import pytest

from intergrax.agents.agent_contract import Agent
from intergrax.agents.agent_engine import AgentEngine
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer, RuntimeRequest
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

    def validate(self, answer: RuntimeAnswer, *, context: RuntimeContext) -> object:
        from intergrax.contracts.validation import ValidationResult

        _ = context
        return ValidationResult(valid=True, errors=[])


@pytest.mark.unit
@pytest.mark.gate
async def test_agent_engine_runtime_fallback_emits_deprecation_warning() -> None:
    agent = _LegacyPipelineAgent()
    request = RuntimeRequest(
        tenant_id="t1",
        user_id="u1",
        session_id="s1",
        agent_id="legacy-pipeline",
        message="hello",
    )
    with patch(
        "intergrax.agents.agent_engine.RuntimeEngine.run",
        new_callable=AsyncMock,
        return_value=RuntimeAnswer(run_id="legacy-run", answer="legacy-ok"),
    ):
        with pytest.warns(DeprecationWarning, match="RuntimeEngine fallback is deprecated"):
            answer = await AgentEngine.run_agent(agent, request)
    assert answer.answer == "legacy-ok"
