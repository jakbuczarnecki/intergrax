# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.agents.agent_engine import AgentEngine
from intergrax.agents.authoring.base import IntergraxAgent
from intergrax.agents.authoring.step_outcome import StepOutcome
from intergrax.agents.runtime_request_bridge import (
    acp_session_enabled,
    runtime_request_to_agent_run,
)
from intergrax.contracts.agent_contract_meta import AgentRiskLevel
from intergrax.contracts.agent_run_enums import TerminalReason
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager


class _BridgeAgent(IntergraxAgent):
    contract_id = "bridge-agent"
    capabilities = ("demo.bridge",)
    agent_name = "Bridge"
    agent_description = "ACP bridge test"
    risk_level = AgentRiskLevel.LOW
    max_steps = 3

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        config = RuntimeConfig(
            llm_adapter=FakeLLMAdapter(),
            production_mode=False,
            enable_rag=False,
            enable_websearch=False,
        )
        return RuntimeContext.build(
            config=config,
            session_manager=build_in_memory_session_manager(),
        )

    async def on_next_step(self, step_ctx: AgentStepContext) -> StepOutcome:
        return StepOutcome.complete(
            output={"answer": f"ok-{step_ctx.step_index}"},
            terminal_reason=TerminalReason.GOAL_MET,
        )


@pytest.mark.unit
@pytest.mark.gate
def test_runtime_request_to_agent_run_maps_identity() -> None:
    agent = _BridgeAgent()
    runtime_request = RuntimeRequest(
        agent_id="bridge-agent",
        user_id="user-9",
        session_id="sess-1",
        message="hello",
        tenant_id="tenant-z",
        metadata={"user_id": "user-9", "acp.session.v1": True},
    )
    agent_run = runtime_request_to_agent_run(runtime_request, contract=agent.get_contract())
    assert agent_run.identity.tenant_id == "tenant-z"
    assert agent_run.identity.user_id == "user-9"
    assert agent_run.input == "hello"


@pytest.mark.unit
@pytest.mark.gate
def test_acp_session_enabled_reads_metadata_flag() -> None:
    request = RuntimeRequest(
        agent_id="a",
        user_id="u",
        session_id="s",
        message="x",
        metadata={"acp.session.v1": True},
    )
    assert acp_session_enabled(request) is True


@pytest.mark.unit
@pytest.mark.gate
async def test_agent_engine_acp_session_bridge() -> None:
    agent = _BridgeAgent()
    request = RuntimeRequest(
        agent_id="bridge-agent",
        user_id="user-1",
        session_id="sess-1",
        message="bridge",
        tenant_id="tenant-a",
        metadata={"user_id": "user-1", "acp.session.v1": True},
    )
    answer = await AgentEngine.run_agent(agent, request)
    assert "ok-0" in answer.answer
