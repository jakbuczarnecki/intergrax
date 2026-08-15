# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.agents.agent_engine import AgentEngine
from intergrax.agents.authoring.base import IntergraxAgent
from intergrax.agents.authoring.step_outcome import StepOutcome
from intergrax.agents.runtime_request_bridge import (
    acp_session_enabled,
    runtime_request_to_agent_run,
)
from intergrax.llm.messages import (
    build_model_input_messages_envelope,
    MODEL_INPUT_MESSAGES_METADATA_KEY,
)
from intergrax.contracts.agent_contract_meta import AgentRiskLevel
from intergrax.contracts.agent_run_enums import TerminalReason
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.llm.messages import ChatMessage
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


@pytest.mark.unit
@pytest.mark.gate
def test_runtime_request_with_model_envelope_uses_final_user_as_acp_input() -> None:
    agent = _BridgeAgent()
    messages = [
        ChatMessage(role="system", content="[context:task_message:t1] objective"),
        ChatMessage(role="user", content="history", entry_id="h1"),
        ChatMessage(role="user", content="final objective", entry_id="final"),
    ]
    envelope = build_model_input_messages_envelope(messages)
    runtime_request = RuntimeRequest(
        agent_id="bridge-agent",
        user_id="user-1",
        session_id="sess-1",
        message="[context:task_message:t1] objective\n\nfinal objective",
        tenant_id="tenant-a",
        metadata={
            "user_id": "user-1",
            "acp.session.v1": True,
            MODEL_INPUT_MESSAGES_METADATA_KEY: envelope,
        },
    )
    agent_run = runtime_request_to_agent_run(runtime_request, contract=agent.get_contract())
    assert agent_run.input == "final objective"
    assert agent_run.metadata[MODEL_INPUT_MESSAGES_METADATA_KEY] == envelope


@pytest.mark.unit
@pytest.mark.gate
def test_runtime_request_malformed_model_envelope_fails_before_execution() -> None:
    agent = _BridgeAgent()
    runtime_request = RuntimeRequest(
        agent_id="bridge-agent",
        user_id="user-1",
        session_id="sess-1",
        message="hello",
        tenant_id="tenant-a",
        metadata={
            "user_id": "user-1",
            "acp.session.v1": True,
            MODEL_INPUT_MESSAGES_METADATA_KEY: {"schema_version": "model_input_messages.v1", "messages": []},
        },
    )
    with pytest.raises(ValueError):
        runtime_request_to_agent_run(runtime_request, contract=agent.get_contract())


@pytest.mark.unit
@pytest.mark.gate
async def test_acp_run_parses_model_envelope_before_on_run_start() -> None:
    from unittest.mock import AsyncMock, patch

    from intergrax.agents.authoring.acp_run import run_acp_session
    from intergrax.agents.authoring.llm_router import StepLLMRouter
    from intergrax.contracts.agent_run import AgentRunRequest, RequestIdentity
    from intergrax.contracts.agent_run_enums import AgentRunStatus

    agent = _BridgeAgent()
    on_run_start = AsyncMock()
    agent.on_run_start = on_run_start
    messages = [ChatMessage(role="user", content="final only", entry_id="u1")]
    envelope = build_model_input_messages_envelope(messages)
    request = AgentRunRequest(
        input="final only",
        identity=RequestIdentity(tenant_id="tenant-a", user_id="user-1"),
        agent_id="bridge-agent",
        metadata={
            "acp.session.v1": True,
            MODEL_INPUT_MESSAGES_METADATA_KEY: envelope,
        },
    )
    captured: list[tuple[ChatMessage, ...]] = []
    original_router = StepLLMRouter

    def _router_factory(*args, **kwargs):
        captured.append(kwargs.get("model_input_messages", ()))
        return original_router(*args, **kwargs)

    with patch("intergrax.agents.authoring.acp_run.StepLLMRouter", side_effect=_router_factory):
        result = await run_acp_session(agent, request)
    on_run_start.assert_awaited_once()
    assert captured
    assert len(captured[0]) == 1
    assert captured[0][0].content == "final only"
    assert result.status == AgentRunStatus.SUCCEEDED


@pytest.mark.unit
@pytest.mark.gate
async def test_acp_run_malformed_envelope_fails_before_on_run_start() -> None:
    from unittest.mock import AsyncMock

    from intergrax.agents.authoring.acp_run import run_acp_session
    from intergrax.contracts.agent_run import AgentRunRequest, RequestIdentity
    from intergrax.contracts.agent_run_enums import AgentRunStatus

    agent = _BridgeAgent()
    on_run_start = AsyncMock()
    agent.on_run_start = on_run_start
    request = AgentRunRequest(
        input="x",
        identity=RequestIdentity(tenant_id="tenant-a", user_id="user-1"),
        agent_id="bridge-agent",
        metadata={
            MODEL_INPUT_MESSAGES_METADATA_KEY: {
                "schema_version": "model_input_messages.v1",
                "messages": [],
            },
        },
    )
    result = await run_acp_session(agent, request)
    on_run_start.assert_not_awaited()
    assert result.status == AgentRunStatus.FAILED
