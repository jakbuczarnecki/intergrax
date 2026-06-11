# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.agents.authoring.base import IntergraxAgent
from intergrax.agents.authoring.step_outcome import StepOutcome
from intergrax.contracts.acp_metadata_keys import AcpMetadataKey
from intergrax.contracts.agent_contract_meta import AgentRiskLevel
from intergrax.contracts.agent_run import AgentRunRequest, RequestIdentity
from intergrax.contracts.agent_run_enums import TerminalReason
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager


class _LlmTraceAgent(IntergraxAgent):
    contract_id = "llm-trace-agent"
    capabilities = ("demo.llm_trace",)
    agent_name = "LlmTrace"
    agent_description = "Wave 3 LLM trace"
    risk_level = AgentRiskLevel.LOW
    max_steps = 2

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
        assert step_ctx.llm_router is not None
        llm_result = await step_ctx.llm_router.complete("ping", model_hint="balanced")
        return StepOutcome.complete(
            output={"answer": llm_result.text},
            terminal_reason=TerminalReason.GOAL_MET,
        )


@pytest.mark.unit
@pytest.mark.gate
async def test_acp_run_populates_llm_calls_in_trace() -> None:
    agent = _LlmTraceAgent()
    request = AgentRunRequest(
        input="wave3",
        identity=RequestIdentity(tenant_id="tenant-1", user_id="user-1"),
        metadata={AcpMetadataKey.SESSION_ENABLED: True},
    )
    result = await agent.run(request)
    assert result.trace.steps
    assert result.trace.steps[0].llm_calls
    assert result.trace.steps[0].llm_calls[0].model_id == "balanced"
    assert result.structured_data
