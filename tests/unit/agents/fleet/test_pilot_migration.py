# © Artur Czarnecki. All rights reserved.

import pytest

from echo.echo_agent import EchoAgent
from research.research_agent import ResearchAgent
from signoff_probe.signoff_probe_agent import SignoffProbeAgent
from intergrax.agents.agent_engine import AgentEngine
from intergrax.agents import supports_uaep
from intergrax.contracts.agent_execution_result import AgentExecutionStatus
from intergrax.contracts.agent_run import AgentRunRequest, RequestIdentity
from intergrax.contracts.agent_run_enums import AgentRunStatus, CognitivePattern
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("factory", "expected_fragment"),
    [
        (EchoAgent, "echo:"),
        (SignoffProbeAgent, "signoff_probe:"),
        (ResearchAgent, "research findings"),
    ],
)
async def test_pilot_agents_typed_run(factory, expected_fragment) -> None:
    agent = factory()
    contract = agent.get_contract()
    assert contract.cognitive_pattern == CognitivePattern.REFLEX
    result = await agent.run(
        AgentRunRequest(
            input="pilot smoke",
            identity=RequestIdentity(tenant_id="t1", user_id="u1"),
        )
    )
    assert result.status == AgentRunStatus.SUCCEEDED
    if isinstance(result.output, dict):
        summary = str(result.output.get("summary") or result.output.get("answer") or "")
    else:
        summary = str(result.output or "")
    assert expected_fragment in summary


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_echo_still_supports_uaep_after_migration() -> None:
    agent = EchoAgent()
    assert supports_uaep(agent) is True
    engine = AgentEngine({"echo": agent})
    result = await engine.run_with_result(
        RuntimeRequest(
            tenant_id="t1",
            user_id="u1",
            session_id="s1",
            agent_id="echo",
            message="uaep compat",
            metadata={"run_id": "run_echo_mig", "task_id": "task_echo_mig"},
        )
    )
    assert result.status == AgentExecutionStatus.COMPLETED
    assert "uaep compat" in result.summary
