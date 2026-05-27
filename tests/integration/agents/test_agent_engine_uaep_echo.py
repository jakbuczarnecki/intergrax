# © Artur Czarnecki. All rights reserved.

import pytest

from echo.echo_agent import EchoAgent
from intergrax.agents.agent_engine import AgentEngine
from intergrax.agents.uaep import supports_uaep
from intergrax.contracts.agent_execution_result import AgentExecutionStatus
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gate
async def test_echo_agent_uses_uaep_protocol():
    agent = EchoAgent()
    assert supports_uaep(agent) is True


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gate
async def test_agent_engine_runs_echo_via_uaep():
    bus = RuntimeEventBus()
    engine = AgentEngine({"echo": EchoAgent()}, event_bus=bus)
    request = RuntimeRequest(
        tenant_id="t1",
        user_id="u1",
        session_id="s1",
        agent_id="echo",
        message="uaep path",
        metadata={"run_id": "run_echo_uaep", "task_id": "task_echo_uaep"},
    )

    result = await engine.run_with_result(request)

    assert result.agent_id == "echo"
    assert result.status == AgentExecutionStatus.COMPLETED
    assert "uaep path" in result.summary
    assert result.cost == 0.0
    assert any(e.event_type == RuntimeEventType.CONTEXT_BUILT for e in bus.history)
    assert any(e.event_type == RuntimeEventType.STEP_STARTED for e in bus.history)
    assert any(e.event_type == RuntimeEventType.DECISION_EMITTED for e in bus.history)
