# © Artur Czarnecki. All rights reserved.

import pytest

from echo.echo_agent import EchoAgent
from intergrax.agents.agent_engine import AgentEngine
from intergrax.agents import supports_uaep
from intergrax.contracts.agent_execution_result import AgentExecutionStatus
from intergrax.llm.messages import (
    build_model_input_messages_envelope,
    MODEL_INPUT_MESSAGES_METADATA_KEY,
    STRUCTURED_MODEL_INPUT_REQUIRED_REASON,
)
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
    assembled = [e for e in bus.history if e.event_type == RuntimeEventType.CONTEXT_ASSEMBLED]
    assert assembled
    assert assembled[0].payload.get("engine_id") == "default"
    assert any(e.event_type == RuntimeEventType.STEP_STARTED for e in bus.history)
    decision_events = [
        e for e in bus.history if e.event_type == RuntimeEventType.DECISION_EMITTED
    ]
    assert decision_events
    for event in decision_events:
        record = event.payload.get("decision_record")
        assert isinstance(record, dict)
        assert record.get("version") == "decision_record.v1"
        assert record.get("task_id") == "task_echo_uaep"
        assert record.get("agent_id") == "echo"
        assert record.get("decision_type")


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gate
async def test_legacy_uaep_fails_closed_for_structured_model_input():
    from intergrax.llm.messages import ChatMessage

    bus = RuntimeEventBus()
    engine = AgentEngine({"echo": EchoAgent()}, event_bus=bus)
    envelope = build_model_input_messages_envelope(
        [
            ChatMessage(role="user", content="history user", entry_id="h1"),
            ChatMessage(role="assistant", content="assistant", entry_id="h2"),
            ChatMessage(role="tool", content="tool out", entry_id="h3", tool_call_id="c1"),
            ChatMessage(role="user", content="final objective", entry_id="h4"),
        ]
    )
    request = RuntimeRequest(
        tenant_id="t1",
        user_id="u1",
        session_id="s1",
        agent_id="echo",
        message="compat text",
        metadata={
            "run_id": "run_echo_structured",
            "task_id": "task_echo_structured",
            MODEL_INPUT_MESSAGES_METADATA_KEY: envelope,
        },
    )

    result = await engine.run_with_result(request)

    assert result.status == AgentExecutionStatus.FAILED
    assert result.errors == [STRUCTURED_MODEL_INPUT_REQUIRED_REASON]
    assert not any(e.event_type == RuntimeEventType.STEP_STARTED for e in bus.history)
