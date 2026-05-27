# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.agents.agent_engine import AgentEngine
from intergrax.agents.uaep import supports_uaep
from intergrax.contracts.agent_execution_result import AgentExecutionStatus
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from research.research_agent import ResearchAgent
from research.summary_agent import SummaryAgent


@pytest.mark.unit
@pytest.mark.gate
def test_research_agents_support_uaep():
    assert supports_uaep(ResearchAgent()) is True
    assert supports_uaep(SummaryAgent()) is True


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gate
async def test_agent_engine_runs_research_via_uaep():
    bus = RuntimeEventBus()
    engine = AgentEngine({"research": ResearchAgent()}, event_bus=bus)
    request = RuntimeRequest(
        tenant_id="t1",
        user_id="u1",
        session_id="s1",
        agent_id="research",
        message="market trends in AI logistics",
        metadata={"run_id": "run_research_uaep", "task_id": "task_research_uaep"},
    )

    result = await engine.run_with_result(request)

    assert result.agent_id == "research"
    assert result.status == AgentExecutionStatus.COMPLETED
    assert "research findings" in result.summary
    assert any(e.event_type == RuntimeEventType.STEP_STARTED for e in bus.history)
    assert any(e.event_type == RuntimeEventType.DECISION_EMITTED for e in bus.history)


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gate
async def test_agent_engine_runs_summary_via_uaep():
    bus = RuntimeEventBus()
    engine = AgentEngine({"research-summary": SummaryAgent()}, event_bus=bus)
    request = RuntimeRequest(
        tenant_id="t1",
        user_id="u1",
        session_id="s1",
        agent_id="research-summary",
        message="--- prior agent outputs ---\nfindings line one",
        metadata={"run_id": "run_summary_uaep", "task_id": "task_summary_uaep"},
    )

    result = await engine.run_with_result(request)

    assert result.status == AgentExecutionStatus.COMPLETED
    assert result.summary.startswith("summary:")
