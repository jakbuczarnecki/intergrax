# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.agents.agent_engine import AgentEngine
from intergrax.agents.uaep import supports_uaep
from intergrax.contracts.agent_execution_result import AgentExecutionStatus
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager
from legal.config.legal_agent_config import LegalAgentConfig
from legal.legal_agent import LegalAgent
from legal.steps.legal_finalize_answer_step import FinalAnswerModel
from testing_support.builder import (
    FakeLLMAdapter,
    build_fake_embedding_manager,
    build_in_memory_session_manager,
    build_in_memory_vectorstore_manager,
)


@pytest.mark.unit
@pytest.mark.gate
def test_legal_agent_supports_uaep():
    cfg = LegalAgentConfig(
        session_manager=build_in_memory_session_manager(),
        llm_adapter=FakeLLMAdapter(fixed_text="No critical issues."),
        production_mode=False,
        enable_rag=False,
        use_legal_tool_decision=False,
        enable_sequential_legal_pipeline=True,
    )
    assert supports_uaep(LegalAgent(config=cfg)) is True


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gate
async def test_agent_engine_runs_legal_via_uaep():
    tenant_id = "legal-uaep-gate"
    cfg = LegalAgentConfig(
        session_manager=SessionManager(storage=InMemorySessionStorage()),
        llm_adapter=FakeLLMAdapter(
            fixed_text="No critical issues identified.",
            fake_structured_data=FinalAnswerModel(
                answer="No critical issues identified."
            ),
        ),
        production_mode=False,
        enable_rag=True,
        embedding_manager=build_fake_embedding_manager(),
        vectorstore_manager=build_in_memory_vectorstore_manager(tenant_id=tenant_id),
        use_legal_tool_decision=False,
        enable_sequential_legal_pipeline=True,
        use_llm_legal_route_planner=False,
        use_legal_run_evaluator=False,
        organization_compliance_policy="",
    )
    agent = LegalAgent(config=cfg)
    bus = RuntimeEventBus()
    engine = AgentEngine({"legal": agent}, event_bus=bus)
    request = RuntimeRequest(
        tenant_id=tenant_id,
        user_id="u1",
        session_id="s1",
        agent_id="legal",
        message="Review payment terms in the attached contract.",
        metadata={"run_id": "run_legal_uaep", "task_id": "task_legal_uaep"},
    )

    result = await engine.run_with_result(request)

    assert result.agent_id == "legal"
    assert result.status == AgentExecutionStatus.COMPLETED
    assert result.summary
    assert any(e.event_type == RuntimeEventType.CONTEXT_BUILT for e in bus.history)
    assert any(e.event_type == RuntimeEventType.STEP_STARTED for e in bus.history)
    assert any(e.event_type == RuntimeEventType.DECISION_EMITTED for e in bus.history)
