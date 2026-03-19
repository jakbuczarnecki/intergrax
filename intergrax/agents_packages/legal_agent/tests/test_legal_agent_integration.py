# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import pytest

from intergrax.agents.agent_engine import AgentEngine
from intergrax.agents_packages.legal_agent.pipeline import Decision, LegalAnalysisOutput
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager

from intergrax.agents_packages.legal_agent.agent import LegalAgent
from tests._support.builder import FakeLLMAdapter

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_legal_agent_e2e():
    fake_output = LegalAnalysisOutput(
        summary="Test summary",
        contract_type="nda",
        key_clauses=[],
        risks=[],
        missing_clauses=[],
        decision=Decision(
            sign_recommendation="approve",
            overall_risk="low"
        )
    )

    agent = LegalAgent(
        session_manager=SessionManager(
            storage=InMemorySessionStorage()
        ),
        llm_adapter=FakeLLMAdapter(
            fake_structured_data=fake_output
        ),
        production_mode=False,
    )

    engine = AgentEngine({"legal": agent})

    request = RuntimeRequest(
        tenant_id="t1",
        user_id="u1",
        session_id="s1",
        agent_id="legal",
        message="This agreement is between..."
    )

    response = await engine.run(request)

    # --- assertions ---
    assert response.answer is not None

    # response.answer is JSON string
    import json
    data = json.loads(response.answer)

    assert data["summary"] == "Test summary"
    assert data["decision"]["overall_risk"] == "low"