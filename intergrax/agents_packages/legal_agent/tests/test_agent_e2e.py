# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import pytest

from intergrax.agents.agent_engine import AgentEngine
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager

from intergrax.agents_packages.legal_agent.agent import LegalAgent
from tests._support.builder import FakeLLMAdapter

pytestmark = pytest.mark.e2e


@pytest.mark.asyncio
async def test_legal_agent_e2e():
    agent = LegalAgent(
        session_manager=SessionManager(
            storage=InMemorySessionStorage()
        ),
        llm_adapter=FakeLLMAdapter(
            fixed_text="SUMMARY: ...\nKEY CLAUSES: ...\nRISKS: ..."
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

    assert response.answer is not None
    assert "SUMMARY" in response.answer