# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import json

import pytest

from intergrax.agents.agent_engine import AgentEngine
from intergrax.agents_packages.legal_agent.pipeline import Decision, LegalAnalysisOutput
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager

from intergrax.agents_packages.legal_agent.agent import LegalAgent
from tests._support.builder import FakeLLMAdapter

pytestmark = pytest.mark.e2e


@pytest.mark.asyncio
async def test_legal_agent_e2e_real_ollama():
    llm = LLMAdapterRegistry.create(LLMProvider.OLLAMA)

    agent = LegalAgent(
        session_manager=SessionManager(
            storage=InMemorySessionStorage()
        ),
        llm_adapter=llm,
        production_mode=False,
    )

    engine = AgentEngine({"legal": agent})

    request = RuntimeRequest(
        tenant_id="t1",
        user_id="u1",
        session_id="s1",
        agent_id="legal",
        message="""
        This Agreement is made between Company A and Company B.
        Company A agrees to deliver software services.
        Payment will be made within 60 days.
        There is no liability limitation clause.
        """
    )

    response = await engine.run(request)

    assert response.answer is not None

    # --- parse JSON ---
    data = json.loads(response.answer)

    # --- structural validation ---
    assert "summary" in data
    assert "risks" in data
    assert "decision" in data

    # --- minimal sanity checks ---
    assert isinstance(data["risks"], list)
    assert data["decision"]["overall_risk"] in ["low", "medium", "high", "critical"]