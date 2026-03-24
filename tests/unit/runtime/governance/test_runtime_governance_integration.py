# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import pytest

from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime import RuntimeEngine
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager
from testing_support.builder import FakeLLMAdapter

pytestmark = pytest.mark.unit


class DummyGovernanceService:
    def __init__(self):
        self.called = False
        self.run_id = None
        self.agent_id = None

    def evaluate(self, run_id: str, agent_id: str):
        self.called = True
        self.run_id = run_id
        self.agent_id = agent_id
        return None


@pytest.mark.asyncio
async def test_runtime_calls_governance_after_run():
    config = RuntimeConfig(
        llm_adapter=FakeLLMAdapter(),
        enable_rag=False,
        enable_websearch=False,
    )
    config.production_mode = False

    sm = SessionManager(storage=InMemorySessionStorage())

    governance = DummyGovernanceService()

    context = RuntimeContext.build(
        config=config,
        session_manager=sm,
        governance_service=governance,
    )

    engine = RuntimeEngine(context=context)

    request = RuntimeRequest(        
        tenant_id="test-tenant",
        user_id="test-user",
        session_id="test-session",
        message="test",
        agent_id="agent-1",
    )
    
    answer = await engine.run(request)

    assert governance.called is True
    assert governance.run_id == answer.run_id
    assert governance.agent_id == "agent-1"
