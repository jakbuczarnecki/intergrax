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


class FailingGovernanceService:
    def evaluate(self, run_id: str, agent_id: str):
        raise RuntimeError("Policy blocked")


@pytest.mark.asyncio
async def test_runtime_handles_governance_failure_gracefully():
    config = RuntimeConfig(
        llm_adapter=None,
        enable_rag=False,
        enable_websearch=False,
    )
    config.production_mode = False

    sm = SessionManager(storage=InMemorySessionStorage())

    governance = FailingGovernanceService()

    context = RuntimeContext.build(
        config=config,
        session_manager=sm,
        governance_service=governance,
    )

    engine = RuntimeEngine(context=context)

    request = RuntimeRequest(
        user_id="u",
        session_id="s",
        message="test",
        metadata={"agent_id": "agent-1"},
    )

    # Should not raise despite governance failure
    answer = await engine.run(request)

    assert answer.run_id is not None
