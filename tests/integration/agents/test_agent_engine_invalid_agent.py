# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import pytest

from intergrax.runtime.nexus.agents.agent_engine import AgentEngine
from testing_support.builder import build_runtime_request_for_tests


# ----------------------------------------
# TEST
# ----------------------------------------
@pytest.mark.asyncio
async def test_agent_engine_raises_for_unknown_agent():
    engine = AgentEngine({})  # no agents registered

    request = build_runtime_request_for_tests(
        seed="unknown-agent",
        tenant_id="t1",
        user_id="u1",
        session_id="s1",
        agent_id="unknown",
        message="hello",
    )

    with pytest.raises(Exception):
        await engine.run(request)