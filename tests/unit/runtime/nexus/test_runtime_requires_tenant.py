# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import pytest

from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_runtime_engine_rejects_empty_tenant_id(harness_static):
    engine = harness_static.engine

    request = RuntimeRequest(
        agent_id="agent_test",
        user_id="user_test",
        session_id="session_test",
        message="Hello",
        tenant_id="",  # invalid
    )

    with pytest.raises(ValueError):
        await engine.run(request)
