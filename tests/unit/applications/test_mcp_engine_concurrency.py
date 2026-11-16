# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest
from intergrax.applications._shared.mcp_engine_concurrency import (
    McpConcurrentToolCallLimitExceeded,
    McpEngineConcurrentCallAdmission,
)

pytestmark = [pytest.mark.unit]


@pytest.mark.asyncio
async def test_concurrent_admission_rejects_when_saturated() -> None:
    admission = McpEngineConcurrentCallAdmission(max_concurrent_calls=1)
    await admission.__aenter__()
    with pytest.raises(McpConcurrentToolCallLimitExceeded):
        await admission.__aenter__()
    await admission.__aexit__(None, None, None)
