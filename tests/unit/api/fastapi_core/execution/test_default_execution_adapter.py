from intergrax.fastapi_core.execution.adapters.default_adapter import DefaultExecutionAdapter
from intergrax.fastapi_core.execution.models import ExecutionRequest

import pytest

@pytest.mark.asyncio
async def test_default_execution_adapter_is_noop():
    adapter = DefaultExecutionAdapter()

    request = ExecutionRequest(
        run_id="r1",
        tenant_id="t1",
        user_id=None,
        input_payload={},
    )

    await adapter.start_execution(request)
