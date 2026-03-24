# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import asyncio
from collections.abc import Generator

import pytest
from redis import Redis

from intergrax.distributed.providers.redis_execution_semaphore import (
    RedisExecutionSemaphore,
)
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.pipelines.contract import RuntimePipeline
from intergrax.runtime.nexus.responses.response_schema import (
    RuntimeRequest,
    RuntimeAnswer,
)
from testing_support.builder import (
    build_runtime_config_deterministic,
    build_engine_harness,
)

pytestmark = pytest.mark.integration

REDIS_URL = "redis://localhost:6379/0"


@pytest.fixture
def redis_client() -> Generator[Redis, None, None]:
    client = Redis.from_url(REDIS_URL)
    client.flushdb()
    yield client
    client.flushdb()


class SlowPipeline(RuntimePipeline):
    async def _inner_run(self, state: RuntimeState) -> RuntimeAnswer:
        await asyncio.sleep(0.3)
        answer = RuntimeAnswer(answer="ok")
        state.runtime_answer = answer
        return answer


@pytest.mark.asyncio
async def test_runtime_execution_semaphore_limits_parallel_runs(
    redis_client: Redis,
) -> None:
    semaphore = RedisExecutionSemaphore(
        client=redis_client,
        ttl_seconds=60,
    )

    cfg = build_runtime_config_deterministic()

    # 🔹 Inject deterministic slow pipeline via official extension point
    cfg.pipeline = SlowPipeline()

    harness = build_engine_harness(cfg=cfg)

    harness.engine.context.execution_semaphore = semaphore
    harness.engine.context.max_parallel_per_tenant = 1

    request = RuntimeRequest(
        tenant_id="test-tenant",
        agent_id="agent_test",
        user_id="user_test",
        session_id="session_test",
        message="hello",
    )

    async def run_once():
        return await harness.engine.run(request)

    task1 = asyncio.create_task(run_once())
    await asyncio.sleep(0)  # ensure task1 acquires slot

    with pytest.raises(RuntimeError):
        await run_once()

    await task1

    result = await run_once()
    assert result.answer == "ok"