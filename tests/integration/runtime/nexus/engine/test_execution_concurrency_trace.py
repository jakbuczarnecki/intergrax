# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import pytest

from intergrax.runtime.nexus.engine.runtime import RuntimeEngine
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.distributed.contracts.execution_semaphore import DistributedExecutionSemaphore, ExecutionSlot
from tests._support.builder import build_engine_harness, build_runtime_config_deterministic


class FakeExecutionSemaphore(DistributedExecutionSemaphore):
    def __init__(self, allow: bool):
        self.allow = allow
        self.acquired = False
        self.released = False

    def acquire(
        self,
        *,
        tenant_id: str,
        max_parallel: int,
    ) -> ExecutionSlot | None:
        if not self.allow:
            return None
        self.acquired = True
        return ExecutionSlot(slot_id="fake")

    def release(
        self,
        *,
        tenant_id: str,
        slot: ExecutionSlot,
    ) -> None:
        self.released = True


@pytest.mark.asyncio
async def test_execution_concurrency_trace_success():
    cfg = build_runtime_config_deterministic()
    harness = build_engine_harness(cfg=cfg)

    harness.engine.context.execution_semaphore = FakeExecutionSemaphore(allow=True)
    harness.engine.context.max_parallel_per_tenant = 1

    request = RuntimeRequest(
        tenant_id="tenant_A",
        user_id="user_A",
        session_id="session_A",
        agent_id="agent_A",
        message="hello",
    )

    answer = await harness.engine.run(request)

    steps = [e.step for e in answer.trace_events]

    assert "execution.acquire.success" in steps
    assert "execution.release" in steps


@pytest.mark.asyncio
async def test_execution_concurrency_trace_rejected():
    cfg = build_runtime_config_deterministic()
    harness = build_engine_harness(cfg=cfg)

    harness.engine.context.execution_semaphore = FakeExecutionSemaphore(allow=False)
    harness.engine.context.max_parallel_per_tenant = 1

    request = RuntimeRequest(
        tenant_id="tenant_A",
        user_id="user_A",
        session_id="session_A",
        agent_id="agent_A",
        message="hello",
    )

    with pytest.raises(RuntimeError):
        await harness.engine.run(request)

    
@pytest.mark.asyncio
async def test_execution_slot_long_hold_warning():
    from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest

    cfg = build_runtime_config_deterministic()
    cfg.execution_slot_warn_threshold_ms = 0  # always warn

    harness = build_engine_harness(cfg=cfg)
    harness.engine.context.execution_semaphore = FakeExecutionSemaphore(allow=True)
    harness.engine.context.max_parallel_per_tenant = 1

    request = RuntimeRequest(
        tenant_id="tenant_A",
        user_id="user_A",
        session_id="session_A",
        agent_id="agent_A",
        message="hello",
    )

    answer = await harness.engine.run(request)

    steps = [e.step for e in answer.trace_events]

    assert "execution.slot_long_hold_warning" in steps