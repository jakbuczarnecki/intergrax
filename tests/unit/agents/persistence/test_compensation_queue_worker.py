# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.agents.persistence.compensation_enqueue import build_compensation_idempotency_key
from intergrax.agents.persistence.compensation_queue_store import CompensationJob, InMemoryCompensationQueueStore
from intergrax.agents.persistence.compensation_queue_worker import drain_pending_compensation_jobs
from intergrax.agents.persistence.declarative_tool_executor import (
    CallableDeclarativeToolInvoker,
    DeclarativeToolInvokeResult,
)
from intergrax.contracts.side_effect import CompensationRequest
from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from tests.unit.agents.persistence.compensation_execution_test_support import (
    build_test_admitted_compensation_side_effect_execution,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.mark.asyncio
async def test_drain_pending_compensation_jobs_marks_completed() -> None:
    store = InMemoryCompensationQueueStore()
    key = build_compensation_idempotency_key("acp:worker")
    store.enqueue(
        CompensationJob(
            run_id=str(mint_run_id()),
            task_id=str(mint_task_id()),
            tenant_id="tenant-a",
            agent_id="agent-a",
            step_index=0,
            request=CompensationRequest(
                original_side_effect_id="se-worker",
                compensation_tool_id="email.recall",
                args={},
                idempotency_key=key,
            ),
        ),
    )
    invoked: list[str] = []

    async def _invoke(**kwargs):  # type: ignore[no-untyped-def]
        invoked.append(kwargs["tool_id"])
        return DeclarativeToolInvokeResult(status="success")

    results = await drain_pending_compensation_jobs(
        store,
        tenant_id="tenant-a",
        side_effect_execution=build_test_admitted_compensation_side_effect_execution(
            CallableDeclarativeToolInvoker(_invoke),
        ),
    )
    assert invoked == ["email.recall"]
    assert results[0].status == "compensated"
    assert store.list_pending("tenant-a") == []
