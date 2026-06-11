# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.agents.persistence.compensation_enqueue import build_compensation_idempotency_key
from intergrax.agents.persistence.compensation_queue_store import (
    CompensationJob,
    CompensationJobStatus,
    InMemoryCompensationQueueStore,
    SQLiteCompensationQueueStore,
)
from intergrax.contracts.side_effect import CompensationRequest

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _sample_job(*, tenant_id: str = "tenant-a") -> CompensationJob:
    key = build_compensation_idempotency_key("acp:orig")
    return CompensationJob(
        run_id="run-1",
        tenant_id=tenant_id,
        agent_id="agent-a",
        step_index=0,
        request=CompensationRequest(
            original_side_effect_id="se-1",
            compensation_tool_id="email.recall",
            args={"original_external_ref": "msg-1"},
            idempotency_key=key,
        ),
    )


def test_in_memory_compensation_queue_enqueue_and_list_pending() -> None:
    store = InMemoryCompensationQueueStore()
    job = _sample_job()
    store.enqueue(job)
    pending = store.list_pending("tenant-a")
    assert len(pending) == 1
    assert pending[0].request.compensation_tool_id == "email.recall"


def test_sqlite_compensation_queue_persists_job(tmp_path) -> None:
    db_path = tmp_path / "compensation_queue.db"
    store = SQLiteCompensationQueueStore(db_path)
    job = _sample_job()
    store.enqueue(job)
    reloaded = SQLiteCompensationQueueStore(db_path)
    loaded = reloaded.get_by_idempotency_key("tenant-a", job.request.idempotency_key)
    assert loaded is not None
    assert loaded.status == CompensationJobStatus.PENDING
