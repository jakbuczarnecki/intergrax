# © Artur Czarnecki. All rights reserved.

"""PCM-03 compensation queue claim/lease/fence tests."""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime, timedelta

import pytest

from intergrax.agents.persistence.compensation_enqueue import build_compensation_idempotency_key
from intergrax.agents.persistence.compensation_queue_store import (
    CompensationClaim,
    CompensationJob,
    CompensationJobStatus,
    CompensationQueueStore,
    InMemoryCompensationQueueStore,
    SQLiteCompensationQueueStore,
)
from intergrax.agents.persistence.compensation_queue_worker import drain_pending_compensation_jobs
from intergrax.agents.persistence.declarative_tool_executor import (
    CallableDeclarativeToolInvoker,
    DeclarativeToolInvokeResult,
)
from intergrax.contracts.lease_claim import StaleClaimError
from intergrax.contracts.side_effect import CompensationRequest

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


def _sample_job(*, tenant_id: str = "tenant-a", key_suffix: str = "orig") -> CompensationJob:
    key = build_compensation_idempotency_key(f"acp:{key_suffix}")
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


@pytest.mark.asyncio
async def test_b1_atomic_compensation_claim() -> None:
    store = InMemoryCompensationQueueStore()
    job = _sample_job()
    store.enqueue(job)
    winners: list[str] = []
    barrier = threading.Barrier(2)

    def racer(owner: str) -> None:
        barrier.wait()
        claims = store.claim_pending("tenant-a", owner, lease_seconds=30, limit=1)
        if claims:
            winners.append(owner)

    t1 = threading.Thread(target=racer, args=("worker-a",))
    t2 = threading.Thread(target=racer, args=("worker-b",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert winners == ["worker-a"] or winners == ["worker-b"]
    assert len(winners) == 1


@pytest.mark.asyncio
async def test_b2_second_worker_does_not_invoke() -> None:
    store = InMemoryCompensationQueueStore()
    job = _sample_job(key_suffix="race")
    store.enqueue(job)
    invoked: list[str] = []

    async def _invoke(**kwargs):  # type: ignore[no-untyped-def]
        invoked.append(kwargs["tool_id"])
        return DeclarativeToolInvokeResult(status="success")

    invoker = CallableDeclarativeToolInvoker(_invoke)
    store.claim_pending("tenant-a", "worker-a", lease_seconds=30, limit=1)
    await drain_pending_compensation_jobs(
        store,
        tenant_id="tenant-a",
        invoker=invoker,
        owner_id="worker-b",
        limit=1,
    )
    assert invoked == []


@pytest.mark.asyncio
async def test_b3_running_ownership_stored() -> None:
    store = InMemoryCompensationQueueStore()
    store.enqueue(_sample_job(key_suffix="own"))
    claims = store.claim_pending("tenant-a", "worker-own", lease_seconds=30, limit=1)
    assert len(claims) == 1
    claim = claims[0]
    assert claim.owner_id == "worker-own"
    assert claim.fence == 1
    assert claim.lease_expires_at > datetime.now(UTC)
    assert claim.job.status == CompensationJobStatus.RUNNING


@pytest.mark.asyncio
async def test_a1_crash_after_compensation_effect_becomes_uncertain() -> None:
    store = InMemoryCompensationQueueStore()
    job = _sample_job(key_suffix="crash-effect")
    store.enqueue(job)
    invoked: list[str] = []

    async def _invoke(**kwargs):  # type: ignore[no-untyped-def]
        invoked.append(kwargs["tool_id"])
        return DeclarativeToolInvokeResult(status="success")

    invoker = CallableDeclarativeToolInvoker(_invoke)
    claim = store.claim_pending("tenant-a", "worker-a", lease_seconds=1, limit=1)[0]
    await invoker.invoke(
        tool_id=claim.job.request.compensation_tool_id,
        args=claim.job.request.args,
        idempotency_key=claim.job.request.idempotency_key,
    )
    time.sleep(1.2)
    second_claims = store.claim_pending("tenant-a", "worker-b", lease_seconds=30, limit=1)
    await drain_pending_compensation_jobs(
        store,
        tenant_id="tenant-a",
        invoker=invoker,
        owner_id="worker-b",
        limit=1,
    )
    loaded = store.get_by_idempotency_key("tenant-a", job.request.idempotency_key)
    assert invoked == ["email.recall"]
    assert second_claims == []
    assert loaded is not None
    assert loaded.status == CompensationJobStatus.UNCERTAIN


@pytest.mark.asyncio
async def test_a2_uncertain_is_not_claimable() -> None:
    store = InMemoryCompensationQueueStore()
    job = _sample_job(key_suffix="uncertain")
    store.enqueue(job)
    claim = store.claim_pending("tenant-a", "worker-a", lease_seconds=1, limit=1)[0]

    async def _invoke(**kwargs):  # type: ignore[no-untyped-def]
        return DeclarativeToolInvokeResult(status="success")

    await CallableDeclarativeToolInvoker(_invoke).invoke(
        tool_id=claim.job.request.compensation_tool_id,
        args=claim.job.request.args,
        idempotency_key=claim.job.request.idempotency_key,
    )
    time.sleep(1.2)
    store.claim_pending("tenant-a", "worker-b", lease_seconds=30, limit=1)
    assert store.claim_pending("tenant-a", "worker-c", lease_seconds=30, limit=1) == []
    assert store.list_uncertain("tenant-a")


@pytest.mark.asyncio
async def test_a3_explicit_retryable_still_claimable() -> None:
    store = InMemoryCompensationQueueStore()
    store.enqueue(_sample_job(key_suffix="retryable"))
    claim = store.claim_pending("tenant-a", "worker-a", lease_seconds=30, limit=1)[0]
    store.fail_claim(claim, "known transient failure", retryable=True)
    retry_claim = store.claim_pending("tenant-a", "worker-b", lease_seconds=30, limit=1)[0]
    assert retry_claim.fence > claim.fence
    assert retry_claim.owner_id == "worker-b"
    loaded = store.get_by_idempotency_key("tenant-a", claim.idempotency_key)
    assert loaded is not None
    assert loaded.status == CompensationJobStatus.RUNNING


def test_b1_no_unfenced_terminal_mutation_api() -> None:
    abstract = set(CompensationQueueStore.__abstractmethods__)
    assert "complete_claim" in abstract
    assert "fail_claim" in abstract
    assert "mark_completed" not in abstract
    assert "mark_failed" not in abstract


@pytest.mark.asyncio
async def test_b4_expired_running_becomes_uncertain_not_reclaimed() -> None:
    store = InMemoryCompensationQueueStore()
    job = _sample_job(key_suffix="uncertain-reclaim")
    store.enqueue(job)
    first = store.claim_pending("tenant-a", "worker-a", lease_seconds=1, limit=1)[0]
    time.sleep(1.2)
    second = store.claim_pending("tenant-a", "worker-b", lease_seconds=30, limit=1)
    loaded = store.get_by_idempotency_key("tenant-a", job.request.idempotency_key)
    assert second == []
    assert loaded is not None
    assert loaded.status == CompensationJobStatus.UNCERTAIN
    assert loaded.fence == first.fence


@pytest.mark.asyncio
async def test_b5_old_worker_completion_rejected() -> None:
    store = InMemoryCompensationQueueStore()
    store.enqueue(_sample_job(key_suffix="stale"))
    first = store.claim_pending("tenant-a", "worker-a", lease_seconds=30, limit=1)[0]
    store.fail_claim(first, "known failure", retryable=True)
    second = store.claim_pending("tenant-a", "worker-b", lease_seconds=30, limit=1)[0]
    assert second.fence > first.fence
    with pytest.raises(StaleClaimError):
        store.complete_claim(first)


@pytest.mark.asyncio
async def test_b6_current_worker_completes() -> None:
    store = InMemoryCompensationQueueStore()
    job = _sample_job(key_suffix="complete")
    store.enqueue(job)
    claim = store.claim_pending("tenant-a", "worker-b", lease_seconds=30, limit=1)[0]
    store.complete_claim(claim)
    loaded = store.get_by_idempotency_key("tenant-a", job.request.idempotency_key)
    assert loaded is not None
    assert loaded.status == CompensationJobStatus.COMPLETED


@pytest.mark.asyncio
async def test_b7_worker_failure_terminal_semantics() -> None:
    store = InMemoryCompensationQueueStore()
    store.enqueue(_sample_job(key_suffix="fail"))
    claim = store.claim_pending("tenant-a", "worker-fail", lease_seconds=30, limit=1)[0]
    store.fail_claim(claim, "boom", retryable=False)
    loaded = store.get_by_idempotency_key("tenant-a", claim.idempotency_key)
    assert loaded is not None
    assert loaded.status == CompensationJobStatus.FAILED
    assert loaded.error == "boom"


@pytest.mark.asyncio
async def test_b8_sqlite_claim_atomicity(tmp_path) -> None:
    db_path = tmp_path / "compensation_race.db"
    store_a = SQLiteCompensationQueueStore(db_path)
    store_b = SQLiteCompensationQueueStore(db_path)
    store_a.enqueue(_sample_job(key_suffix="sqlite-race"))
    winners: list[str] = []
    barrier = threading.Barrier(2)

    def racer(store: SQLiteCompensationQueueStore, owner: str) -> None:
        barrier.wait()
        claims = store.claim_pending("tenant-a", owner, lease_seconds=30, limit=1)
        if claims:
            winners.append(owner)

    t1 = threading.Thread(target=racer, args=(store_a, "conn-a"))
    t2 = threading.Thread(target=racer, args=(store_b, "conn-b"))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert len(winners) == 1


@pytest.mark.asyncio
async def test_b9_inmemory_positive_control_lifecycle() -> None:
    store = InMemoryCompensationQueueStore()
    job = _sample_job(key_suffix="lifecycle")
    store.enqueue(job)
    claim = store.claim_pending("tenant-a", "worker-lc", lease_seconds=30, limit=1)[0]
    assert claim.job.status == CompensationJobStatus.RUNNING
    store.complete_claim(claim)
    assert store.list_pending("tenant-a") == []


@pytest.mark.asyncio
async def test_b10_idempotency_key_is_not_ownership() -> None:
    store = InMemoryCompensationQueueStore()
    job = _sample_job(key_suffix="idem")
    store.enqueue(job)
    invoked: list[str] = []

    async def _invoke(**kwargs):  # type: ignore[no-untyped-def]
        invoked.append(kwargs["idempotency_key"])
        return DeclarativeToolInvokeResult(status="success")

    store.claim_pending("tenant-a", "worker-a", lease_seconds=30, limit=1)
    await drain_pending_compensation_jobs(
        store,
        tenant_id="tenant-a",
        invoker=CallableDeclarativeToolInvoker(_invoke),
        owner_id="worker-b",
        limit=1,
    )
    assert invoked == []
