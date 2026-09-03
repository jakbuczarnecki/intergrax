# © Artur Czarnecki. All rights reserved.

"""PCM-02 side-effect coordination tests (idempotency claim/lease/fence)."""

from __future__ import annotations

import threading
import time

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import BaseModel

from intergrax.contracts.idempotency_store import (
    ActiveInvocationClaimError,
    ClaimOutcome,
    InvocationClaim,
    InvocationStatus,
    InvocationUncertaintyError,
)
from intergrax.contracts.lease_claim import StaleClaimError
from intergrax.contracts.persistence_topology import PersistenceTopology
from intergrax.distributed.providers.redis_idempotency_store import RedisIdempotencyStore
from intergrax.runtime.tools.idempotency_pre_effect_coordinator import (
    IdempotencyPreEffectCoordinator,
)
from intergrax.runtime.tools.in_memory_idempotency_store import InMemoryIdempotencyStore
from intergrax.runtime.tools.sqlite_idempotency_store import SQLiteIdempotencyStore
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.execution_models import ToolExecutionRequest, ToolExecutionResult
from intergrax.tools.registry import ToolRegistry
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


class DummyInput(BaseModel):
    value: int


class DummyOutput(BaseModel):
    result: int


class CountingExecutor:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, request: ToolExecutionRequest[DummyInput]) -> DummyOutput:
        self.calls += 1
        return DummyOutput(result=request.input.value * 2)


class DummyHandler:
    def execute(self, request: ToolExecutionRequest[DummyInput]) -> DummyOutput:
        return DummyOutput(result=request.input.value * 2)


class DummyState:
    def __init__(self) -> None:
        self._tenant_id = "tenant_test"

    @property
    def tenant_id(self) -> str:
        return self._tenant_id

    @property
    def context(self):
        return type("Ctx", (), {"config": type("Cfg", (), {"policy_bundle": None})()})()

    def trace_event(self, *args, **kwargs) -> None:
        del args, kwargs


def _build_invoker(store: InMemoryIdempotencyStore, executor: CountingExecutor) -> RuntimeToolInvoker:
    registry = ToolRegistry()
    registry.register(
        contract=ToolContract(
            tool_id="double",
            name="double",
            description="double value",
            input_schema=DummyInput,
            output_schema=DummyOutput,
            error_mapping={},
            side_effects=True,
        ),
        handler=DummyHandler(),
    )
    coordinator = IdempotencyPreEffectCoordinator(
        idempotency_store=store,
        lease_seconds=2,
    )
    return RuntimeToolInvoker(
        registry=registry,
        executor=executor,
        pre_effect_coordinator=coordinator,
    )


def _request() -> ToolExecutionRequest[DummyInput]:
    return ToolExecutionRequest(
        run_id="run1",
        step_id="step1",
        tool_id="double",
        input=DummyInput(value=5),
        idempotency_key="key-123",
    )


def test_a1_one_active_claim() -> None:
    store = InMemoryIdempotencyStore()
    results: list[ClaimOutcome] = []
    barrier = threading.Barrier(2)

    def racer(owner: str) -> None:
        barrier.wait()
        outcome = store.claim("tenant-a", "race-key", owner, lease_seconds=30)
        results.append(outcome.outcome)

    t1 = threading.Thread(target=racer, args=("owner-a",))
    t2 = threading.Thread(target=racer, args=("owner-b",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    acquired = [item for item in results if item == ClaimOutcome.ACQUIRED]
    blocked = [item for item in results if item == ClaimOutcome.BLOCKED_ACTIVE]
    assert len(acquired) == 1
    assert len(blocked) == 1


def test_a2_active_claim_blocks_second_execution() -> None:
    store = InMemoryIdempotencyStore()
    executor = CountingExecutor()
    invoker = _build_invoker(store, executor)
    state = DummyState()
    request = _request()

    store.claim("tenant_test", "key-123", "owner-a", lease_seconds=30)
    with pytest.raises(ActiveInvocationClaimError):
        invoker.invoke(state=state, agent_id="agent-a", request=request)
    assert executor.calls == 0


def test_a3_completed_replays_result() -> None:
    store = InMemoryIdempotencyStore()
    executor = CountingExecutor()
    invoker = _build_invoker(store, executor)
    state = DummyState()
    request = _request()

    r1 = invoker.invoke(state=state, agent_id="agent-a", request=request)
    r2 = invoker.invoke(state=state, agent_id="agent-a", request=request)
    assert r1.success and r2.success
    assert r1.output == r2.output
    assert executor.calls == 1


def test_a4_crash_after_effect_becomes_uncertain() -> None:
    store = InMemoryIdempotencyStore()
    executor = CountingExecutor()
    invoker = _build_invoker(store, executor)
    state = DummyState()
    request = _request()

    claim = store.claim("tenant_test", "key-123", "owner-crash", lease_seconds=1)
    assert claim.outcome == ClaimOutcome.ACQUIRED
    executor.execute(request)
    time.sleep(1.2)

    with pytest.raises(InvocationUncertaintyError):
        invoker.invoke(state=state, agent_id="agent-a", request=request)
    assert executor.calls == 1
    assert store.get_status("tenant_test", "key-123") == InvocationStatus.UNCERTAIN


def test_a5_stale_completion_rejected() -> None:
    store = InMemoryIdempotencyStore()
    acquired = store.claim("tenant-a", "stale-key", "owner-a", lease_seconds=30)
    assert acquired.claim is not None
    stale_claim = InvocationClaim(
        tenant_id="tenant-a",
        key="stale-key",
        owner_id="owner-a",
        lease_expires_at=acquired.claim.lease_expires_at,
        fence=1,
    )
    entry = store._store[("tenant-a", "stale-key")]  # noqa: SLF001
    entry.claim = acquired.claim.model_copy(update={"fence": 2, "owner_id": "owner-b"})
    result = ToolExecutionResult.ok(DummyOutput(result=1))
    with pytest.raises(StaleClaimError):
        store.complete_with_claim("tenant-a", "stale-key", stale_claim, result)


def test_a5b_expired_lease_completion_rejected() -> None:
    store = InMemoryIdempotencyStore()
    acquired = store.claim("tenant-a", "expired-complete-key", "owner-a", lease_seconds=30)
    assert acquired.claim is not None
    expired_claim = acquired.claim
    entry = store._store[("tenant-a", "expired-complete-key")]  # noqa: SLF001
    entry.claim = acquired.claim.model_copy(
        update={"lease_expires_at": datetime.now(UTC) - timedelta(seconds=1)},
    )
    result = ToolExecutionResult.ok(DummyOutput(result=1))
    with pytest.raises(StaleClaimError):
        store.complete_with_claim("tenant-a", "expired-complete-key", expired_claim, result)


def test_a5c_expired_lease_mark_uncertain_rejected() -> None:
    store = InMemoryIdempotencyStore()
    acquired = store.claim("tenant-a", "expired-uncertain-key", "owner-a", lease_seconds=30)
    assert acquired.claim is not None
    expired_claim = acquired.claim
    entry = store._store[("tenant-a", "expired-uncertain-key")]  # noqa: SLF001
    entry.claim = acquired.claim.model_copy(
        update={"lease_expires_at": datetime.now(UTC) - timedelta(seconds=1)},
    )
    with pytest.raises(StaleClaimError):
        store.mark_uncertain_with_claim("tenant-a", "expired-uncertain-key", expired_claim)


def test_a5d_superseded_fence_current_owner_completes() -> None:
    store = InMemoryIdempotencyStore()
    acquired_a = store.claim("tenant-a", "fence-handoff-key", "owner-a", lease_seconds=30)
    assert acquired_a.claim is not None
    stale_claim = acquired_a.claim
    entry = store._store[("tenant-a", "fence-handoff-key")]  # noqa: SLF001
    current_claim = acquired_a.claim.model_copy(update={"fence": 2, "owner_id": "owner-b"})
    entry.claim = current_claim
    result = ToolExecutionResult.ok(DummyOutput(result=7))
    with pytest.raises(StaleClaimError):
        store.complete_with_claim("tenant-a", "fence-handoff-key", stale_claim, result)
    store.complete_with_claim("tenant-a", "fence-handoff-key", current_claim, result)
    assert store.get_status("tenant-a", "fence-handoff-key") == InvocationStatus.COMPLETED


def test_a9_tenant_isolation_same_key() -> None:
    store = InMemoryIdempotencyStore()
    claim_a = store.claim("tenant-a", "shared-key", "owner-a", lease_seconds=30)
    claim_b = store.claim("tenant-b", "shared-key", "owner-b", lease_seconds=30)
    assert claim_a.outcome == ClaimOutcome.ACQUIRED
    assert claim_b.outcome == ClaimOutcome.ACQUIRED
    assert claim_a.claim is not None and claim_b.claim is not None
    assert claim_a.claim.fence == 1 and claim_b.claim.fence == 1


def test_a6_current_owner_completes() -> None:
    store = InMemoryIdempotencyStore()
    acquired = store.claim("tenant-a", "complete-key", "owner-b", lease_seconds=30)
    assert acquired.claim is not None
    result = ToolExecutionResult.ok(DummyOutput(result=10))
    store.complete_with_claim("tenant-a", "complete-key", acquired.claim, result)
    assert store.get_status("tenant-a", "complete-key") == InvocationStatus.COMPLETED


def test_a7_false_exactly_once_claim_removed() -> None:
    doc = RuntimeToolInvoker.__doc__ or ""
    assert "enforces exactly-once" not in doc.lower()


def test_a8_topology_regression() -> None:
    assert InMemoryIdempotencyStore().persistence_topology is PersistenceTopology.PROCESS_LOCAL
    assert SQLiteIdempotencyStore(":memory:").persistence_topology is (
        PersistenceTopology.DURABLE_SINGLE_HOST
    )

    class _FakeRedis:
        def register_script(self, _script: str) -> object:
            return object()

    assert RedisIdempotencyStore(_FakeRedis()).persistence_topology is (
        PersistenceTopology.SHARED_MULTI_HOST
    )
