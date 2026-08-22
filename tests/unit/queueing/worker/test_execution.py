# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import time
from typing import Optional

import pytest
from pydantic import BaseModel

from intergrax.contracts.idempotency_store import (
    ClaimOutcome,
    ClaimResult,
    IdempotencyStore,
    InvocationClaim,
    InvocationStatus,
    InvocationUncertaintyError,
)
from intergrax.contracts.lease_claim import StaleClaimError
from intergrax.contracts.persistence_topology import PersistenceTopology
from intergrax.queueing.worker.execution import (
    execute_logical_task,
    IdempotencyLockConflictError,
)
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.runtime.tools.in_memory_idempotency_store import InMemoryIdempotencyStore
from intergrax.tools.execution_models import ToolExecutionResult

pytestmark = pytest.mark.unit


class DummyOutput(BaseModel):
    value: str


class SpyIdempotencyStore(InMemoryIdempotencyStore):
    """Tracks claim protocol usage and optional outcome overrides."""

    def __init__(self) -> None:
        super().__init__()
        self.claim_calls: list[tuple[str, str, str, int]] = []
        self.complete_with_claim_calls: list[tuple[str, str, InvocationClaim]] = []
        self.record_started_calls = 0
        self.record_completed_calls = 0
        self._forced_outcome: ClaimOutcome | None = None
        self._crash_on_complete = False

    def claim(
        self,
        tenant_id: str,
        key: str,
        owner_id: str,
        lease_seconds: int,
    ) -> ClaimResult:
        self.claim_calls.append((tenant_id, key, owner_id, lease_seconds))
        if self._forced_outcome is not None:
            if self._forced_outcome == ClaimOutcome.BLOCKED_ACTIVE:
                return ClaimResult(outcome=ClaimOutcome.BLOCKED_ACTIVE)
            if self._forced_outcome == ClaimOutcome.UNCERTAIN:
                return ClaimResult(outcome=ClaimOutcome.UNCERTAIN)
        return super().claim(tenant_id, key, owner_id, lease_seconds)

    def complete_with_claim(
        self,
        tenant_id: str,
        key: str,
        claim: InvocationClaim,
        result: ToolExecutionResult,
        completed_ttl_seconds: Optional[int] = None,
    ) -> None:
        self.complete_with_claim_calls.append((tenant_id, key, claim))
        if self._crash_on_complete:
            raise RuntimeError("simulated crash before completion persistence")
        return super().complete_with_claim(
            tenant_id,
            key,
            claim,
            result,
            completed_ttl_seconds=completed_ttl_seconds,
        )

    def record_started(
        self,
        tenant_id: str,
        key: str,
        lease_seconds: int | None = None,
    ) -> None:
        self.record_started_calls += 1
        return super().record_started(tenant_id, key, lease_seconds)

    def record_completed(
        self,
        tenant_id: str,
        key: str,
        result: ToolExecutionResult,
        completed_ttl_seconds: int | None = None,
    ) -> None:
        self.record_completed_calls += 1
        return super().record_completed(
            tenant_id,
            key,
            result,
            completed_ttl_seconds=completed_ttl_seconds,
        )


class LegacyOnlyIdempotencyStore(IdempotencyStore):
    """Minimal store for lock-held regression using BLOCKED_ACTIVE claim outcome."""

    @property
    def persistence_topology(self) -> PersistenceTopology:
        return PersistenceTopology.PROCESS_LOCAL

    def __init__(self) -> None:
        self._started: set[tuple[str, str]] = set()

    def claim(
        self,
        tenant_id: str,
        key: str,
        owner_id: str,
        lease_seconds: int,
    ) -> ClaimResult:
        del owner_id, lease_seconds
        if (tenant_id, key) in self._started:
            return ClaimResult(outcome=ClaimOutcome.BLOCKED_ACTIVE)
        return ClaimResult(outcome=ClaimOutcome.ACQUIRED, claim=None)

    def complete_with_claim(
        self,
        tenant_id: str,
        key: str,
        claim: InvocationClaim,
        result: ToolExecutionResult,
        completed_ttl_seconds: int | None = None,
    ) -> None:
        del claim, result, completed_ttl_seconds

    def record_started(
        self,
        tenant_id: str,
        key: str,
        lease_seconds: int | None = None,
    ) -> None:
        del lease_seconds
        self._started.add((tenant_id, key))

    def record_completed(
        self,
        tenant_id: str,
        key: str,
        result: ToolExecutionResult,
        completed_ttl_seconds: int | None = None,
    ) -> None:
        del result, completed_ttl_seconds

    def get_status(self, tenant_id: str, key: str):
        if (tenant_id, key) in self._started:
            return InvocationStatus.STARTED
        return None

    def get_completed_result(self, tenant_id: str, key: str):
        return None


@pytest.fixture
def registry() -> TaskExecutionRegistry:
    registry = TaskExecutionRegistry()

    def handler(**kwargs):
        return ToolExecutionResult(
            success=True,
            output=DummyOutput(value="ok"),
            error=None,
        )

    registry.register("task.a", handler)
    return registry


def _counting_registry(counter: list[int]) -> TaskExecutionRegistry:
    registry = TaskExecutionRegistry()

    def handler(**kwargs):
        counter[0] += 1
        return ToolExecutionResult(
            success=True,
            output=DummyOutput(value="ok"),
            error=None,
        )

    registry.register("task.a", handler)
    return registry


def test_execute_without_idempotency(registry):
    result = execute_logical_task(
        registry=registry,
        logical_task_name="task.a",
        tenant_id="t1",
        run_id="r1",
        payload=b"data",
        idempotency_key=None,
        idempotency_store=None,
    )

    assert result.success is True
    assert result.output is not None
    assert isinstance(result.output, DummyOutput)


def test_r2_1_logical_task_uses_claim_protocol(registry):
    store = SpyIdempotencyStore()

    execute_logical_task(
        registry=registry,
        logical_task_name="task.a",
        tenant_id="t1",
        run_id="r1",
        payload=b"data",
        idempotency_key="k1",
        idempotency_store=store,
        lease_seconds=60,
    )

    assert len(store.claim_calls) == 1
    assert len(store.complete_with_claim_calls) == 1
    assert store.record_started_calls == 0
    assert store.record_completed_calls == 0


def test_execute_with_idempotency_fresh(registry):
    store = SpyIdempotencyStore()

    result = execute_logical_task(
        registry=registry,
        logical_task_name="task.a",
        tenant_id="t1",
        run_id="r1",
        payload=b"data",
        idempotency_key="k1",
        idempotency_store=store,
        lease_seconds=60,
    )

    assert result.success is True
    assert result.output is not None
    assert isinstance(result.output, DummyOutput)


def test_r2_5_completed_replay(registry):
    store = SpyIdempotencyStore()
    counter: list[int] = [0]
    counting_registry = _counting_registry(counter)

    result1 = execute_logical_task(
        registry=counting_registry,
        logical_task_name="task.a",
        tenant_id="t1",
        run_id="r1",
        payload=b"data",
        idempotency_key="k1",
        idempotency_store=store,
        lease_seconds=60,
    )

    result2 = execute_logical_task(
        registry=counting_registry,
        logical_task_name="task.a",
        tenant_id="t1",
        run_id="r2",
        payload=b"data",
        idempotency_key="k1",
        idempotency_store=store,
        lease_seconds=60,
    )

    assert result1.success is True
    assert result2.success is True
    assert result2.output.value == "ok"
    assert counter[0] == 1


def test_r2_2_active_claim_is_retryable_contention(registry):
    store = SpyIdempotencyStore()
    store._forced_outcome = ClaimOutcome.BLOCKED_ACTIVE
    counter: list[int] = [0]
    counting_registry = _counting_registry(counter)

    with pytest.raises(IdempotencyLockConflictError):
        execute_logical_task(
            registry=counting_registry,
            logical_task_name="task.a",
            tenant_id="t1",
            run_id="r1",
            payload=b"data",
            idempotency_key="k1",
            idempotency_store=store,
            lease_seconds=60,
        )

    assert counter[0] == 0


def test_r2_3_uncertain_is_not_lock_conflict(registry):
    store = SpyIdempotencyStore()
    store._forced_outcome = ClaimOutcome.UNCERTAIN
    counter: list[int] = [0]
    counting_registry = _counting_registry(counter)

    with pytest.raises(InvocationUncertaintyError):
        execute_logical_task(
            registry=counting_registry,
            logical_task_name="task.a",
            tenant_id="t1",
            run_id="r1",
            payload=b"data",
            idempotency_key="k1",
            idempotency_store=store,
            lease_seconds=60,
        )

    assert counter[0] == 0


def test_execute_lock_held(registry):
    store = LegacyOnlyIdempotencyStore()
    store._started.add(("t1", "task.a:k1"))

    with pytest.raises(IdempotencyLockConflictError):
        execute_logical_task(
            registry=registry,
            logical_task_name="task.a",
            tenant_id="t1",
            run_id="r1",
            payload=b"data",
            idempotency_key="k1",
            idempotency_store=store,
            lease_seconds=60,
        )


def test_r2_4_crash_after_handler_effect():
    store = SpyIdempotencyStore()
    counter: list[int] = [0]
    counting_registry = _counting_registry(counter)
    store._crash_on_complete = True

    with pytest.raises(RuntimeError, match="simulated crash"):
        execute_logical_task(
            registry=counting_registry,
            logical_task_name="task.a",
            tenant_id="t1",
            run_id="r1",
            payload=b"data",
            idempotency_key="k1",
            idempotency_store=store,
            lease_seconds=1,
        )

    assert counter[0] == 1
    time.sleep(1.2)

    with pytest.raises(InvocationUncertaintyError):
        execute_logical_task(
            registry=counting_registry,
            logical_task_name="task.a",
            tenant_id="t1",
            run_id="r2",
            payload=b"data",
            idempotency_key="k1",
            idempotency_store=store,
            lease_seconds=1,
        )

    assert counter[0] == 1


def test_r2_6_completion_uses_original_claim(registry):
    store = SpyIdempotencyStore()

    execute_logical_task(
        registry=registry,
        logical_task_name="task.a",
        tenant_id="t1",
        run_id="r1",
        payload=b"data",
        idempotency_key="k1",
        idempotency_store=store,
        lease_seconds=60,
    )

    acquired_claim = store.complete_with_claim_calls[0][2]
    assert acquired_claim.owner_id == store.claim_calls[0][2]
    assert acquired_claim.fence == 1


def test_r2_7_stale_completion_propagates(registry):
    store = SpyIdempotencyStore()
    claim_result = store.claim("t1", "task.a:k1", "owner-a", 60)
    assert claim_result.claim is not None

    # Supersede active claim with a different owner/fence.
    store._store[("t1", "task.a:k1")] = store._store[("t1", "task.a:k1")]
    entry = store._store[("t1", "task.a:k1")]
    stale_claim = claim_result.claim
    superseding = InvocationClaim(
        tenant_id="t1",
        key="task.a:k1",
        owner_id="owner-b",
        lease_expires_at=stale_claim.lease_expires_at,
        fence=stale_claim.fence + 1,
    )
    entry.claim = superseding

    result = ToolExecutionResult(
        success=True,
        output=DummyOutput(value="ok"),
        error=None,
    )

    with pytest.raises(StaleClaimError):
        store.complete_with_claim("t1", "task.a:k1", stale_claim, result)
