# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import pytest
from pydantic import BaseModel

from intergrax.contracts.idempotency_store import (
    ClaimResult,
    IdempotencyStore,
    InvocationClaim,
    InvocationStatus,
)
from intergrax.contracts.persistence_topology import PersistenceTopology
from intergrax.queueing.worker.execution import (
    execute_logical_task,
    IdempotencyLockConflictError,
)
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.tools.execution_models import ToolExecutionResult


class DummyOutput(BaseModel):
    value: str


class DummyIdempotencyStore(IdempotencyStore):
    @property
    def persistence_topology(self) -> PersistenceTopology:
        return PersistenceTopology.PROCESS_LOCAL

    def __init__(self) -> None:
        self._store: dict[str, tuple[InvocationStatus, ToolExecutionResult]] = {}

    def claim(
        self,
        tenant_id: str,
        key: str,
        owner_id: str,
        lease_seconds: int,
    ) -> ClaimResult:
        del owner_id, lease_seconds
        full_key = f"{tenant_id}:{key}"
        if full_key in self._store:
            status, result = self._store[full_key]
            if status == InvocationStatus.COMPLETED:
                from intergrax.contracts.idempotency_store import ClaimOutcome

                return ClaimResult(outcome=ClaimOutcome.REPLAY_COMPLETED, completed_result=result)
            raise RuntimeError("already exists")
        raise NotImplementedError

    def complete_with_claim(
        self,
        tenant_id: str,
        key: str,
        claim: InvocationClaim,
        result: ToolExecutionResult,
        completed_ttl_seconds: int | None = None,
    ) -> None:
        del claim, completed_ttl_seconds
        self.record_completed(tenant_id, key, result)

    def record_started(
        self,
        tenant_id: str,
        key: str,
        lease_seconds: int | None = None,
    ) -> None:
        full_key = f"{tenant_id}:{key}"
        if full_key in self._store:
            raise RuntimeError("already exists")
        # temporary placeholder
        self._store[full_key] = (InvocationStatus.STARTED, None)  # type: ignore

    def record_completed(
        self,
        tenant_id: str,
        key: str,
        result: ToolExecutionResult,
        completed_ttl_seconds: int | None = None,
    ) -> None:
        full_key = f"{tenant_id}:{key}"
        status, _ = self._store.get(full_key, (None, None))
        if status != InvocationStatus.STARTED:
            raise RuntimeError("invalid state")
        self._store[full_key] = (InvocationStatus.COMPLETED, result)

    def get_status(self, tenant_id: str, key: str):
        full_key = f"{tenant_id}:{key}"
        entry = self._store.get(full_key)
        if not entry:
            return None
        return entry[0]

    def get_completed_result(self, tenant_id: str, key: str):
        full_key = f"{tenant_id}:{key}"
        entry = self._store.get(full_key)
        if not entry:
            return None
        status, result = entry
        if status == InvocationStatus.COMPLETED:
            return result
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


def test_execute_with_idempotency_fresh(registry):
    store = DummyIdempotencyStore()

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


def test_execute_with_existing_result(registry):
    store = DummyIdempotencyStore()

    # first execution
    result1 = execute_logical_task(
        registry=registry,
        logical_task_name="task.a",
        tenant_id="t1",
        run_id="r1",
        payload=b"data",
        idempotency_key="k1",
        idempotency_store=store,
        lease_seconds=60,
    )

    # replay
    result2 = execute_logical_task(
        registry=registry,
        logical_task_name="task.a",
        tenant_id="t1",
        run_id="r2",
        payload=b"data",
        idempotency_key="k1",
        idempotency_store=store,
        lease_seconds=60,
    )

    assert result2.success is True
    assert result2.output.value == "ok"


def test_execute_lock_held(registry):
    store = DummyIdempotencyStore()

    # Manually insert STARTED
    store._store["t1:task.a:k1"] = (InvocationStatus.STARTED, None)

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