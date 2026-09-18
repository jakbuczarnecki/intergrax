# © Artur Czarnecki. All rights reserved.

"""MEM-FINAL-AUDIT-3-R: Task Memory parallel-domain security/lifecycle/resilience proofs."""

from __future__ import annotations

import pytest

from intergrax.runtime.task_memory import InMemoryTaskMemoryStore, TaskMemoryCoordinator
from intergrax.runtime.task_memory.models import TaskMemoryRecord
from intergrax.runtime.task_memory.persistence_contract import TaskMemoryPersistence

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class FailingTaskMemoryStore(InMemoryTaskMemoryStore):
    """Contract-based fault injection for TaskMemoryPersistence."""

    def __init__(self) -> None:
        super().__init__()
        self.fail_operation: str = ""
        self.fail_tenant: str | None = None

    def put(self, record: TaskMemoryRecord) -> TaskMemoryRecord:
        if self.fail_operation == "put" and (
            self.fail_tenant is None or record.tenant_id == self.fail_tenant
        ):
            raise RuntimeError("task memory put failed")
        return super().put(record)

    def get(
        self,
        *,
        tenant_id: str,
        task_id: str,
        namespace: str,
        key: str,
    ) -> TaskMemoryRecord | None:
        if self.fail_operation == "get" and (
            self.fail_tenant is None or tenant_id == self.fail_tenant
        ):
            raise RuntimeError("task memory get failed")
        return super().get(
            tenant_id=tenant_id,
            task_id=task_id,
            namespace=namespace,
            key=key,
        )

    def delete(
        self,
        *,
        tenant_id: str,
        task_id: str,
        namespace: str,
        key: str,
    ) -> bool:
        if self.fail_operation == "delete" and (
            self.fail_tenant is None or tenant_id == self.fail_tenant
        ):
            raise RuntimeError("task memory delete failed")
        return super().delete(
            tenant_id=tenant_id,
            task_id=task_id,
            namespace=namespace,
            key=key,
        )


def _record(
    *,
    tenant_id: str,
    task_id: str,
    namespace: str = "ns",
    key: str = "k",
    value: dict[str, object] | None = None,
) -> TaskMemoryRecord:
    return TaskMemoryRecord(
        tenant_id=tenant_id,
        task_id=task_id,
        namespace=namespace,
        key=key,
        value=value or {"marker": tenant_id},
    )


def test_in_memory_task_store_documents_caller_serialized_concurrency() -> None:
    doc = InMemoryTaskMemoryStore.__doc__ or ""
    lowered = doc.lower()
    assert "caller" in lowered
    assert "thread-safe" in lowered or "not thread" in lowered


def test_task_memory_store_tenant_isolation_at_persistence_layer() -> None:
    store = InMemoryTaskMemoryStore()
    store.put(_record(tenant_id="tenant-a", task_id="task-x", value={"v": "a"}))
    assert (
        store.get(
            tenant_id="tenant-b",
            task_id="task-x",
            namespace="ns",
            key="k",
        )
        is None
    )


def test_task_memory_store_task_isolation_at_persistence_layer() -> None:
    store = InMemoryTaskMemoryStore()
    store.put(_record(tenant_id="tenant-a", task_id="task-x", value={"v": "x"}))
    assert (
        store.get(
            tenant_id="tenant-a",
            task_id="task-y",
            namespace="ns",
            key="k",
        )
        is None
    )


def test_task_memory_coordinator_replace_semantics_on_second_write() -> None:
    store = InMemoryTaskMemoryStore()
    TaskMemoryCoordinator.write(
        store,
        tenant_id="t1",
        task_id="task_1",
        namespace="ns",
        key="draft",
        value={"version": 1},
    )
    second = TaskMemoryCoordinator.write(
        store,
        tenant_id="t1",
        task_id="task_1",
        namespace="ns",
        key="draft",
        value={"version": 2},
    )
    loaded = TaskMemoryCoordinator.read(
        store,
        tenant_id="t1",
        task_id="task_1",
        namespace="ns",
        key="draft",
    )
    assert loaded is not None
    assert loaded.value["version"] == 2
    assert second.record_id == loaded.record_id


def test_task_memory_delete_is_idempotent_on_missing_key() -> None:
    store = InMemoryTaskMemoryStore()
    first = store.delete(
        tenant_id="t1",
        task_id="task_1",
        namespace="ns",
        key="missing",
    )
    second = store.delete(
        tenant_id="t1",
        task_id="task_1",
        namespace="ns",
        key="missing",
    )
    assert first is False
    assert second is False


def test_task_memory_delete_twice_after_write_returns_false_second_time() -> None:
    store = InMemoryTaskMemoryStore()
    TaskMemoryCoordinator.write(
        store,
        tenant_id="t1",
        task_id="task_1",
        namespace="ns",
        key="k",
        value={"v": 1},
    )
    assert store.delete(
        tenant_id="t1",
        task_id="task_1",
        namespace="ns",
        key="k",
    )
    assert store.delete(
        tenant_id="t1",
        task_id="task_1",
        namespace="ns",
        key="k",
    ) is False


def test_task_memory_put_failure_does_not_mutate_store() -> None:
    store = FailingTaskMemoryStore()
    store.fail_operation = "put"
    store.fail_tenant = "tenant-fail"
    good = TaskMemoryCoordinator.write(
        store,
        tenant_id="tenant-ok",
        task_id="task_1",
        namespace="ns",
        key="good",
        value={"v": 1},
    )
    with pytest.raises(RuntimeError, match="put failed"):
        TaskMemoryCoordinator.write(
            store,
            tenant_id="tenant-fail",
            task_id="task_1",
            namespace="ns",
            key="bad",
            value={"v": 2},
        )
    still = TaskMemoryCoordinator.read(
        store,
        tenant_id="tenant-ok",
        task_id="task_1",
        namespace="ns",
        key="good",
    )
    assert still is not None
    assert still.record_id == good.record_id


def test_task_memory_get_failure_surfaces_to_caller() -> None:
    store = FailingTaskMemoryStore()
    store.fail_operation = "get"
    store.put(_record(tenant_id="t1", task_id="task_1"))
    with pytest.raises(RuntimeError, match="get failed"):
        TaskMemoryCoordinator.read(
            store,
            tenant_id="t1",
            task_id="task_1",
            namespace="ns",
            key="k",
        )


def test_task_memory_delete_failure_surfaces_to_caller() -> None:
    store = FailingTaskMemoryStore()
    store.fail_operation = "delete"
    store.put(_record(tenant_id="t1", task_id="task_1"))
    with pytest.raises(RuntimeError, match="delete failed"):
        store.delete(
            tenant_id="t1",
            task_id="task_1",
            namespace="ns",
            key="k",
        )
    assert (
        store.get(
            tenant_id="t1",
            task_id="task_1",
            namespace="ns",
            key="k",
        )
        is not None
    )


def test_task_memory_persistence_contract_is_implemented_by_failing_store() -> None:
    assert isinstance(FailingTaskMemoryStore(), TaskMemoryPersistence)
