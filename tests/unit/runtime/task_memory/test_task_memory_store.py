# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.runtime.task_memory import (
    InMemoryTaskMemoryStore,
    NullTaskMemoryPersistence,
    SQLiteTaskMemoryStore,
    TaskMemoryCoordinator,
    TaskMemoryLimits,
    open_task_memory_store,
    resolve_task_memory_persistence,
)


@pytest.mark.unit
@pytest.mark.gate
def test_resolve_task_memory_persistence_explicit_implementation():
    custom = InMemoryTaskMemoryStore()
    store = resolve_task_memory_persistence(implementation=custom)
    assert store is custom


@pytest.mark.unit
@pytest.mark.gate
def test_resolve_task_memory_persistence_disabled_by_default():
    assert resolve_task_memory_persistence() is None


@pytest.mark.unit
@pytest.mark.gate
def test_open_task_memory_store_sqlite(tmp_path):
    store = open_task_memory_store(tmp_path / "task_memory.db")
    assert isinstance(store, SQLiteTaskMemoryStore)


@pytest.mark.unit
@pytest.mark.gate
def test_resolve_task_memory_persistence_with_explicit_path(tmp_path):
    store = resolve_task_memory_persistence(db_path=tmp_path / "task_memory.db")
    assert isinstance(store, SQLiteTaskMemoryStore)


@pytest.mark.unit
@pytest.mark.gate
def test_null_task_memory_persistence_is_noop():
    store = NullTaskMemoryPersistence()
    assert store.get(tenant_id="t1", task_id="task_1", namespace="ns", key="k") is None


@pytest.mark.unit
@pytest.mark.gate
def test_task_memory_coordinator_write_read_roundtrip():
    store = InMemoryTaskMemoryStore()
    written = TaskMemoryCoordinator.write(
        store,
        tenant_id="t1",
        task_id="task_1",
        namespace="vendor_report",
        key="draft",
        value={"subject": "Acme Corp Q1", "status": "draft"},
        provenance={"agent_id": "organization_worker"},
    )
    loaded = TaskMemoryCoordinator.read(
        store,
        tenant_id="t1",
        task_id="task_1",
        namespace="vendor_report",
        key="draft",
    )
    assert loaded is not None
    assert loaded.record_id == written.record_id
    assert loaded.value["subject"] == "Acme Corp Q1"
    assert loaded.provenance["agent_id"] == "organization_worker"


@pytest.mark.unit
@pytest.mark.gate
def test_task_memory_coordinator_upsert_preserves_record_id():
    store = InMemoryTaskMemoryStore()
    first = TaskMemoryCoordinator.write(
        store,
        tenant_id="t1",
        task_id="task_1",
        namespace="vendor_report",
        key="draft",
        value={"version": 1},
    )
    second = TaskMemoryCoordinator.write(
        store,
        tenant_id="t1",
        task_id="task_1",
        namespace="vendor_report",
        key="draft",
        value={"version": 2},
    )
    assert second.record_id == first.record_id
    assert second.created_at_utc == first.created_at_utc
    assert second.updated_at_utc >= first.updated_at_utc
    assert second.value["version"] == 2


@pytest.mark.unit
@pytest.mark.gate
def test_task_memory_coordinator_enforces_record_limit():
    store = InMemoryTaskMemoryStore()
    limits = TaskMemoryLimits(max_records_per_task=2)
    TaskMemoryCoordinator.write(
        store,
        tenant_id="t1",
        task_id="task_1",
        namespace="ns",
        key="a",
        value={"v": 1},
        limits=limits,
    )
    TaskMemoryCoordinator.write(
        store,
        tenant_id="t1",
        task_id="task_1",
        namespace="ns",
        key="b",
        value={"v": 2},
        limits=limits,
    )
    with pytest.raises(ValueError, match="record limit"):
        TaskMemoryCoordinator.write(
            store,
            tenant_id="t1",
            task_id="task_1",
            namespace="ns",
            key="c",
            value={"v": 3},
            limits=limits,
        )


@pytest.mark.unit
@pytest.mark.gate
def test_task_memory_coordinator_enforces_value_size():
    store = InMemoryTaskMemoryStore()
    limits = TaskMemoryLimits(max_value_bytes=32)
    with pytest.raises(ValueError, match="value exceeds"):
        TaskMemoryCoordinator.write(
            store,
            tenant_id="t1",
            task_id="task_1",
            namespace="ns",
            key="big",
            value={"payload": "x" * 100},
            limits=limits,
        )


@pytest.mark.unit
@pytest.mark.gate
def test_task_memory_list_namespace_prefix():
    store = InMemoryTaskMemoryStore()
    TaskMemoryCoordinator.write(
        store,
        tenant_id="t1",
        task_id="task_1",
        namespace="findings",
        key="vendor.a",
        value={"score": 1},
    )
    TaskMemoryCoordinator.write(
        store,
        tenant_id="t1",
        task_id="task_1",
        namespace="findings",
        key="vendor.b",
        value={"score": 2},
    )
    TaskMemoryCoordinator.write(
        store,
        tenant_id="t1",
        task_id="task_1",
        namespace="findings",
        key="summary",
        value={"text": "ok"},
    )
    rows = TaskMemoryCoordinator.list_namespace(
        store,
        tenant_id="t1",
        task_id="task_1",
        namespace="findings",
        prefix="vendor.",
    )
    assert [row.key for row in rows] == ["vendor.a", "vendor.b"]


@pytest.mark.unit
@pytest.mark.gate
def test_sqlite_task_memory_store_roundtrip(tmp_path):
    store = open_task_memory_store(tmp_path / "task_memory.db")
    TaskMemoryCoordinator.write(
        store,
        tenant_id="t1",
        task_id="task_1",
        namespace="vendor_report",
        key="draft",
        value={"subject": "Contoso Q2"},
    )
    loaded = TaskMemoryCoordinator.read(
        store,
        tenant_id="t1",
        task_id="task_1",
        namespace="vendor_report",
        key="draft",
    )
    assert loaded is not None
    assert loaded.value["subject"] == "Contoso Q2"
    assert store.count_for_task(tenant_id="t1", task_id="task_1") == 1


@pytest.mark.unit
@pytest.mark.gate
def test_task_memory_clear_task():
    store = InMemoryTaskMemoryStore()
    TaskMemoryCoordinator.write(
        store,
        tenant_id="t1",
        task_id="task_1",
        namespace="ns",
        key="a",
        value={"v": 1},
    )
    deleted = store.clear_task(tenant_id="t1", task_id="task_1")
    assert deleted == 1
    assert store.count_for_task(tenant_id="t1", task_id="task_1") == 0
