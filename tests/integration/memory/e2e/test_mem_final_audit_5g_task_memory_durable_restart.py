# © Artur Czarnecki. All rights reserved.

"""MEM-FINAL-AUDIT-5G — Task Memory SQLite durable restart qualification."""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

from intergrax.runtime.persistence.sqlite_opens import open_task_memory_store_at
from intergrax.runtime.task_memory.coordinator import TaskMemoryCoordinator
from intergrax.runtime.task_memory.store import open_task_memory_store
from tests.integration.memory.e2e.mem_final_audit_5g_sqlite_restart_support import (
    parse_worker_json,
    run_restart_worker,
    write_fixture,
)

pytestmark = [pytest.mark.integration, pytest.mark.gate]

_UNICODE_MARKER = "Instrukcja: żółć 🚀"
_VALUE_V1 = {"subject": "Acme", "version": 1, "note": _UNICODE_MARKER}
_VALUE_V2 = {"subject": "Acme", "version": 2, "note": _UNICODE_MARKER}


def _write(
    db_path: Path,
    *,
    tenant_id: str,
    task_id: str,
    namespace: str = "qual",
    key: str = "draft",
    value: dict[str, object],
    provenance: dict[str, object] | None = None,
) -> None:
    store = open_task_memory_store_at(db_path)
    try:
        TaskMemoryCoordinator.write(
            store,
            tenant_id=tenant_id,
            task_id=task_id,
            namespace=namespace,
            key=key,
            value=value,
            provenance=provenance or {"agent_id": "5g-harness"},
        )
    finally:
        store.close()


def _read(
    db_path: Path,
    *,
    tenant_id: str,
    task_id: str,
    namespace: str = "qual",
    key: str = "draft",
):
    store = open_task_memory_store_at(db_path)
    try:
        return TaskMemoryCoordinator.read(
            store,
            tenant_id=tenant_id,
            task_id=task_id,
            namespace=namespace,
            key=key,
        )
    finally:
        store.close()


@pytest.mark.unit
def test_task_write_reopen_read(tmp_path: Path) -> None:
    db_path = tmp_path / "task.db"
    _write(db_path, tenant_id="tenant-a", task_id="task-1", value=_VALUE_V1)
    loaded = _read(db_path, tenant_id="tenant-a", task_id="task-1")
    assert loaded is not None
    assert loaded.value["note"] == _UNICODE_MARKER


@pytest.mark.unit
def test_task_update_survives_double_reopen(tmp_path: Path) -> None:
    db_path = tmp_path / "update.db"
    _write(db_path, tenant_id="tenant-a", task_id="task-1", value=_VALUE_V1)
    first = _read(db_path, tenant_id="tenant-a", task_id="task-1")
    assert first is not None
    record_id = first.record_id

    _write(db_path, tenant_id="tenant-a", task_id="task-1", value=_VALUE_V2)
    second = _read(db_path, tenant_id="tenant-a", task_id="task-1")
    assert second is not None
    assert second.record_id == record_id
    assert second.value["version"] == 2
    assert second.value["version"] != 1


@pytest.mark.unit
def test_task_delete_survives_reopen(tmp_path: Path) -> None:
    db_path = tmp_path / "delete.db"
    _write(db_path, tenant_id="tenant-a", task_id="task-1", value=_VALUE_V1)
    store = open_task_memory_store_at(db_path)
    try:
        assert store.delete(
            tenant_id="tenant-a",
            task_id="task-1",
            namespace="qual",
            key="draft",
        )
    finally:
        store.close()
    assert _read(db_path, tenant_id="tenant-a", task_id="task-1") is None


@pytest.mark.unit
def test_task_triple_reopen_cycle(tmp_path: Path) -> None:
    db_path = tmp_path / "triple.db"
    for version in (1, 2, 3):
        _write(
            db_path,
            tenant_id="tenant-a",
            task_id="task-1",
            value={"version": version},
        )
    loaded = _read(db_path, tenant_id="tenant-a", task_id="task-1")
    assert loaded is not None
    assert loaded.value["version"] == 3


@pytest.mark.unit
def test_task_tenant_isolation_after_restart(tmp_path: Path) -> None:
    db_path = tmp_path / "tenant-iso.db"
    _write(db_path, tenant_id="tenant-a", task_id="shared", value={"who": "A"})
    _write(db_path, tenant_id="tenant-b", task_id="shared", value={"who": "B"})
    open_task_memory_store_at(db_path).close()
    reopened = open_task_memory_store_at(db_path)
    try:
        a = TaskMemoryCoordinator.read(
            reopened,
            tenant_id="tenant-a",
            task_id="shared",
            namespace="qual",
            key="draft",
        )
        b = TaskMemoryCoordinator.read(
            reopened,
            tenant_id="tenant-b",
            task_id="shared",
            namespace="qual",
            key="draft",
        )
    finally:
        reopened.close()
    assert a is not None and a.value["who"] == "A"
    assert b is not None and b.value["who"] == "B"


@pytest.mark.unit
def test_task_id_isolation_after_restart(tmp_path: Path) -> None:
    db_path = tmp_path / "task-iso.db"
    _write(db_path, tenant_id="tenant-a", task_id="task-x", value={"slot": "x"})
    _write(db_path, tenant_id="tenant-a", task_id="task-y", value={"slot": "y"})
    x = _read(db_path, tenant_id="tenant-a", task_id="task-x")
    y = _read(db_path, tenant_id="tenant-a", task_id="task-y")
    assert x is not None and x.value["slot"] == "x"
    assert y is not None and y.value["slot"] == "y"


@pytest.mark.unit
def test_task_multiple_records_per_scope(tmp_path: Path) -> None:
    db_path = tmp_path / "multi.db"
    for key in ("a", "b"):
        _write(
            db_path,
            tenant_id="tenant-a",
            task_id="task-1",
            key=key,
            value={"key": key},
        )
    store = open_task_memory_store_at(db_path)
    try:
        rows = TaskMemoryCoordinator.list_namespace(
            store,
            tenant_id="tenant-a",
            task_id="task-1",
            namespace="qual",
        )
    finally:
        store.close()
    assert {row.key for row in rows} == {"a", "b"}


@pytest.mark.unit
def test_task_schema_idempotent_on_reopen(tmp_path: Path) -> None:
    db_path = tmp_path / "schema.db"
    _write(db_path, tenant_id="tenant-a", task_id="task-1", value=_VALUE_V1)
    open_task_memory_store_at(db_path).close()
    open_task_memory_store_at(db_path).close()
    loaded = _read(db_path, tenant_id="tenant-a", task_id="task-1")
    assert loaded is not None


@pytest.mark.unit
def test_task_missing_db_creates_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "task.db"
    store = open_task_memory_store(db_path)
    assert db_path.is_file()
    store.close()
    assert _read(db_path, tenant_id="t", task_id="task") is None


@pytest.mark.unit
def test_task_wrong_db_path_does_not_read_peer_data(tmp_path: Path) -> None:
    db_a = tmp_path / "a.db"
    db_b = tmp_path / "b.db"
    _write(db_a, tenant_id="tenant-a", task_id="task-1", value={"secret": "a"})
    assert _read(db_b, tenant_id="tenant-a", task_id="task-1") is None


@pytest.mark.unit
def test_task_failed_update_preserves_committed_state(tmp_path: Path) -> None:
    if sys.platform == "win32":
        pytest.skip(
            "read-only chmod semantics are not reliable on Windows for this proof"
        )
    db_path = tmp_path / "failed.db"
    _write(db_path, tenant_id="tenant-a", task_id="task-1", value=_VALUE_V1)
    os.chmod(db_path, stat.S_IREAD)
    store = open_task_memory_store_at(db_path)
    try:
        with pytest.raises(Exception):
            TaskMemoryCoordinator.write(
                store,
                tenant_id="tenant-a",
                task_id="task-1",
                namespace="qual",
                key="draft",
                value=_VALUE_V2,
            )
    finally:
        store.close()
    os.chmod(db_path, stat.S_IWRITE | stat.S_IREAD)
    loaded = _read(db_path, tenant_id="tenant-a", task_id="task-1")
    assert loaded is not None
    assert loaded.value["version"] == 1


@pytest.mark.unit
def test_task_fresh_process_write_read(tmp_path: Path) -> None:
    db_path = tmp_path / "process.db"
    fixture = write_fixture(
        tmp_path,
        "task-fixture.json",
        {
            "tenant_id": "tenant-proc",
            "task_id": "task-proc",
            "namespace": "qual",
            "key": "draft",
            "value": _VALUE_V1,
            "provenance": {"phase": "A"},
        },
    )
    write_proc = run_restart_worker("task_write", db_path=db_path, fixture_path=fixture)
    assert write_proc.returncode == 0, write_proc.stderr

    read_proc = run_restart_worker("task_read", db_path=db_path, fixture_path=fixture)
    assert read_proc.returncode == 0, read_proc.stderr
    payload = parse_worker_json(read_proc.stdout)
    assert payload["present"] is True
    assert payload["value"]["note"] == _UNICODE_MARKER
