# © Artur Czarnecki. All rights reserved.

"""ARCH-FREEZE package B — durable rows visible across OS processes."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from intergrax.applications._shared.production_queue_resolver import (
    ProductionQueueBackend,
    UnsupportedQueueBackendError,
    UnsupportedQueueBackendReason,
    require_celery_broker_url_for_strict,
)
from intergrax.applications._shared.queue_worker_wiring import wire_optional_queue_execution
from intergrax.contracts.lease_claim import StaleClaimError
from intergrax.integrations.providers.message_bus.celery.config import CeleryIntegrationConfig
from intergrax.queueing.contracts.task_queue import TaskRequest, TaskStatus
from intergrax.queueing.providers.document_store.document_store_task_queue import (
    DocumentStoreTaskQueue,
    TaskDeliveryMismatchError,
    TaskRowClaimContext,
)
from intergrax.queueing.providers.document_store.durable_wake_task_queue import DurableWakeTaskQueue
from intergrax.queueing.providers.document_store.task_row_lease import read_claim_from_row
from intergrax.runtime.tools.in_memory_idempotency_store import InMemoryIdempotencyStore
from intergrax.runtime.tools.sqlite_idempotency_store import SQLiteIdempotencyStore
from testing_support.cross_process_spine.durable_document_store import SqliteFileDocumentStore

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[3]
_WORKER = Path(__file__).with_name("_arch_freeze_b_queue_worker.py")


def _store(tmp_path: Path) -> SqliteFileDocumentStore:
    return SqliteFileDocumentStore(
        tmp_path / "tasks.db",
        cursor_secret=b"arch-freeze-b-queue-tests-cursor-secret-32b",
    )


def test_celery_from_env_without_broker_url_is_empty() -> None:
    env = dict(os.environ)
    env.pop("INTERGRAX_CELERY_BROKER_URL", None)
    old = os.environ
    try:
        os.environ = env  # type: ignore[assignment]
        assert CeleryIntegrationConfig.from_env().broker_url == ""
    finally:
        os.environ = old


def test_strict_posture_requires_explicit_celery_broker() -> None:
    with pytest.raises(UnsupportedQueueBackendError) as refusal:
        require_celery_broker_url_for_strict(broker_url="")
    assert refusal.value.reason is UnsupportedQueueBackendReason.CELERY_BROKER_URL_REQUIRED


def test_durable_enqueue_survives_first_process(tmp_path: Path) -> None:
    db_path = tmp_path / "tasks.db"
    store_path = str(db_path)
    script = f"""
import sys
from pathlib import Path
sys.path.insert(0, {str(_REPO)!r})
from intergrax.queueing.contracts.task_queue import TaskRequest, TaskExecutionIdentity
from intergrax.queueing.providers.document_store.document_store_task_queue import DocumentStoreTaskQueue
from testing_support.cross_process_spine.durable_document_store import SqliteFileDocumentStore
store = SqliteFileDocumentStore(Path({store_path!r}), cursor_secret=b"x" * 32)
queue = DocumentStoreTaskQueue(store)
identity = TaskExecutionIdentity(
    task_id="task_" + "ab" * 16,
    run_id="run_" + "cd" * 16,
    attempt_id="attempt_" + "ef" * 16,
    execution_id="exec_" + "01" * 16,
)
queue.enqueue(TaskRequest(
    tenant_id="tenant-a",
    run_id="transport-r1",
    task_name="demo.echo",
    payload=b"{{}}",
    identity=identity,
))
"""
    proc = subprocess.run([sys.executable, "-c", script], cwd=str(_REPO), check=True)
    assert proc.returncode == 0
    store = _store(tmp_path)
    queue = DocumentStoreTaskQueue(store)
    rows = queue.list_tasks("tenant-a", status_filter=TaskStatus.PENDING)
    assert len(rows) == 1
    row = store.get(
        "intergrax.task_queue.v1:tenant-a",
        rows[0].task_id,
    )
    assert row is not None
    assert row.data.get("identity", {}).get("task_id") == "task_" + "ab" * 16


def test_concurrent_claim_runs_handler_once(tmp_path: Path) -> None:
    db_path = tmp_path / "tasks.db"
    seed = subprocess.run(
        [sys.executable, str(_WORKER), "--mode", "seed"],
        cwd=str(_REPO),
        env={**os.environ, "ARCH_FREEZE_B_DB": str(db_path)},
        check=True,
    )
    assert seed.returncode == 0
    env = {**os.environ, "ARCH_FREEZE_B_DB": str(db_path)}
    worker_a = subprocess.Popen(
        [sys.executable, str(_WORKER), "--mode", "claim"],
        cwd=str(_REPO),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    worker_b = subprocess.Popen(
        [sys.executable, str(_WORKER), "--mode", "claim"],
        cwd=str(_REPO),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out_a, err_a = worker_a.communicate(timeout=30)
    out_b, err_b = worker_b.communicate(timeout=30)
    assert worker_a.returncode == 0, err_a
    assert worker_b.returncode == 0, err_b
    fired = int(out_a.strip() or "0") + int(out_b.strip() or "0")
    assert fired == 1


def test_stale_complete_after_lease_expiry(tmp_path: Path) -> None:
    store = _store(tmp_path)
    idem = InMemoryIdempotencyStore()
    queue = DocumentStoreTaskQueue(store, idempotency_store=idem)
    handle = queue.enqueue(
        TaskRequest(
            tenant_id="tenant-a",
            run_id="r1",
            task_name="demo.echo",
            payload=b"{}",
            idempotency_key="idem-1",
        )
    )
    claimed = queue.claim_pending(
        limit=1,
        claim_context=TaskRowClaimContext(owner_id="owner-a", lease_seconds=1),
    )
    assert claimed
    record = store.get(f"intergrax.task_queue.v1:{handle.tenant_id}", handle.task_id)
    assert record is not None
    stale_ownership, _ = read_claim_from_row(record.data)
    assert stale_ownership is not None
    time.sleep(1.2)
    queue.recover_interrupted_running(max_retries=3)
    reclaimed = queue.claim_pending(
        limit=1,
        claim_context=TaskRowClaimContext(owner_id="owner-b", lease_seconds=300),
    )
    assert reclaimed
    with pytest.raises(StaleClaimError):
        queue.mark_succeeded(handle, presented_claim=stale_ownership)


def test_rejected_publish_leaves_row_pending(tmp_path: Path) -> None:
    store = _store(tmp_path)
    durable = DocumentStoreTaskQueue(store)
    published: list[str] = []

    def _publish(handle, _request) -> None:  # noqa: ANN001
        published.append(handle.task_id)
        raise RuntimeError("broker down")

    queue = DurableWakeTaskQueue(
        durable_queue=durable,
        publish_wake=_publish,
    )
    handle = queue.enqueue(
        TaskRequest(tenant_id="t", run_id="r", task_name="demo", payload=b"{}"),
    )
    assert queue.get_status(handle) is TaskStatus.PENDING
    assert published == [handle.task_id]


def test_empty_celery_url_in_strict_wiring_fails(tmp_path: Path) -> None:
    store = _store(tmp_path)
    env = dict(os.environ)
    env.pop("INTERGRAX_CELERY_BROKER_URL", None)
    old = os.environ
    try:
        os.environ = env  # type: ignore[assignment]
        with pytest.raises(UnsupportedQueueBackendError) as refusal:
            wire_optional_queue_execution(
                enabled=True,
                host_execution=MagicMock(),
                registry=MagicMock(),
                run_service=MagicMock(),
                queue_backend=ProductionQueueBackend.CELERY,
                kv_store=MagicMock(),
                causal_evidence_persistence=MagicMock(),
                document_store=store,
                idempotency_store=SQLiteIdempotencyStore(str(tmp_path / "idem.sqlite")),
                production_mode=True,
            )
        assert refusal.value.reason is UnsupportedQueueBackendReason.CELERY_BROKER_URL_REQUIRED
    finally:
        os.environ = old


def test_tenant_mismatch_does_not_settle(tmp_path: Path) -> None:
    store = _store(tmp_path)
    queue = DocumentStoreTaskQueue(store)
    handle = queue.enqueue(
        TaskRequest(tenant_id="tenant-a", run_id="r", task_name="demo", payload=b"{}"),
    )
    queue.claim_pending(limit=1)
    with pytest.raises(TaskDeliveryMismatchError):
        queue.assert_delivery_matches(
            handle,
            tenant_id="tenant-b",
            identity=None,
        )
    assert queue.get_status(handle) is TaskStatus.RUNNING


def test_other_process_cannot_settle_fenced_row_without_claim(tmp_path: Path) -> None:
    store = _store(tmp_path)
    queue = DocumentStoreTaskQueue(
        store,
        idempotency_store=SQLiteIdempotencyStore(str(tmp_path / "idem.sqlite")),
    )
    handle = queue.enqueue(
        TaskRequest(
            tenant_id="tenant-a",
            run_id="r1",
            task_name="demo.echo",
            payload=b"{}",
            idempotency_key="fence-1",
        )
    )
    claimed = queue.claim_pending(
        limit=1,
        claim_context=TaskRowClaimContext(owner_id="owner-a", lease_seconds=30),
    )
    assert claimed
    proc = subprocess.run(
        [sys.executable, str(_WORKER), "--mode", "complete-without-claim"],
        cwd=str(_REPO),
        env={
            **os.environ,
            "ARCH_FREEZE_B_DB": str(tmp_path / "tasks.db"),
            "ARCH_FREEZE_B_TASK_ID": handle.task_id,
            "ARCH_FREEZE_B_TENANT": handle.tenant_id,
        },
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "STALE"
    assert queue.get_status(handle) is TaskStatus.RUNNING


def test_in_memory_ledger_refused_for_out_of_process_worker(tmp_path: Path) -> None:
    store = _store(tmp_path)
    env = dict(os.environ)
    env["INTERGRAX_CELERY_BROKER_URL"] = "redis://broker.example/0"
    old = os.environ
    try:
        os.environ = env  # type: ignore[assignment]
        with pytest.raises(UnsupportedQueueBackendError) as refusal:
            wire_optional_queue_execution(
                enabled=True,
                host_execution=MagicMock(),
                registry=MagicMock(),
                run_service=MagicMock(),
                queue_backend=ProductionQueueBackend.CELERY,
                kv_store=MagicMock(),
                causal_evidence_persistence=MagicMock(),
                document_store=store,
                idempotency_store=InMemoryIdempotencyStore(),
                production_mode=True,
            )
        assert refusal.value.reason is UnsupportedQueueBackendReason.TASK_STORE_NOT_DURABLE
    finally:
        os.environ = old
