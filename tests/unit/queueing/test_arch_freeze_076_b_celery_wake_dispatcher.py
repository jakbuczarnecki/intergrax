# © Artur Czarnecki. All rights reserved.

"""ARCH-FREEZE package B — Celery wake consumes durable document-store rows."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from celery import Celery
from intergrax.queueing.contracts.task_queue import TaskRequest, TaskStatus
from intergrax.queueing.providers.document_store.document_store_task_queue import (
    DocumentStoreTaskQueue,
)
from intergrax.queueing.worker.dispatcher import register_dispatcher_task
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.runtime.background_execution.admission_wiring import (
    wire_background_execution_admission_dependencies,
)
from intergrax.runtime.background_execution.identity_persistence import (
    wire_background_execution_identity_persistence,
)
from intergrax.runtime.tools.in_memory_idempotency_store import InMemoryIdempotencyStore
from intergrax.tools.execution_models import ToolExecutionResult
from pydantic import BaseModel
from testing_support.cross_process_spine.durable_document_store import SqliteFileDocumentStore
from tests.unit.queueing.worker.dispatcher_test_kv import (
    DispatcherTestKVStore,
    dispatcher_test_causal_persistence,
)

pytestmark = pytest.mark.unit


class _DummyOutput(BaseModel):
    value: str = "ok"


def _register_wake_dispatcher(
    registry: TaskExecutionRegistry,
    queue: DocumentStoreTaskQueue,
) -> Celery:
    app = Celery("arch-freeze-b-wake-dispatcher")
    app.conf.task_always_eager = True
    app.conf.task_eager_propagates = True
    kv_store = DispatcherTestKVStore()
    admission = wire_background_execution_admission_dependencies(kv_store=kv_store)
    register_dispatcher_task(
        app=app,
        registry=registry,
        idempotency_store=queue._idempotency_store,
        identity_persistence=wire_background_execution_identity_persistence(
            kv_store=kv_store,
        ),
        causal_evidence_persistence=dispatcher_test_causal_persistence(),
        attempt_lifecycle=admission.attempt_lifecycle,
        execution_terminal=admission.execution_terminal,
        document_store_task_queue=queue,
    )
    return app


def test_wake_kwargs_document_task_id_in_registered_signature(tmp_path: Path) -> None:
    store = SqliteFileDocumentStore(
        tmp_path / "tasks.db",
        cursor_secret=b"arch-freeze-b-wake-sign-secret-32b",
    )
    queue = DocumentStoreTaskQueue(store, idempotency_store=InMemoryIdempotencyStore())
    app = _register_wake_dispatcher(TaskExecutionRegistry(), queue)
    task = app.tasks["intergrax.execute"]
    assert "document_task_id" in inspect.signature(task.run).parameters


def test_terminal_row_wake_acknowledged_without_handler(tmp_path: Path) -> None:
    store = SqliteFileDocumentStore(
        tmp_path / "tasks.db",
        cursor_secret=b"arch-freeze-b-wake-term-secret-32b",
    )
    queue = DocumentStoreTaskQueue(store, idempotency_store=InMemoryIdempotencyStore())
    handle = queue.enqueue(
        TaskRequest(
            tenant_id="tenant-a",
            run_id="run-1",
            task_name="demo.echo",
            payload=b"{}",
        )
    )
    queue.mark_succeeded(handle)
    calls = {"n": 0}

    class _Registry(TaskExecutionRegistry):
        def get_handler(self, logical_task_name: str):
            def handler(**_kwargs) -> ToolExecutionResult[_DummyOutput]:
                calls["n"] += 1
                return ToolExecutionResult.ok(_DummyOutput())

            return handler

    app = _register_wake_dispatcher(_Registry(), queue)
    app.tasks["intergrax.execute"].apply(
        kwargs={
            "logical_task_name": "demo.echo",
            "tenant_id": "tenant-a",
            "run_id": "run-1",
            "payload": b"{}",
            "document_task_id": handle.task_id,
        }
    )
    assert calls["n"] == 0
    assert queue.get_status(handle) is TaskStatus.SUCCEEDED


def test_pending_row_claimed_handler_runs_once_row_terminal_before_return(tmp_path: Path) -> None:
    store = SqliteFileDocumentStore(
        tmp_path / "tasks.db",
        cursor_secret=b"arch-freeze-b-wake-pend-secret-32b",
    )
    queue = DocumentStoreTaskQueue(store, idempotency_store=InMemoryIdempotencyStore())
    handle = queue.enqueue(
        TaskRequest(
            tenant_id="tenant-a",
            run_id="run-1",
            task_name="demo.echo",
            payload=b"{}",
            idempotency_key="wake-idem-1",
        )
    )
    calls = {"n": 0}

    class _Registry(TaskExecutionRegistry):
        def get_handler(self, logical_task_name: str):
            def handler(**_kwargs) -> ToolExecutionResult[_DummyOutput]:
                calls["n"] += 1
                assert queue.get_status(handle) is TaskStatus.RUNNING
                return ToolExecutionResult.ok(_DummyOutput())

            return handler

    app = _register_wake_dispatcher(_Registry(), queue)
    app.tasks["intergrax.execute"].apply(
        kwargs={
            "logical_task_name": "demo.echo",
            "tenant_id": "tenant-a",
            "run_id": "run-1",
            "payload": b"{}",
            "idempotency_key": "wake-idem-1",
            "document_task_id": handle.task_id,
        }
    )
    assert calls["n"] == 1
    assert queue.get_status(handle) is TaskStatus.SUCCEEDED
