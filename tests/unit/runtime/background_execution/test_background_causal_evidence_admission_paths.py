# © Artur Czarnecki. All rights reserved.

"""DIAG-1I — required causal evidence admission on supported background paths."""

from __future__ import annotations

import base64
import json
from typing import Optional
from unittest.mock import Mock

import pytest
from celery import Celery
from pydantic import BaseModel

from intergrax.background_tasks.definition import TaskDefinition
from intergrax.background_tasks.registry import TaskRegistry
from intergrax.background_tasks.state_store import TaskResultStore, TaskStateStore
from intergrax.background_tasks.worker_runtime import WorkerRuntime
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.queueing.contracts.task_queue import TaskRequest, TaskStatus
from intergrax.queueing.providers.broker_worker_base import BrokerWorkerBase
from intergrax.queueing.providers.document_store import DocumentStoreTaskQueue
from intergrax.queueing.providers.document_store.colocated_worker import DocumentStoreTaskWorker
from intergrax.queueing.worker.dispatcher import register_dispatcher_task
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.runtime.background_execution.bootstrap import (
    BackgroundExecutionIdentity,
    bootstrap_background_execution,
)
from intergrax.runtime.background_execution.identity_persistence import (
    KvBackgroundExecutionIdentityPersistence,
    wire_background_execution_identity_persistence,
)
from intergrax.runtime.background_execution.required_audit_evidence import (
    RequiredAuditEvidencePersistenceError,
    admit_background_execution_handler,
)
from intergrax.runtime.background_execution.transport_ref import (
    BackgroundTransportExecutionRef,
)
from intergrax.runtime.observability.causal_evidence import CausalRelationKind
from intergrax.runtime.observability.document_store_causal_evidence_persistence import (
    wire_causal_evidence_persistence,
)
from intergrax.tools.execution_models import ToolExecutionResult
from tests.unit.runtime.background_execution.causal_admission_doubles import (
    failing_causal_persistence,
    make_causal_persistence,
    make_kv_store,
)

pytestmark = pytest.mark.unit


class _Output(BaseModel):
    value: str = "ok"


class _BrokerWorker(BrokerWorkerBase):
    def start(self) -> None:
        raise NotImplementedError


def _handler(
    *,
    tenant_id: str,
    run_id: str,
    payload: bytes,
    idempotency_key: Optional[str],
    execution_identity: BackgroundExecutionIdentity,
) -> ToolExecutionResult[_Output]:
    _ = tenant_id, run_id, payload, idempotency_key, execution_identity
    return ToolExecutionResult.ok(_Output())


def _build_broker_message(
    *,
    task_id: str = "transport-1",
    provider: str = "broker",
) -> bytes:
    encoded_payload = base64.b64encode(b"{}").decode("ascii")
    return json.dumps(
        {
            "task_id": task_id,
            "tenant_id": "tenant-a",
            "run_id": "queue-correlation",
            "task_name": "demo.task.v1",
            "payload": encoded_payload,
            "provider": provider,
        }
    ).encode("utf-8")


def _assert_causal_matrix(
    *,
    persistence,
    tenant_id: str,
    provider: str,
    transport_task_id: str,
    execution_identity: BackgroundExecutionIdentity,
) -> None:
    by_transport = persistence.list_for_transport_task(
        tenant_id=tenant_id,
        provider=provider,
        transport_task_id=transport_task_id,
    )
    assert len(by_transport) == 1
    evidence = by_transport[0]
    assert evidence.relation_kind == CausalRelationKind.TRANSPORT_TASK_TRIGGERED_EXECUTION
    assert evidence.source.provider == provider
    assert evidence.source.task_id == transport_task_id
    assert evidence.source.tenant_id == tenant_id
    assert evidence.target.task_id == execution_identity.task_id
    assert evidence.target.run_id == execution_identity.run_id
    assert evidence.target.attempt_id == execution_identity.attempt_id
    assert evidence.evidence_id


def test_broker_worker_path_persists_required_causal_evidence() -> None:
    kv = make_kv_store()
    persistence = make_causal_persistence()
    registry = TaskExecutionRegistry()
    registry.register("demo.task.v1", _handler)
    worker = _BrokerWorker(
        registry=registry,
        kv_store=kv,
        provider_name="rabbitmq",
        identity_persistence=wire_background_execution_identity_persistence(kv_store=kv),
        causal_evidence_persistence=persistence,
    )
    worker.process_message(raw_payload=_build_broker_message(provider="rabbitmq"))
    records = persistence.list_for_transport_task(
        tenant_id="tenant-a",
        provider="rabbitmq",
        transport_task_id="transport-1",
    )
    assert len(records) == 1
    identity = BackgroundExecutionIdentity(
        tenant_id="tenant-a",
        task_id=records[0].target.task_id,
        run_id=records[0].target.run_id,
        attempt_id=records[0].target.attempt_id,
    )
    _assert_causal_matrix(
        persistence=persistence,
        tenant_id="tenant-a",
        provider="rabbitmq",
        transport_task_id="transport-1",
        execution_identity=identity,
    )


def test_worker_runtime_path_persists_required_causal_evidence() -> None:
    kv = make_kv_store()
    persistence = make_causal_persistence()
    task_registry = TaskRegistry()
    execution_registry = TaskExecutionRegistry()
    task_registry.register(
        TaskDefinition(
            task_name="demo.task.v1",
            payload_schema=dict,
            handler=_handler,
        )
    )
    task_registry.bind_execution_registry(execution_registry)
    runtime = WorkerRuntime(
        registry=task_registry,
        state_store=TaskStateStore(kv_store=kv, provider="kafka"),
        result_store=TaskResultStore(kv_store=kv),
        execution_registry=execution_registry,
        provider="kafka",
        identity_persistence=wire_background_execution_identity_persistence(kv_store=kv),
        causal_evidence_persistence=persistence,
    )
    request = TaskRequest(
        tenant_id="tenant-a",
        run_id="queue-correlation",
        task_name="demo.task.v1",
        payload=b"{}",
    )
    result = runtime.process_request(request, task_id="transport-kafka-1")
    assert result.status == TaskStatus.SUCCEEDED
    records = persistence.list_for_transport_task(
        tenant_id="tenant-a",
        provider="kafka",
        transport_task_id="transport-kafka-1",
    )
    assert len(records) == 1
    identity = BackgroundExecutionIdentity(
        tenant_id="tenant-a",
        task_id=records[0].target.task_id,
        run_id=records[0].target.run_id,
        attempt_id=records[0].target.attempt_id,
    )
    _assert_causal_matrix(
        persistence=persistence,
        tenant_id="tenant-a",
        provider="kafka",
        transport_task_id="transport-kafka-1",
        execution_identity=identity,
    )


def test_celery_dispatcher_path_persists_required_causal_evidence() -> None:
    kv = make_kv_store()
    persistence = make_causal_persistence()
    app = Celery("test")
    app.conf.task_always_eager = True
    app.conf.task_eager_propagates = True
    registry = TaskExecutionRegistry()
    registry.register("demo.task.v1", _handler)
    register_dispatcher_task(
        app=app,
        registry=registry,
        identity_persistence=wire_background_execution_identity_persistence(kv_store=kv),
        causal_evidence_persistence=persistence,
    )
    app.tasks["intergrax.execute"].apply(
        kwargs={
            "logical_task_name": "demo.task.v1",
            "tenant_id": "tenant-a",
            "run_id": "queue-correlation",
            "payload": b"{}",
            "idempotency_key": None,
        }
    )
    records = tuple(persistence._accepted_by_evidence_id.values())
    assert len(records) == 1
    assert records[0].source.provider == "celery"
    assert records[0].source.tenant_id == "tenant-a"
    assert records[0].source.task_id
    assert records[0].target.task_id
    assert records[0].target.run_id
    assert records[0].target.attempt_id
    assert records[0].evidence_id


def test_document_store_worker_path_persists_required_causal_evidence() -> None:
    store = InMemoryDocumentStore()
    queue = DocumentStoreTaskQueue(store)
    persistence = wire_causal_evidence_persistence(document_store=store)
    registry = TaskExecutionRegistry()
    registry.register("demo.task.v1", _handler)
    handle = queue.enqueue(
        TaskRequest(
            tenant_id="tenant-a",
            run_id="queue-correlation",
            task_name="demo.task.v1",
            payload=b"{}",
        )
    )
    worker = DocumentStoreTaskWorker(
        queue,
        registry,
        identity_persistence=wire_background_execution_identity_persistence(
            document_store=store,
        ),
        causal_evidence_persistence=persistence,
    )
    assert worker.drain_once() == 1
    records = persistence.list_for_transport_task(
        tenant_id="tenant-a",
        provider=handle.provider,
        transport_task_id=handle.task_id,
    )
    assert len(records) == 1
    identity = BackgroundExecutionIdentity(
        tenant_id="tenant-a",
        task_id=records[0].target.task_id,
        run_id=records[0].target.run_id,
        attempt_id=records[0].target.attempt_id,
    )
    _assert_causal_matrix(
        persistence=persistence,
        tenant_id="tenant-a",
        provider=handle.provider,
        transport_task_id=handle.task_id,
        execution_identity=identity,
    )


def test_retry_redelivery_reuses_attempt_and_creates_new_evidence() -> None:
    kv = make_kv_store()
    persistence = make_causal_persistence()
    identity_persistence = KvBackgroundExecutionIdentityPersistence(kv)
    transport_ref = BackgroundTransportExecutionRef(
        tenant_id="tenant-a",
        provider="broker",
        transport_task_id="transport-retry-1",
    )
    handler_calls = 0

    def _counting_handler() -> str:
        nonlocal handler_calls
        handler_calls += 1
        return "ok"

    first_identity = bootstrap_background_execution(
        transport_ref=transport_ref,
        identity_persistence=identity_persistence,
    )
    admit_background_execution_handler(
        transport_ref=transport_ref,
        execution_identity=first_identity,
        causal_evidence_persistence=persistence,
        handler=_counting_handler,
    )
    second_identity = bootstrap_background_execution(
        transport_ref=transport_ref,
        identity_persistence=identity_persistence,
    )
    admit_background_execution_handler(
        transport_ref=transport_ref,
        execution_identity=second_identity,
        causal_evidence_persistence=persistence,
        handler=_counting_handler,
    )

    assert first_identity.task_id == second_identity.task_id
    assert first_identity.run_id == second_identity.run_id
    assert first_identity.attempt_id == second_identity.attempt_id
    assert handler_calls == 2

    records = persistence.list_for_transport_task(
        tenant_id="tenant-a",
        provider="broker",
        transport_task_id="transport-retry-1",
    )
    assert len(records) == 2
    stored_attempts = {record.target.attempt_id for record in records}
    assert stored_attempts == {first_identity.attempt_id}
    assert records[0].evidence_id != records[1].evidence_id


@pytest.mark.parametrize(
    "path_name,factory",
    [
        ("broker", "_broker_failure_case"),
        ("worker_runtime", "_worker_runtime_failure_case"),
        ("celery", "_celery_failure_case"),
        ("document_store", "_document_store_failure_case"),
    ],
)
def test_persistence_failure_blocks_handler(path_name: str, factory: str) -> None:
    _ = path_name
    globals()[factory]()


def _broker_failure_case() -> None:
    kv = make_kv_store()
    registry = TaskExecutionRegistry()
    handler = Mock()
    registry.register("demo.task.v1", handler)
    worker = _BrokerWorker(
        registry=registry,
        kv_store=kv,
        identity_persistence=wire_background_execution_identity_persistence(kv_store=kv),
        causal_evidence_persistence=failing_causal_persistence(),
    )
    with pytest.raises(RequiredAuditEvidencePersistenceError):
        worker.process_message(raw_payload=_build_broker_message())
    handler.assert_not_called()
    assert kv.get("tenant-a", "task:transport-1:status") == TaskStatus.FAILED.value.encode()


def _worker_runtime_failure_case() -> None:
    kv = make_kv_store()
    task_registry = TaskRegistry()
    execution_registry = TaskExecutionRegistry()
    handler = Mock()
    task_registry.register(
        TaskDefinition(
            task_name="demo.task.v1",
            payload_schema=dict,
            handler=handler,
        )
    )
    task_registry.bind_execution_registry(execution_registry)
    runtime = WorkerRuntime(
        registry=task_registry,
        state_store=TaskStateStore(kv_store=kv, provider="kafka"),
        result_store=TaskResultStore(kv_store=kv),
        execution_registry=execution_registry,
        provider="kafka",
        identity_persistence=wire_background_execution_identity_persistence(kv_store=kv),
        causal_evidence_persistence=failing_causal_persistence(),
    )
    with pytest.raises(RequiredAuditEvidencePersistenceError):
        runtime.process_request(
            TaskRequest(
                tenant_id="tenant-a",
                run_id="queue-correlation",
                task_name="demo.task.v1",
                payload=b"{}",
            ),
            task_id="transport-1",
        )
    handler.assert_not_called()


def _celery_failure_case() -> None:
    kv = make_kv_store()
    app = Celery("test")
    app.conf.task_always_eager = True
    app.conf.task_eager_propagates = True
    registry = TaskExecutionRegistry()
    handler = Mock()
    registry.register("demo.task.v1", handler)
    register_dispatcher_task(
        app=app,
        registry=registry,
        identity_persistence=wire_background_execution_identity_persistence(kv_store=kv),
        causal_evidence_persistence=failing_causal_persistence(),
    )
    with pytest.raises(RequiredAuditEvidencePersistenceError):
        app.tasks["intergrax.execute"].apply(
            kwargs={
                "logical_task_name": "demo.task.v1",
                "tenant_id": "tenant-a",
                "run_id": "queue-correlation",
                "payload": b"{}",
                "idempotency_key": None,
            }
        )
    handler.assert_not_called()


def _document_store_failure_case() -> None:
    store = InMemoryDocumentStore()
    queue = DocumentStoreTaskQueue(store)
    registry = TaskExecutionRegistry()
    handler = Mock()
    registry.register("demo.task.v1", handler)
    queue.enqueue(
        TaskRequest(
            tenant_id="tenant-a",
            run_id="queue-correlation",
            task_name="demo.task.v1",
            payload=b"{}",
        )
    )
    worker = DocumentStoreTaskWorker(
        queue,
        registry,
        identity_persistence=wire_background_execution_identity_persistence(
            document_store=store,
        ),
        causal_evidence_persistence=failing_causal_persistence(),
    )
    worker.drain_once()
    handler.assert_not_called()
    rows = queue.list_tasks("tenant-a")
    assert rows[0].status is TaskStatus.FAILED


def test_supported_paths_route_through_single_admission_gate() -> None:
    import ast
    from pathlib import Path

    repo = Path(__file__).resolve().parents[4]
    targets = [
        repo / "intergrax/queueing/providers/broker_worker_base.py",
        repo / "intergrax/background_tasks/worker_runtime.py",
        repo / "intergrax/queueing/worker/dispatcher.py",
        repo / "intergrax/queueing/providers/document_store/colocated_worker.py",
    ]
    direct_execute = 0
    admission_calls = 0
    for path in targets:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name):
                    if func.id == "execute_logical_task":
                        direct_execute += 1
                    if func.id == "admit_background_execution_handler":
                        admission_calls += 1
    assert admission_calls == 4
    assert direct_execute == 4
