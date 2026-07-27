# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Vendor Knowledge sync runtime wiring and two-page proof."""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Optional

import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import DocumentQueryResult, DocumentRecord
from intergrax.queueing.contracts.task_queue import TaskHandle, TaskRequest, TaskStatus
from intergrax.queueing.providers.document_store import DocumentStoreTaskQueue
from intergrax.runtime.vendor_knowledge.models import KnowledgeCursor
from intergrax.runtime.vendor_knowledge.sync_jobs import (
    VENDOR_KNOWLEDGE_SYNC_TASK_NAME,
    decode_vendor_knowledge_sync_job,
    encode_vendor_knowledge_sync_job,
    vendor_knowledge_sync_idempotency_key,
)
from intergrax.runtime.vendor_knowledge.sync_runtime import (
    _NamespacedDocumentStore,
    build_vendor_knowledge_sync_runtime,
)
from tests.unit.runtime.vendor_knowledge._sync_fakes import (
    IdempotentRecordingSink,
    RecordingBindingService,
    RecordingFacade,
    make_binding,
    make_change,
    make_page,
)

_QUEUE_NS = "vendor_knowledge.sync_queue.v1:tenant-1:"
_GLOBAL_PENDING = "intergrax.task_queue.v1:__pending_index__"
_LOGICAL_PENDING = "intergrax.task_queue.v1:__pending_index__"
_LOGICAL_TENANT = "intergrax.task_queue.v1:tenant-1"


class _PlainDocumentStore:
    def get(self, partition_key: str, row_key: str) -> Optional[DocumentRecord]:
        return None

    def put(self, document: DocumentRecord) -> None:
        return None

    def delete(self, partition_key: str, row_key: str) -> None:
        return None

    def query(
        self,
        partition_key: str,
        *,
        limit: int = 100,
        row_key_prefix: Optional[str] = None,
    ) -> DocumentQueryResult:
        return DocumentQueryResult(documents=[], total=0)

    def close(self) -> None:
        return None


def _build(**overrides):
    store = overrides.pop("document_store", InMemoryDocumentStore())
    binding = overrides.pop("binding", make_binding())
    kwargs = {
        "document_store": store,
        "binding_service": RecordingBindingService(binding=binding),
        "facade": RecordingFacade(),
        "sink": IdempotentRecordingSink(),
        "owner_id": "owner-1",
        "tenant_id": "tenant-1",
        "lease_ttl_seconds": 60,
        "retry_delays_seconds": (),
    }
    kwargs.update(overrides)
    return build_vendor_knowledge_sync_runtime(**kwargs), store, kwargs


@pytest.mark.unit
def test_builder_requires_conditional_and_wires_components() -> None:
    with pytest.raises(TypeError, match="ConditionalDocumentStore"):
        build_vendor_knowledge_sync_runtime(
            document_store=_PlainDocumentStore(),
            binding_service=RecordingBindingService(binding=make_binding()),  # type: ignore[arg-type]
            facade=RecordingFacade(),  # type: ignore[arg-type]
            sink=IdempotentRecordingSink(),  # type: ignore[arg-type]
            owner_id="owner-1",
            tenant_id="tenant-1",
        )
    runtime, _store, _kwargs = _build()
    assert runtime.lease_repository is not None
    assert runtime.checkpoint_repository is not None
    assert runtime.item_state_repository is not None
    assert runtime.registry.get_handler(VENDOR_KNOWLEDGE_SYNC_TASK_NAME) is not None


@pytest.mark.unit
def test_runtime_start_stop_and_main_loop_bind() -> None:
    runtime, _store, _kwargs = _build()
    loop = asyncio.new_event_loop()
    runtime.bind_main_loop(loop)
    assert runtime.main_loop() is loop
    runtime.start()
    runtime.stop()


@pytest.mark.unit
def test_interrupted_sync_task_recovery_and_isolation() -> None:
    store = InMemoryDocumentStore()
    global_queue = DocumentStoreTaskQueue(store)
    other = global_queue.enqueue(
        TaskRequest(
            tenant_id="tenant-1",
            run_id="run-x",
            task_name="other.task",
            payload=b"{}",
            idempotency_key="other-1",
        )
    )
    assert len(global_queue.claim_pending(limit=1)) == 1
    assert global_queue.get_status(other) is TaskStatus.RUNNING
    other_row_before = store.get(_LOGICAL_TENANT, other.task_id)
    assert other_row_before is not None
    other_data_before = dict(other_row_before.data)
    running_before = store.query("intergrax.task_queue.v1:__running_index__", limit=50)

    runtime, _, _ = _build(document_store=store)
    queue = runtime.task_queue
    runtime.scheduler.enqueue_incremental(
        tenant_id="tenant-1",
        binding_id="binding-1",
        operation_id="op-1",
        run_id="run-1",
    )
    claimed = queue.claim_pending(limit=1)
    assert len(claimed) == 1
    handle, request = claimed[0]
    original = decode_vendor_knowledge_sync_job(request.payload)
    original_key = vendor_knowledge_sync_idempotency_key(original)

    queue.enqueue(
        TaskRequest(
            tenant_id="tenant-1",
            run_id="run-bad",
            task_name=VENDOR_KNOWLEDGE_SYNC_TASK_NAME,
            payload=b"{not-json",
            idempotency_key="bad-1",
        )
    )
    assert len(queue.claim_pending(limit=1)) == 1

    runtime.start()
    runtime.stop()

    assert global_queue.get_status(other) is TaskStatus.RUNNING
    other_row_after = store.get(_LOGICAL_TENANT, other.task_id)
    assert other_row_after is not None
    assert dict(other_row_after.data) == other_data_before
    running_after = store.query("intergrax.task_queue.v1:__running_index__", limit=50)
    assert [doc.row_key for doc in running_after.documents] == [
        doc.row_key for doc in running_before.documents
    ]

    recovered_jobs = []
    for item in store.query(f"{_QUEUE_NS}{_LOGICAL_TENANT}", limit=100).documents:
        if str(item.data.get("task_name")) != VENDOR_KNOWLEDGE_SYNC_TASK_NAME:
            continue
        if "payload_base64" not in item.data:
            continue
        try:
            decoded = decode_vendor_knowledge_sync_job(
                base64.b64decode(str(item.data["payload_base64"]))
            )
        except (ValueError, TypeError):
            continue
        if decoded.recovery_attempt == original.recovery_attempt + 1:
            recovered_jobs.append(decoded)
    assert recovered_jobs
    recovered_job = recovered_jobs[0]
    assert vendor_knowledge_sync_idempotency_key(recovered_job) != original_key
    assert "cursor" not in json.loads(
        encode_vendor_knowledge_sync_job(recovered_job).decode("utf-8")
    )
    assert handle.task_id
    global_queue.mark_failed(other, error_message="test cleanup")


@pytest.mark.unit
def test_pending_task_isolation_from_global_queue() -> None:
    store = InMemoryDocumentStore()
    global_queue = DocumentStoreTaskQueue(store)
    other = global_queue.enqueue(
        TaskRequest(
            tenant_id="tenant-1",
            run_id="run-other",
            task_name="other.task",
            payload=b"{}",
            idempotency_key="other-pending",
        )
    )
    facade = RecordingFacade(
        pages_by_cursor={
            None: make_page(
                changes=(make_change(remote_id="item-1"),),
                has_more=False,
                proposed_checkpoint=KnowledgeCursor(value="done"),
            ),
        }
    )
    runtime, _, _ = _build(document_store=store, facade=facade)
    vendor_handle = runtime.scheduler.enqueue_incremental(
        tenant_id="tenant-1",
        binding_id="binding-1",
        operation_id="op-pending",
        run_id="run-pending",
    )
    assert runtime.worker.drain_once() == 1
    assert runtime.task_queue.get_status(vendor_handle) is TaskStatus.SUCCEEDED
    assert global_queue.get_status(other) is TaskStatus.PENDING
    with pytest.raises(ValueError, match="not registered"):
        runtime.registry.get_handler("other.task")
    assert global_queue.get_status(other) is not TaskStatus.FAILED


@pytest.mark.unit
def test_tenant_runtime_queue_isolation() -> None:
    store = InMemoryDocumentStore()
    service_a = RecordingBindingService(
        binding=make_binding(binding_id="binding-a", tenant_id="tenant-a")
    )
    service_b = RecordingBindingService(
        binding=make_binding(binding_id="binding-b", tenant_id="tenant-b")
    )
    facade_a = RecordingFacade(
        pages_by_cursor={
            None: make_page(
                changes=(make_change(remote_id="a-1"),),
                has_more=False,
                proposed_checkpoint=KnowledgeCursor(value="a-done"),
            ),
        }
    )
    facade_b = RecordingFacade(
        pages_by_cursor={
            None: make_page(
                changes=(make_change(remote_id="b-1"),),
                has_more=False,
                proposed_checkpoint=KnowledgeCursor(value="b-done"),
            ),
        }
    )
    runtime_a, _, _ = _build(
        document_store=store,
        binding_service=service_a,
        facade=facade_a,
        tenant_id="tenant-a",
        owner_id="owner-a",
    )
    runtime_b, _, _ = _build(
        document_store=store,
        binding_service=service_b,
        facade=facade_b,
        tenant_id="tenant-b",
        owner_id="owner-b",
    )
    handle_a = runtime_a.scheduler.enqueue_incremental(
        tenant_id="tenant-a", binding_id="binding-a", operation_id="op-a", run_id="run-a"
    )
    handle_b = runtime_b.scheduler.enqueue_incremental(
        tenant_id="tenant-b", binding_id="binding-b", operation_id="op-b", run_id="run-b"
    )
    assert runtime_a.worker.drain_once() == 1
    assert runtime_a.task_queue.get_status(handle_a) is TaskStatus.SUCCEEDED
    assert runtime_b.task_queue.get_status(handle_b) is TaskStatus.PENDING
    assert runtime_b.worker.drain_once() == 1
    assert runtime_b.task_queue.get_status(handle_b) is TaskStatus.SUCCEEDED
    assert service_a.get_calls == ["binding-a"]
    assert service_b.get_calls == ["binding-b"]


@pytest.mark.unit
def test_vendor_queue_rejects_foreign_tenant_and_task_type() -> None:
    store = InMemoryDocumentStore()
    runtime, _, _ = _build(document_store=store)
    queue = runtime.task_queue
    with pytest.raises(ValueError, match="tenant mismatch"):
        queue.enqueue(
            TaskRequest(
                tenant_id="other-tenant",
                run_id="run-1",
                task_name=VENDOR_KNOWLEDGE_SYNC_TASK_NAME,
                payload=b"{}",
                idempotency_key="bad-tenant",
            )
        )
    with pytest.raises(ValueError, match="task type mismatch") as type_exc:
        queue.enqueue(
            TaskRequest(
                tenant_id="tenant-1",
                run_id="run-1",
                task_name="other.task",
                payload=b"secret-payload",
                idempotency_key="bad-type",
            )
        )
    assert "secret-payload" not in str(type_exc.value)
    assert store.query(f"{_QUEUE_NS}{_LOGICAL_TENANT}", limit=50).total == 0
    assert store.query(f"{_QUEUE_NS}{_LOGICAL_PENDING}", limit=50).total == 0
    handle = runtime.scheduler.enqueue_incremental(
        tenant_id="tenant-1", binding_id="binding-1", operation_id="op-ok", run_id="run-ok"
    )
    assert queue.get_status(handle) is TaskStatus.PENDING


@pytest.mark.unit
def test_namespaced_document_store_isolation() -> None:
    underlying = InMemoryDocumentStore()
    ns_a = _NamespacedDocumentStore(
        underlying, namespace="vendor_knowledge.sync_queue.v1:tenant-a:"
    )
    ns_b = _NamespacedDocumentStore(
        underlying, namespace="vendor_knowledge.sync_queue.v1:tenant-b:"
    )
    logical = "intergrax.task_queue.v1:tenant-1"
    ns_a.put(
        DocumentRecord(
            partition_key=logical,
            row_key="task-1",
            data={"task_id": "task-1", "value": 1},
            ttl_seconds=30,
        )
    )
    fetched = ns_a.get(logical, "task-1")
    assert fetched is not None
    assert fetched.partition_key == logical
    assert fetched.row_key == "task-1"
    assert dict(fetched.data) == {"task_id": "task-1", "value": 1}
    assert fetched.ttl_seconds == 30
    assert ns_a.query(logical, limit=10).total == 1
    assert ns_b.get(logical, "task-1") is None
    underlying.put(DocumentRecord(partition_key=logical, row_key="outside", data={"x": 1}))
    assert ns_a.get(logical, "outside") is None
    ns_a.delete(logical, "task-1")
    assert ns_a.get(logical, "task-1") is None
    ns_a.close()
    assert underlying.get(logical, "outside") is not None


@pytest.mark.unit
def test_no_global_state_between_runtimes() -> None:
    first, _, _ = _build()
    second, _, _ = _build()
    assert first.registry is not second.registry
    assert first.task_queue is not second.task_queue


@pytest.mark.unit
def test_two_page_runtime_proof() -> None:
    cp1 = KnowledgeCursor(value="cp-1")
    final_delta = KnowledgeCursor(value="final-delta")
    facade = RecordingFacade(
        pages_by_cursor={
            None: make_page(
                changes=(make_change(remote_id="item-1"),),
                has_more=True,
                next_cursor=cp1,
                proposed_checkpoint=cp1,
            ),
            "cp-1": make_page(
                changes=(make_change(remote_id="item-2"),),
                proposed_checkpoint=final_delta,
                has_more=False,
            ),
        }
    )
    sink = IdempotentRecordingSink()
    store = InMemoryDocumentStore()
    runtime, _, _ = _build(document_store=store, facade=facade, sink=sink)
    first_handle = runtime.scheduler.enqueue_incremental(
        tenant_id="tenant-1",
        binding_id="binding-1",
        operation_id="op-flow",
        run_id="run-flow",
        page_size=10,
    )
    assert runtime.worker.drain_once() == 1
    assert runtime.task_queue.get_status(first_handle) is TaskStatus.SUCCEEDED
    assert len(sink.durable_delivery_ids) == 1
    first_delivery = sink.durable_delivery_ids[0]
    checkpoint = runtime.checkpoint_repository.get(
        tenant_id="tenant-1", binding_id="binding-1"
    )
    assert checkpoint is not None and checkpoint.cursor == cp1
    assert (
        runtime.item_state_repository.get(
            tenant_id="tenant-1", binding_id="binding-1", remote_id="item-1"
        )
        is not None
    )
    assert store.get("vendor_knowledge.source_lease.v1:tenant-1", "binding:binding-1") is None

    ns_pending = f"{_QUEUE_NS}{_LOGICAL_PENDING}"
    pending = [
        doc
        for doc in store.query(ns_pending, limit=50).documents
        if str(doc.data.get("tenant_id")) == "tenant-1"
    ]
    assert pending
    cont_task_id = str(pending[0].data["task_id"])
    cont_row = store.get(f"{_QUEUE_NS}{_LOGICAL_TENANT}", cont_task_id)
    assert cont_row is not None
    cont_job = decode_vendor_knowledge_sync_job(
        base64.b64decode(str(cont_row.data["payload_base64"]))
    )
    assert cont_job.trigger_delivery_id == first_delivery
    payload_obj = json.loads(
        base64.b64decode(str(cont_row.data["payload_base64"])).decode("utf-8")
    )
    assert "cursor" not in payload_obj
    cont_key = vendor_knowledge_sync_idempotency_key(cont_job)
    assert cont_key != vendor_knowledge_sync_idempotency_key(
        cont_job.model_copy(update={"trigger_delivery_id": "0" * 64})
    )
    assert first_delivery not in cont_key
    assert store.query(_GLOBAL_PENDING, limit=50).total == 0

    assert runtime.worker.drain_once() == 1
    cont_handle = TaskHandle(
        task_id=cont_task_id, provider="document_store", tenant_id="tenant-1"
    )
    assert runtime.task_queue.get_status(cont_handle) is TaskStatus.SUCCEEDED
    assert len(sink.durable_delivery_ids) == 2
    final_checkpoint = runtime.checkpoint_repository.get(
        tenant_id="tenant-1", binding_id="binding-1"
    )
    assert final_checkpoint is not None and final_checkpoint.cursor == final_delta
    assert (
        runtime.item_state_repository.get(
            tenant_id="tenant-1", binding_id="binding-1", remote_id="item-2"
        )
        is not None
    )
    assert store.query(ns_pending, limit=50).total == 0
    assert store.query(_GLOBAL_PENDING, limit=50).total == 0
    assert store.get("vendor_knowledge.source_lease.v1:tenant-1", "binding:binding-1") is None
