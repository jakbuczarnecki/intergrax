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
from intergrax.queueing.contracts.task_queue import TaskRequest, TaskStatus
from intergrax.runtime.vendor_knowledge.models import KnowledgeCursor
from intergrax.runtime.vendor_knowledge.sync_jobs import (
    VENDOR_KNOWLEDGE_SYNC_TASK_NAME,
    decode_vendor_knowledge_sync_job,
    encode_vendor_knowledge_sync_job,
    vendor_knowledge_sync_idempotency_key,
)
from intergrax.runtime.vendor_knowledge.sync_runtime import (
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
    binding = make_binding()
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
    runtime, store, _kwargs = _build()
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
    assert request.task_name == VENDOR_KNOWLEDGE_SYNC_TASK_NAME
    original = decode_vendor_knowledge_sync_job(request.payload)
    original_key = vendor_knowledge_sync_idempotency_key(original)

    other = queue.enqueue(
        TaskRequest(
            tenant_id="tenant-1",
            run_id="run-x",
            task_name="other.task",
            payload=b"{}",
            idempotency_key="other-1",
        )
    )
    other_claimed = queue.claim_pending(limit=1)
    assert len(other_claimed) == 1

    # Corrupt recovery payload should not stop worker start.
    queue.enqueue(
        TaskRequest(
            tenant_id="tenant-1",
            run_id="run-bad",
            task_name=VENDOR_KNOWLEDGE_SYNC_TASK_NAME,
            payload=b"{not-json",
            idempotency_key="bad-1",
        )
    )
    bad_claimed = queue.claim_pending(limit=1)
    assert len(bad_claimed) == 1

    runtime.start()
    runtime.stop()

    recovered_jobs = []
    for item in store.query("intergrax.task_queue.v1:tenant-1", limit=100).documents:
        if "task_name" not in item.data:
            continue
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
    assert queue.get_status(other) is TaskStatus.FAILED
    assert handle.task_id


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
        tenant_id="tenant-1",
        binding_id="binding-1",
    )
    assert checkpoint is not None
    assert checkpoint.cursor == cp1
    assert (
        runtime.item_state_repository.get(
            tenant_id="tenant-1",
            binding_id="binding-1",
            remote_id="item-1",
        )
        is not None
    )
    assert (
        store.get("vendor_knowledge.source_lease.v1:tenant-1", "binding:binding-1")
        is None
    )

    pending = [
        doc
        for doc in store.query("intergrax.task_queue.v1:__pending_index__", limit=50).documents
        if str(doc.data.get("tenant_id")) == "tenant-1"
    ]
    assert pending
    cont_task_id = str(pending[0].data["task_id"])
    cont_row = store.get("intergrax.task_queue.v1:tenant-1", cont_task_id)
    assert cont_row is not None
    cont_job = decode_vendor_knowledge_sync_job(
        base64.b64decode(str(cont_row.data["payload_base64"]))
    )
    assert cont_job.trigger_delivery_id == first_delivery
    assert "cursor" not in json.loads(cont_row.data["payload_base64"] and "{}")
    payload_obj = json.loads(
        base64.b64decode(str(cont_row.data["payload_base64"])).decode("utf-8")
    )
    assert "cursor" not in payload_obj
    assert first_delivery in vendor_knowledge_sync_idempotency_key(cont_job)

    assert runtime.worker.drain_once() == 1
    from intergrax.queueing.contracts.task_queue import TaskHandle

    cont_handle = TaskHandle(
        task_id=cont_task_id,
        provider="document_store",
        tenant_id="tenant-1",
    )
    assert runtime.task_queue.get_status(cont_handle) is TaskStatus.SUCCEEDED
    assert len(sink.durable_delivery_ids) == 2
    assert sink.durable_delivery_ids[0] != sink.durable_delivery_ids[1]
    final_checkpoint = runtime.checkpoint_repository.get(
        tenant_id="tenant-1",
        binding_id="binding-1",
    )
    assert final_checkpoint is not None
    assert final_checkpoint.cursor == final_delta
    assert (
        runtime.item_state_repository.get(
            tenant_id="tenant-1",
            binding_id="binding-1",
            remote_id="item-2",
        )
        is not None
    )
    # no further continuation
    still_pending = store.query("intergrax.task_queue.v1:__pending_index__", limit=50)
    assert still_pending.total == 0
    assert (
        store.get("vendor_knowledge.source_lease.v1:tenant-1", "binding:binding-1")
        is None
    )
