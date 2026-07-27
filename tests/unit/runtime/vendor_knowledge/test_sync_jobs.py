# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for durable Vendor Knowledge sync jobs and scheduler."""

from __future__ import annotations

import base64
import hashlib
import json

import pytest
from pydantic import ValidationError

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.queueing.contracts.task_queue import TaskHandle
from intergrax.queueing.providers.document_store import DocumentStoreTaskQueue
from intergrax.runtime.vendor_knowledge.sync_jobs import (
    VENDOR_KNOWLEDGE_SYNC_JOB_SCHEMA,
    VENDOR_KNOWLEDGE_SYNC_TASK_NAME,
    VendorKnowledgeSyncJob,
    VendorKnowledgeSyncScheduler,
    decode_vendor_knowledge_sync_job,
    encode_vendor_knowledge_sync_job,
    vendor_knowledge_sync_idempotency_key,
)
from intergrax.runtime.vendor_knowledge.sync_models import KnowledgeSyncMode


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _job_from_handle(queue: DocumentStoreTaskQueue, handle: TaskHandle) -> VendorKnowledgeSyncJob:
    row = queue._load(handle)  # noqa: SLF001 - test inspects durable queue payload
    assert row is not None
    return decode_vendor_knowledge_sync_job(base64.b64decode(str(row["payload_base64"])))


@pytest.mark.unit
def test_job_strict_frozen_and_json_round_trip() -> None:
    job = VendorKnowledgeSyncJob(
        tenant_id="tenant-1",
        binding_id="binding-1",
        operation_id="op-1",
        mode=KnowledgeSyncMode.INCREMENTAL,
        restart=False,
        page_size=50,
        trigger_delivery_id=None,
        recovery_attempt=0,
    )
    assert job.schema_version == VENDOR_KNOWLEDGE_SYNC_JOB_SCHEMA
    assert job.model_config.get("frozen") is True
    with pytest.raises((ValidationError, TypeError)):
        job.tenant_id = "other"  # type: ignore[misc]
    copied = job.model_copy(update={"tenant_id": "other"})
    assert copied.tenant_id == "other"
    assert job.tenant_id == "tenant-1"
    payload = encode_vendor_knowledge_sync_job(job)
    decoded = decode_vendor_knowledge_sync_job(payload)
    assert decoded == job
    raw = json.loads(payload.decode("utf-8"))
    assert "cursor" not in raw
    assert "connection_ref" not in raw
    assert "credential_ref" not in raw
    assert "scope" not in raw


@pytest.mark.unit
def test_job_mode_restart_rules_and_delivery_validation() -> None:
    with pytest.raises(ValidationError):
        VendorKnowledgeSyncJob(
            tenant_id="t",
            binding_id="b",
            operation_id="o",
            mode=KnowledgeSyncMode.INCREMENTAL,
            restart=True,
            page_size=10,
            recovery_attempt=0,
        )
    with pytest.raises(ValidationError):
        VendorKnowledgeSyncJob(
            tenant_id="t",
            binding_id="b",
            operation_id="o",
            mode=KnowledgeSyncMode.RECONCILIATION,
            restart=False,
            page_size=10,
            recovery_attempt=0,
        )
    delivery = _sha("page-1")
    with pytest.raises(ValidationError):
        VendorKnowledgeSyncJob(
            tenant_id="t",
            binding_id="b",
            operation_id="o",
            mode=KnowledgeSyncMode.RECONCILIATION,
            restart=True,
            page_size=10,
            trigger_delivery_id=delivery,
            recovery_attempt=0,
        )
    with pytest.raises(ValidationError):
        VendorKnowledgeSyncJob(
            tenant_id="t",
            binding_id="b",
            operation_id="o",
            mode=KnowledgeSyncMode.INCREMENTAL,
            restart=False,
            page_size=10,
            trigger_delivery_id="not-a-hash",
            recovery_attempt=0,
        )
    continuation = VendorKnowledgeSyncJob(
        tenant_id="t",
        binding_id="b",
        operation_id="o",
        mode=KnowledgeSyncMode.RECONCILIATION,
        restart=False,
        page_size=10,
        trigger_delivery_id=delivery,
        recovery_attempt=0,
    )
    assert continuation.trigger_delivery_id == delivery


@pytest.mark.unit
def test_scheduler_start_continuation_recovery_and_idempotency() -> None:
    queue = DocumentStoreTaskQueue(InMemoryDocumentStore())
    scheduler = VendorKnowledgeSyncScheduler(task_queue=queue)

    incremental = scheduler.enqueue_incremental(
        tenant_id="tenant-1",
        binding_id="binding-1",
        operation_id="op-1",
        run_id="run-1",
        page_size=25,
    )
    job = _job_from_handle(queue, incremental)
    assert job.mode is KnowledgeSyncMode.INCREMENTAL
    assert job.restart is False
    assert "cursor" not in json.loads(encode_vendor_knowledge_sync_job(job).decode("utf-8"))

    recon = scheduler.enqueue_reconciliation(
        tenant_id="tenant-1",
        binding_id="binding-1",
        operation_id="op-2",
        run_id="run-2",
    )
    recon_job = _job_from_handle(queue, recon)
    assert recon_job.restart is True

    delivery = _sha("finished-page")
    cont = scheduler.enqueue_continuation(
        parent_job=recon_job,
        run_id="run-2",
        trigger_delivery_id=delivery,
    )
    cont_job = _job_from_handle(queue, cont)
    assert cont_job.restart is False
    assert cont_job.trigger_delivery_id == delivery
    assert cont_job.recovery_attempt == 0

    same = scheduler.enqueue_continuation(
        parent_job=recon_job,
        run_id="run-2",
        trigger_delivery_id=delivery,
    )
    assert same.task_id == cont.task_id

    other = scheduler.enqueue_continuation(
        parent_job=recon_job,
        run_id="run-2",
        trigger_delivery_id=_sha("other-page"),
    )
    assert other.task_id != cont.task_id

    start_key = vendor_knowledge_sync_idempotency_key(job)
    cont_key = vendor_knowledge_sync_idempotency_key(cont_job)
    assert cont_key != vendor_knowledge_sync_idempotency_key(
        cont_job.model_copy(update={"trigger_delivery_id": _sha("other-page")})
    )
    assert start_key.startswith(f"{VENDOR_KNOWLEDGE_SYNC_TASK_NAME}:")

    recovery = scheduler.enqueue_recovery(interrupted_job=cont_job, run_id="run-2")
    recovery_job = _job_from_handle(queue, recovery)
    assert recovery_job.recovery_attempt == 1
    assert recovery_job.restart is False
    assert vendor_knowledge_sync_idempotency_key(recovery_job) != cont_key
    assert recovery.task_id != cont.task_id
    assert vendor_knowledge_sync_idempotency_key(recovery_job).startswith(
        f"{VENDOR_KNOWLEDGE_SYNC_TASK_NAME}:"
    )


@pytest.mark.unit
def test_idempotency_key_collision_resistant() -> None:
    base = dict(
        tenant_id="tenant-1",
        binding_id="binding-1",
        operation_id="op-1",
        mode=KnowledgeSyncMode.INCREMENTAL,
        restart=False,
        page_size=10,
        trigger_delivery_id=None,
        recovery_attempt=0,
    )
    start = VendorKnowledgeSyncJob(**base)
    start_key = vendor_knowledge_sync_idempotency_key(start)
    assert start_key == vendor_knowledge_sync_idempotency_key(
        VendorKnowledgeSyncJob(**base)
    )
    assert start_key.startswith(f"{VENDOR_KNOWLEDGE_SYNC_TASK_NAME}:")
    digest = start_key.split(":", 1)[1]
    assert len(digest) == 64
    assert digest == digest.lower()
    assert "op-1" not in start_key
    assert "binding-1" not in start_key

    delivery = _sha("page-cont")
    continuation = VendorKnowledgeSyncJob(
        tenant_id="tenant-1",
        binding_id="binding-1",
        operation_id="op-1",
        mode=KnowledgeSyncMode.INCREMENTAL,
        restart=False,
        page_size=10,
        trigger_delivery_id=delivery,
        recovery_attempt=0,
    )
    cont_key = vendor_knowledge_sync_idempotency_key(continuation)
    assert cont_key != start_key
    assert delivery not in cont_key
    assert len(cont_key.split(":", 1)[1]) == 64

    assert vendor_knowledge_sync_idempotency_key(
        start.model_copy(update={"operation_id": "op-2"})
    ) != start_key
    assert vendor_knowledge_sync_idempotency_key(
        start.model_copy(update={"binding_id": "binding-2"})
    ) != start_key
    recon = VendorKnowledgeSyncJob(
        tenant_id="tenant-1",
        binding_id="binding-1",
        operation_id="op-1",
        mode=KnowledgeSyncMode.RECONCILIATION,
        restart=True,
        page_size=10,
        trigger_delivery_id=None,
        recovery_attempt=0,
    )
    assert vendor_knowledge_sync_idempotency_key(recon) != start_key
    assert vendor_knowledge_sync_idempotency_key(
        start.model_copy(update={"recovery_attempt": 1})
    ) != start_key
    assert vendor_knowledge_sync_idempotency_key(
        continuation.model_copy(update={"trigger_delivery_id": _sha("other")})
    ) != cont_key

    job_a = VendorKnowledgeSyncJob(
        tenant_id="tenant-1",
        binding_id="c",
        operation_id="a:b",
        mode=KnowledgeSyncMode.INCREMENTAL,
        restart=False,
        page_size=10,
        recovery_attempt=0,
    )
    job_b = VendorKnowledgeSyncJob(
        tenant_id="tenant-1",
        binding_id="b:c",
        operation_id="a",
        mode=KnowledgeSyncMode.INCREMENTAL,
        restart=False,
        page_size=10,
        recovery_attempt=0,
    )
    assert vendor_knowledge_sync_idempotency_key(job_a) != vendor_knowledge_sync_idempotency_key(
        job_b
    )


@pytest.mark.unit
def test_recovery_preserves_reconciliation_restart() -> None:
    queue = DocumentStoreTaskQueue(InMemoryDocumentStore())
    scheduler = VendorKnowledgeSyncScheduler(task_queue=queue)
    handle = scheduler.enqueue_reconciliation(
        tenant_id="tenant-1",
        binding_id="binding-1",
        operation_id="op-9",
        run_id="run-9",
    )
    job = _job_from_handle(queue, handle)
    recovered = scheduler.enqueue_recovery(interrupted_job=job, run_id="run-9")
    recovered_job = _job_from_handle(queue, recovered)
    assert recovered_job.restart is True
    assert recovered_job.recovery_attempt == 1
