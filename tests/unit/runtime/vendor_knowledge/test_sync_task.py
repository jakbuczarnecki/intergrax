# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Vendor Knowledge sync queue payload and worker wiring."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.queueing.contracts.task_queue import TaskHandle
from intergrax.queueing.providers.document_store import (
    DocumentStoreTaskQueue,
    DocumentStoreTaskWorker,
)
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.runtime.vendor_knowledge.errors import (
    VendorKnowledgeError,
    VendorKnowledgeErrorCode,
)
from intergrax.runtime.vendor_knowledge.sync_models import (
    KnowledgeSyncMode,
    KnowledgeSyncRunResult,
    KnowledgeSyncRunStatus,
)
from intergrax.runtime.vendor_knowledge.sync_task import (
    VENDOR_KNOWLEDGE_SYNC_TASK_NAME,
    VendorKnowledgeSyncDispatcher,
    VendorKnowledgeSyncJob,
    VendorKnowledgeSyncWorkerOutput,
    decode_vendor_knowledge_sync_job,
    encode_vendor_knowledge_sync_job,
    make_vendor_knowledge_sync_handler,
    owner_id_for_sync_run,
    register_vendor_knowledge_sync_handler,
    vendor_knowledge_sync_idempotency_key,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _job(
    *,
    tenant_id: str = "tenant-1",
    run_id: str = "run-1",
    binding_id: str = "binding-1",
    mode: KnowledgeSyncMode = KnowledgeSyncMode.INCREMENTAL,
    page_size: int = 50,
    restart: bool = False,
) -> VendorKnowledgeSyncJob:
    return VendorKnowledgeSyncJob(
        tenant_id=tenant_id,
        run_id=run_id,
        binding_id=binding_id,
        mode=mode,
        page_size=page_size,
        restart=restart,
    )


def _completed_result(
    *,
    binding_id: str = "binding-1",
    mode: KnowledgeSyncMode = KnowledgeSyncMode.INCREMENTAL,
    has_more: bool = False,
) -> KnowledgeSyncRunResult:
    return KnowledgeSyncRunResult(
        status=KnowledgeSyncRunStatus.COMPLETED,
        mode=mode,
        tenant_id="tenant-1",
        binding_id=binding_id,
        delivery_id=_sha("delivery-1"),
        changes_count=1,
        active_count=1,
        tombstone_count=0,
        checkpoint_advanced=True,
        has_more=has_more,
        retryable=False,
    )


def _job_from_handle(queue: DocumentStoreTaskQueue, handle: TaskHandle) -> VendorKnowledgeSyncJob:
    row = queue._load(handle)  # noqa: SLF001 - inspect durable queue payload
    assert row is not None
    return decode_vendor_knowledge_sync_job(base64.b64decode(str(row["payload_base64"])))


@pytest.mark.unit
def test_job_encode_decode_round_trip_and_extra_fields_rejected() -> None:
    job = _job()
    payload = encode_vendor_knowledge_sync_job(job)
    assert decode_vendor_knowledge_sync_job(payload) == job
    raw = json.loads(payload.decode("utf-8"))
    assert set(raw) == {
        "schema_version",
        "tenant_id",
        "run_id",
        "binding_id",
        "mode",
        "page_size",
        "restart",
    }
    assert "cursor" not in raw
    assert "token" not in raw
    assert "credential_ref" not in raw
    assert "content" not in raw
    with pytest.raises(ValidationError):
        VendorKnowledgeSyncJob.model_validate({**raw, "cursor": "secret"})
    with pytest.raises(ValidationError):
        VendorKnowledgeSyncJob(
            tenant_id="t",
            run_id="r",
            binding_id="b",
            mode=KnowledgeSyncMode.INCREMENTAL,
            page_size=0,
            restart=False,
        )


@pytest.mark.unit
def test_deterministic_idempotency_key() -> None:
    job = _job()
    key_a = vendor_knowledge_sync_idempotency_key(job)
    key_b = vendor_knowledge_sync_idempotency_key(_job())
    assert key_a == key_b
    assert key_a.startswith("vendor-knowledge-sync:v1:")
    assert len(key_a.split(":")[-1]) == 64
    other = vendor_knowledge_sync_idempotency_key(_job(run_id="run-2"))
    assert other != key_a


@pytest.mark.unit
def test_dispatcher_duplicate_enqueue_returns_same_task() -> None:
    queue = DocumentStoreTaskQueue(InMemoryDocumentStore())
    dispatcher = VendorKnowledgeSyncDispatcher(queue)
    job = _job()
    first = dispatcher.enqueue(job)
    second = dispatcher.enqueue(job)
    assert first.task_id == second.task_id
    stored = _job_from_handle(queue, first)
    assert stored == job


@pytest.mark.unit
def test_handler_rejects_tenant_and_run_mismatch() -> None:
    coordinator = AsyncMock()
    handler = make_vendor_knowledge_sync_handler(lambda tenant, owner: coordinator)
    payload = encode_vendor_knowledge_sync_job(_job())
    tenant_mismatch = handler(
        tenant_id="other-tenant",
        run_id="run-1",
        payload=payload,
    )
    assert tenant_mismatch.success is False
    assert tenant_mismatch.error is not None
    assert tenant_mismatch.error.error_code == "vendor_knowledge_sync_tenant_mismatch"
    run_mismatch = handler(
        tenant_id="tenant-1",
        run_id="other-run",
        payload=payload,
    )
    assert run_mismatch.success is False
    assert run_mismatch.error is not None
    assert run_mismatch.error.error_code == "vendor_knowledge_sync_run_mismatch"
    coordinator.sync_once.assert_not_called()
    coordinator.reconcile_once.assert_not_called()


@pytest.mark.unit
def test_handler_incremental_and_reconciliation_dispatch() -> None:
    coordinator = AsyncMock()
    coordinator.sync_once = AsyncMock(return_value=_completed_result())
    coordinator.reconcile_once = AsyncMock(
        return_value=_completed_result(mode=KnowledgeSyncMode.RECONCILIATION)
    )
    captured: list[tuple[str, str]] = []

    def _factory(tenant_id: str, owner_id: str) -> Any:
        captured.append((tenant_id, owner_id))
        return coordinator

    handler = make_vendor_knowledge_sync_handler(_factory)
    incremental = handler(
        tenant_id="tenant-1",
        run_id="run-1",
        payload=encode_vendor_knowledge_sync_job(_job(page_size=25)),
    )
    assert incremental.success is True
    assert incremental.output is not None
    assert incremental.output.has_more is False
    coordinator.sync_once.assert_awaited_once_with(binding_id="binding-1", page_size=25)
    coordinator.reconcile_once.assert_not_called()
    assert captured[0] == ("tenant-1", owner_id_for_sync_run("run-1"))

    reconciliation = handler(
        tenant_id="tenant-1",
        run_id="run-2",
        payload=encode_vendor_knowledge_sync_job(
            _job(
                run_id="run-2",
                mode=KnowledgeSyncMode.RECONCILIATION,
                restart=True,
                page_size=10,
            )
        ),
    )
    assert reconciliation.success is True
    coordinator.reconcile_once.assert_awaited_once_with(
        binding_id="binding-1",
        page_size=10,
        restart=True,
    )


@pytest.mark.unit
def test_handler_lease_busy_and_error_normalization(caplog: pytest.LogCaptureFixture) -> None:
    coordinator = AsyncMock()
    coordinator.sync_once = AsyncMock(
        return_value=KnowledgeSyncRunResult(
            status=KnowledgeSyncRunStatus.LEASE_BUSY,
            mode=KnowledgeSyncMode.INCREMENTAL,
            tenant_id="tenant-1",
            binding_id="binding-1",
            delivery_id=None,
            changes_count=0,
            active_count=0,
            tombstone_count=0,
            checkpoint_advanced=False,
            has_more=False,
            retryable=True,
        )
    )
    handler = make_vendor_knowledge_sync_handler(lambda tenant, owner: coordinator)
    busy = handler(
        tenant_id="tenant-1",
        run_id="run-1",
        payload=encode_vendor_knowledge_sync_job(_job()),
    )
    assert busy.success is True
    assert busy.output is not None
    assert busy.output.status is KnowledgeSyncRunStatus.LEASE_BUSY
    assert busy.output.retryable is True
    assert busy.output.delivery_id is None

    coordinator.sync_once = AsyncMock(
        side_effect=VendorKnowledgeError(
            code=VendorKnowledgeErrorCode.RATE_LIMITED,
            safe_message="provider temporarily unavailable",
            retryable=True,
        )
    )
    failed = handler(
        tenant_id="tenant-1",
        run_id="run-1",
        payload=encode_vendor_knowledge_sync_job(_job()),
    )
    assert failed.success is False
    assert failed.error is not None
    assert failed.error.error_code == "vendor_knowledge_sync_rate_limited"
    assert "provider temporarily unavailable" in failed.error.error_message

    coordinator.sync_once = AsyncMock(side_effect=RuntimeError("raw provider boom SECRET"))
    with caplog.at_level(logging.ERROR):
        unknown = handler(
            tenant_id="tenant-1",
            run_id="run-1",
            payload=encode_vendor_knowledge_sync_job(_job()),
        )
    assert unknown.success is False
    assert unknown.error is not None
    assert unknown.error.error_code == "vendor_knowledge_sync_failed"
    assert "SECRET" not in unknown.error.error_message
    assert "raw provider boom" not in unknown.error.error_message
    assert "SECRET" not in caplog.text
    assert "RuntimeError" in caplog.text
    assert VENDOR_KNOWLEDGE_SYNC_TASK_NAME in caplog.text


@pytest.mark.unit
def test_registry_and_worker_drain_one_page() -> None:
    store = InMemoryDocumentStore()
    queue = DocumentStoreTaskQueue(store)
    registry = TaskExecutionRegistry()
    calls: list[dict[str, Any]] = []

    class _Coordinator:
        async def sync_once(self, *, binding_id: str, page_size: int) -> KnowledgeSyncRunResult:
            calls.append({"binding_id": binding_id, "page_size": page_size})
            return _completed_result(has_more=True)

        async def reconcile_once(
            self,
            *,
            binding_id: str,
            page_size: int,
            restart: bool,
        ) -> KnowledgeSyncRunResult:
            raise AssertionError("reconcile must not run")

    register_vendor_knowledge_sync_handler(
        registry,
        lambda tenant_id, owner_id: _Coordinator(),
    )
    with pytest.raises(ValueError, match="already registered"):
        register_vendor_knowledge_sync_handler(
            registry,
            lambda tenant_id, owner_id: _Coordinator(),
        )

    dispatcher = VendorKnowledgeSyncDispatcher(queue)
    handle = dispatcher.enqueue(_job(page_size=7))
    worker = DocumentStoreTaskWorker(queue, registry, claim_limit=4)
    processed = worker.drain_once()
    assert processed == 1
    assert calls == [{"binding_id": "binding-1", "page_size": 7}]
    result = queue.get_result(handle)
    assert result is not None
    assert result.status.value == "SUCCEEDED"
    assert result.output is not None
    output = VendorKnowledgeSyncWorkerOutput.model_validate(
        json.loads(result.output.decode("utf-8"))
    )
    assert output.has_more is True
    assert output.run_id == "run-1"
    # has_more must not auto-enqueue another page
    assert worker.drain_once() == 0
