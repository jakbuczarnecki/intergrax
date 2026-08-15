# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Vendor Knowledge sync worker handler."""

from __future__ import annotations

import asyncio
import hashlib
import threading
from dataclasses import dataclass, field
from typing import Any

import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.queueing.providers.document_store import DocumentStoreTaskQueue
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.runtime.vendor_knowledge.errors import (
    VendorKnowledgeError,
    VendorKnowledgeErrorCode,
)
from intergrax.runtime.vendor_knowledge.sync_jobs import (
    VENDOR_KNOWLEDGE_SYNC_TASK_NAME,
    VendorKnowledgeSyncJob,
    VendorKnowledgeSyncScheduler,
    encode_vendor_knowledge_sync_job,
)
from intergrax.runtime.vendor_knowledge.sync_models import (
    KnowledgeSyncMode,
    KnowledgeSyncRunResult,
    KnowledgeSyncRunStatus,
)
from intergrax.runtime.vendor_knowledge.sync_worker import (
    make_vendor_knowledge_sync_worker_handler,
    register_vendor_knowledge_sync_worker_handler,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _job(
    *,
    mode: KnowledgeSyncMode = KnowledgeSyncMode.INCREMENTAL,
    restart: bool = False,
    trigger_delivery_id: str | None = None,
) -> VendorKnowledgeSyncJob:
    return VendorKnowledgeSyncJob(
        tenant_id="tenant-1",
        binding_id="binding-1",
        operation_id="op-1",
        mode=mode,
        restart=restart,
        page_size=10,
        trigger_delivery_id=trigger_delivery_id,
        recovery_attempt=0,
    )


@dataclass
class _FakeCoordinator:
    sync_calls: list[dict[str, Any]] = field(default_factory=list)
    reconcile_calls: list[dict[str, Any]] = field(default_factory=list)
    results: list[KnowledgeSyncRunResult] = field(default_factory=list)
    error: Exception | None = None
    errors: list[Exception | None] = field(default_factory=list)

    async def sync_once(self, *, binding_id: str, page_size: int = 100) -> KnowledgeSyncRunResult:
        self.sync_calls.append({"binding_id": binding_id, "page_size": page_size})
        if self.errors:
            err = self.errors.pop(0)
            if err is not None:
                raise err
        if self.error is not None:
            raise self.error
        if self.results:
            return self.results.pop(0)
        return KnowledgeSyncRunResult(
            status=KnowledgeSyncRunStatus.COMPLETED,
            mode=KnowledgeSyncMode.INCREMENTAL,
            tenant_id="tenant-1",
            binding_id=binding_id,
            delivery_id=_sha("d1"),
            changes_count=1,
            active_count=1,
            tombstone_count=0,
            checkpoint_advanced=True,
            has_more=False,
            retryable=False,
        )

    async def reconcile_once(
        self,
        *,
        binding_id: str,
        page_size: int = 100,
        restart: bool = True,
        operation_id: str | None = None,
        trigger_delivery_id: str | None = None,
    ) -> KnowledgeSyncRunResult:
        self.reconcile_calls.append(
            {
                "binding_id": binding_id,
                "page_size": page_size,
                "restart": restart,
                "operation_id": operation_id,
                "trigger_delivery_id": trigger_delivery_id,
            }
        )
        if self.error is not None:
            raise self.error
        if self.results:
            return self.results.pop(0)
        return KnowledgeSyncRunResult(
            status=KnowledgeSyncRunStatus.COMPLETED,
            mode=KnowledgeSyncMode.RECONCILIATION,
            tenant_id="tenant-1",
            binding_id=binding_id,
            delivery_id=_sha("d1"),
            changes_count=0,
            active_count=0,
            tombstone_count=0,
            checkpoint_advanced=True,
            has_more=False,
            retryable=False,
        )


class _FailingContinuationScheduler:
    def __init__(self, *, task_queue: DocumentStoreTaskQueue, fail_times: int = 1) -> None:
        self._inner = VendorKnowledgeSyncScheduler(task_queue=task_queue)
        self.fail_times = fail_times
        self.calls = 0

    def enqueue_continuation(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("enqueue boom")
        return self._inner.enqueue_continuation(**kwargs)


def _completed(*, delivery: str, has_more: bool) -> KnowledgeSyncRunResult:
    return KnowledgeSyncRunResult(
        status=KnowledgeSyncRunStatus.COMPLETED,
        mode=KnowledgeSyncMode.INCREMENTAL,
        tenant_id="tenant-1",
        binding_id="binding-1",
        delivery_id=delivery,
        changes_count=1,
        active_count=1,
        tombstone_count=0,
        checkpoint_advanced=True,
        has_more=has_more,
        retryable=False,
    )


@pytest.mark.unit
def test_worker_invalid_payload_and_tenant_mismatch() -> None:
    coordinator = _FakeCoordinator()
    scheduler = VendorKnowledgeSyncScheduler(task_queue=DocumentStoreTaskQueue(InMemoryDocumentStore()))
    handler = make_vendor_knowledge_sync_worker_handler(
        coordinator_resolver=lambda _tenant, _run: coordinator,  # type: ignore[arg-type, return-value]
        scheduler=scheduler,
        sleeper=lambda _: None,
    )
    bad = handler(tenant_id="tenant-1", run_id="run-1", payload=b"{")
    assert bad.success is False
    assert bad.error is not None
    assert bad.error.error_code == "vendor_knowledge_sync_invalid_job"

    payload = encode_vendor_knowledge_sync_job(_job())
    mismatch = handler(tenant_id="other", run_id="run-1", payload=payload)
    assert mismatch.success is False
    assert mismatch.error is not None
    assert mismatch.error.error_code == "vendor_knowledge_sync_tenant_mismatch"


@pytest.mark.unit
def test_worker_incremental_and_reconciliation_paths() -> None:
    coordinator = _FakeCoordinator()
    scheduler = VendorKnowledgeSyncScheduler(task_queue=DocumentStoreTaskQueue(InMemoryDocumentStore()))
    handler = make_vendor_knowledge_sync_worker_handler(
        coordinator_resolver=lambda _tenant, _run: coordinator,  # type: ignore[arg-type, return-value]
        scheduler=scheduler,
        sleeper=lambda _: None,
    )
    result = handler(
        tenant_id="tenant-1",
        run_id="run-1",
        payload=encode_vendor_knowledge_sync_job(_job()),
    )
    assert result.success is True
    assert len(coordinator.sync_calls) == 1
    assert result.output is not None
    assert "cursor" not in result.output.model_dump()

    coordinator.sync_calls.clear()
    recon = handler(
        tenant_id="tenant-1",
        run_id="run-1",
        payload=encode_vendor_knowledge_sync_job(
            _job(mode=KnowledgeSyncMode.RECONCILIATION, restart=True)
        ),
    )
    assert recon.success is True
    assert coordinator.reconcile_calls[0]["restart"] is True

    cont = handler(
        tenant_id="tenant-1",
        run_id="run-1",
        payload=encode_vendor_knowledge_sync_job(
            _job(
                mode=KnowledgeSyncMode.RECONCILIATION,
                restart=False,
                trigger_delivery_id=_sha("prev"),
            )
        ),
    )
    assert cont.success is True
    assert coordinator.reconcile_calls[-1]["restart"] is False


@pytest.mark.unit
def test_worker_schedules_continuation_when_has_more() -> None:
    delivery = _sha("page-1")
    coordinator = _FakeCoordinator(results=[_completed(delivery=delivery, has_more=True)])
    queue = DocumentStoreTaskQueue(InMemoryDocumentStore())
    scheduler = VendorKnowledgeSyncScheduler(task_queue=queue)
    handler = make_vendor_knowledge_sync_worker_handler(
        coordinator_resolver=lambda _tenant, _run: coordinator,  # type: ignore[arg-type, return-value]
        scheduler=scheduler,
        sleeper=lambda _: None,
    )
    result = handler(
        tenant_id="tenant-1",
        run_id="run-1",
        payload=encode_vendor_knowledge_sync_job(_job()),
    )
    assert result.success is True
    assert result.output is not None
    assert result.output.continuation_task_id is not None

    coordinator.results = [_completed(delivery=_sha("final"), has_more=False)]
    result2 = handler(
        tenant_id="tenant-1",
        run_id="run-1",
        payload=encode_vendor_knowledge_sync_job(_job()),
    )
    assert result2.success is True
    assert result2.output is not None
    assert result2.output.continuation_task_id is None


@pytest.mark.unit
def test_worker_retries_lease_busy_and_retryable_errors() -> None:
    delays: list[float] = []
    coordinator = _FakeCoordinator(
        results=[
            KnowledgeSyncRunResult(
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
            ),
            _completed(delivery=_sha("ok"), has_more=False),
        ]
    )
    scheduler = VendorKnowledgeSyncScheduler(task_queue=DocumentStoreTaskQueue(InMemoryDocumentStore()))
    handler = make_vendor_knowledge_sync_worker_handler(
        coordinator_resolver=lambda _tenant, _run: coordinator,  # type: ignore[arg-type, return-value]
        scheduler=scheduler,
        retry_delays_seconds=(0.1, 0.2),
        sleeper=delays.append,
    )
    result = handler(
        tenant_id="tenant-1",
        run_id="run-1",
        payload=encode_vendor_knowledge_sync_job(_job()),
    )
    assert result.success is True
    assert delays == [0.1]
    assert result.output is not None
    assert result.output.execution_attempts == 2

    delays.clear()
    coordinator.results = []
    coordinator.errors = [
        VendorKnowledgeError(
            code=VendorKnowledgeErrorCode.RATE_LIMITED,
            safe_message="rate limited",
            retryable=True,
        ),
        None,
    ]
    coordinator.results = [_completed(delivery=_sha("after-retry"), has_more=False)]
    ok = handler(
        tenant_id="tenant-1",
        run_id="run-1",
        payload=encode_vendor_knowledge_sync_job(_job()),
    )
    assert ok.success is True
    assert delays == [0.1]


@pytest.mark.unit
def test_worker_non_retryable_and_exhausted_and_safe_errors() -> None:
    coordinator = _FakeCoordinator(
        error=VendorKnowledgeError(
            code=VendorKnowledgeErrorCode.INVALID_SCOPE,
            safe_message="invalid scope",
            retryable=False,
        )
    )
    scheduler = VendorKnowledgeSyncScheduler(task_queue=DocumentStoreTaskQueue(InMemoryDocumentStore()))
    delays: list[float] = []
    handler = make_vendor_knowledge_sync_worker_handler(
        coordinator_resolver=lambda _tenant, _run: coordinator,  # type: ignore[arg-type, return-value]
        scheduler=scheduler,
        retry_delays_seconds=(0.1,),
        sleeper=delays.append,
    )
    failed = handler(
        tenant_id="tenant-1",
        run_id="run-1",
        payload=encode_vendor_knowledge_sync_job(_job()),
    )
    assert failed.success is False
    assert failed.error is not None
    assert failed.error.error_code == "vendor_knowledge_sync_invalid_scope"
    assert delays == []

    coordinator.error = None
    coordinator.results = [
        KnowledgeSyncRunResult(
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
    ] * 5
    exhausted = handler(
        tenant_id="tenant-1",
        run_id="run-1",
        payload=encode_vendor_knowledge_sync_job(_job()),
    )
    assert exhausted.success is False
    assert exhausted.error is not None
    assert exhausted.error.error_code == "vendor_knowledge_sync_retry_exhausted"

    coordinator.results = []
    coordinator.error = RuntimeError("secret boom with cursor=ABC")
    boom = handler(
        tenant_id="tenant-1",
        run_id="run-1",
        payload=encode_vendor_knowledge_sync_job(_job()),
    )
    assert boom.success is False
    assert boom.error is not None
    assert "secret boom" not in boom.error.error_message
    assert "cursor=ABC" not in boom.error.error_code


@pytest.mark.unit
def test_worker_retries_continuation_enqueue_and_new_delivery() -> None:
    delays: list[float] = []
    first = _sha("first")
    second = _sha("second")
    coordinator = _FakeCoordinator(
        results=[
            _completed(delivery=first, has_more=True),
            _completed(delivery=second, has_more=True),
        ]
    )
    queue = DocumentStoreTaskQueue(InMemoryDocumentStore())
    scheduler = _FailingContinuationScheduler(task_queue=queue, fail_times=1)
    handler = make_vendor_knowledge_sync_worker_handler(
        coordinator_resolver=lambda _tenant, _run: coordinator,  # type: ignore[arg-type, return-value]
        scheduler=scheduler,
        retry_delays_seconds=(0.05,),
        sleeper=delays.append,
    )
    result = handler(
        tenant_id="tenant-1",
        run_id="run-1",
        payload=encode_vendor_knowledge_sync_job(_job()),
    )
    assert result.success is True
    assert delays == [0.05]
    assert result.output is not None
    assert result.output.delivery_id == second
    assert result.output.continuation_task_id is not None


@pytest.mark.unit
def test_worker_main_loop_and_asyncio_run_fallback() -> None:
    coordinator = _FakeCoordinator()
    scheduler = VendorKnowledgeSyncScheduler(task_queue=DocumentStoreTaskQueue(InMemoryDocumentStore()))
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        handler = make_vendor_knowledge_sync_worker_handler(
            coordinator_resolver=lambda _tenant, _run: coordinator,  # type: ignore[arg-type, return-value]
            scheduler=scheduler,
            main_loop_provider=lambda: loop,
            sleeper=lambda _: None,
        )
        result = handler(
            tenant_id="tenant-1",
            run_id="run-1",
            payload=encode_vendor_knowledge_sync_job(_job()),
        )
        assert result.success is True
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)
        loop.close()

    coordinator2 = _FakeCoordinator()
    handler2 = make_vendor_knowledge_sync_worker_handler(
        coordinator_resolver=lambda _tenant, _run: coordinator2,  # type: ignore[arg-type, return-value]
        scheduler=scheduler,
        main_loop_provider=lambda: None,
        sleeper=lambda _: None,
    )
    ok = handler2(
        tenant_id="tenant-1",
        run_id="run-1",
        payload=encode_vendor_knowledge_sync_job(_job()),
    )
    assert ok.success is True


@pytest.mark.unit
def test_register_duplicate_handler_fails() -> None:
    registry = TaskExecutionRegistry()
    coordinator = _FakeCoordinator()
    scheduler = VendorKnowledgeSyncScheduler(task_queue=DocumentStoreTaskQueue(InMemoryDocumentStore()))
    register_vendor_knowledge_sync_worker_handler(
        registry,
        coordinator_resolver=lambda _tenant, _run: coordinator,  # type: ignore[arg-type, return-value]
        scheduler=scheduler,
        sleeper=lambda _: None,
    )
    with pytest.raises(ValueError, match=VENDOR_KNOWLEDGE_SYNC_TASK_NAME):
        register_vendor_knowledge_sync_worker_handler(
            registry,
            coordinator_resolver=lambda _tenant, _run: coordinator,  # type: ignore[arg-type, return-value]
            scheduler=scheduler,
            sleeper=lambda _: None,
        )
