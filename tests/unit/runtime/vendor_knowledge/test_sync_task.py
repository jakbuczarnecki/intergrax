# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Vendor Knowledge sync application-composition adapter."""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Any

import pytest
from pydantic import ValidationError

from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.queueing.contracts.task_queue import TaskHandle
from intergrax.queueing.providers.document_store import (
    DocumentStoreTaskQueue,
    DocumentStoreTaskWorker,
)
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.runtime.background_execution.bootstrap import bootstrap_background_execution
from intergrax.runtime.background_execution.identity_persistence import (
    KvBackgroundExecutionIdentityPersistence,
)
from intergrax.runtime.background_execution.transport_ref import (
    BackgroundTransportExecutionRef,
)
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.runtime.vendor_knowledge.errors import (
    VendorKnowledgeError,
    VendorKnowledgeErrorCode,
)
from intergrax.runtime.vendor_knowledge.sync_jobs import (
    VENDOR_KNOWLEDGE_SYNC_TASK_NAME as JOBS_TASK_NAME,
)
from intergrax.runtime.vendor_knowledge.sync_jobs import (
    VendorKnowledgeSyncJob as JobsVendorKnowledgeSyncJob,
)
from intergrax.runtime.vendor_knowledge.sync_models import (
    KnowledgeSyncMode,
    KnowledgeSyncRunResult,
    KnowledgeSyncRunStatus,
)
from intergrax.runtime.vendor_knowledge.sync_task import (
    VENDOR_KNOWLEDGE_SYNC_TASK_NAME,
    VendorKnowledgeSyncDispatcher,
    VendorKnowledgeSyncHandlerRegistry,
    VendorKnowledgeSyncJob,
    VendorKnowledgeSyncWorkerOutput,
    decode_vendor_knowledge_sync_job,
    encode_vendor_knowledge_sync_job,
    make_vendor_knowledge_sync_handler,
    owner_id_for_sync_run,
    register_vendor_knowledge_indexed_sync_dimension,
    register_vendor_knowledge_sync_executable,
    register_vendor_knowledge_sync_handler,
    unregister_vendor_knowledge_sync_handler,
    vendor_knowledge_sync_idempotency_key,
)
from intergrax.runtime.vendor_knowledge.sync_worker import (
    VendorKnowledgeSyncWorkerOutput as WorkerOutput,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _execution_identity(*, tenant_id: str = "tenant-1"):
    class _KV(DistributedKVStore):
        def __init__(self) -> None:
            self._data: dict[tuple[str, str], bytes] = {}

        def get(self, tenant_id: str, key: str) -> bytes | None:
            return self._data.get((tenant_id, key))

        def set(
            self,
            tenant_id: str,
            key: str,
            value: bytes,
            *,
            ttl_seconds: int | None = None,
        ) -> None:
            self._data[(tenant_id, key)] = value

        def delete(self, tenant_id: str, key: str) -> None:
            self._data.pop((tenant_id, key), None)

        def compare_and_set(
            self,
            tenant_id: str,
            key: str,
            expected: bytes | None,
            new_value: bytes,
            *,
            ttl_seconds: int | None = None,
        ) -> bool:
            current = self.get(tenant_id, key)
            if expected is None and current is not None:
                return False
            if expected is not None and current != expected:
                return False
            self.set(tenant_id, key, new_value, ttl_seconds=ttl_seconds)
            return True

    return bootstrap_background_execution(
        transport_ref=BackgroundTransportExecutionRef(
            tenant_id=tenant_id,
            provider="document_store",
            transport_task_id=f"sync-task-{tenant_id}",
        ),
        identity_persistence=KvBackgroundExecutionIdentityPersistence(_KV()),
    )


def _job(
    *,
    tenant_id: str = "tenant-1",
    binding_id: str = "binding-1",
    operation_id: str = "op-1",
    mode: KnowledgeSyncMode = KnowledgeSyncMode.INCREMENTAL,
    page_size: int = 50,
    restart: bool = False,
    trigger_delivery_id: str | None = None,
    recovery_attempt: int = 0,
) -> VendorKnowledgeSyncJob:
    return VendorKnowledgeSyncJob(
        tenant_id=tenant_id,
        binding_id=binding_id,
        operation_id=operation_id,
        mode=mode,
        page_size=page_size,
        restart=restart,
        trigger_delivery_id=trigger_delivery_id,
        recovery_attempt=recovery_attempt,
    )


def _completed_result(
    *,
    binding_id: str = "binding-1",
    mode: KnowledgeSyncMode = KnowledgeSyncMode.INCREMENTAL,
    has_more: bool = False,
    delivery: str | None = None,
) -> KnowledgeSyncRunResult:
    return KnowledgeSyncRunResult(
        status=KnowledgeSyncRunStatus.COMPLETED,
        mode=mode,
        tenant_id="tenant-1",
        binding_id=binding_id,
        delivery_id=delivery or _sha("delivery-1"),
        changes_count=1,
        active_count=1,
        tombstone_count=0,
        checkpoint_advanced=True,
        has_more=has_more,
        retryable=False,
    )


def _job_from_handle(queue: DocumentStoreTaskQueue, handle: TaskHandle) -> VendorKnowledgeSyncJob:
    row = queue._load(handle)
    assert row is not None
    return decode_vendor_knowledge_sync_job(base64.b64decode(str(row["payload_base64"])))


@pytest.mark.unit
def test_adapter_reexports_canonical_job_and_output_identity() -> None:
    assert VendorKnowledgeSyncJob is JobsVendorKnowledgeSyncJob
    assert VendorKnowledgeSyncWorkerOutput is WorkerOutput
    assert VENDOR_KNOWLEDGE_SYNC_TASK_NAME == JOBS_TASK_NAME


@pytest.mark.unit
def test_job_encode_decode_round_trip_and_extra_fields_rejected() -> None:
    job = _job()
    payload = encode_vendor_knowledge_sync_job(job)
    assert decode_vendor_knowledge_sync_job(payload) == job
    raw = json.loads(payload.decode("utf-8"))
    assert set(raw) == {
        "schema_version",
        "tenant_id",
        "binding_id",
        "operation_id",
        "mode",
        "page_size",
        "restart",
        "trigger_delivery_id",
        "recovery_attempt",
    }
    assert "run_id" not in raw
    assert "cursor" not in raw
    with pytest.raises(ValidationError):
        VendorKnowledgeSyncJob.model_validate({**raw, "cursor": "secret"})
    with pytest.raises(ValidationError):
        VendorKnowledgeSyncJob.model_validate({**raw, "run_id": "run-1"})


@pytest.mark.unit
def test_owner_id_hashes_run_id() -> None:
    owner = owner_id_for_sync_run("run-1")
    assert owner == f"vendor_knowledge.sync:{_sha('run-1')}"
    assert "run-1" not in owner


@pytest.mark.unit
def test_deterministic_idempotency_key() -> None:
    job = _job()
    key_a = vendor_knowledge_sync_idempotency_key(job)
    key_b = vendor_knowledge_sync_idempotency_key(_job())
    assert key_a == key_b
    assert key_a.startswith(f"{VENDOR_KNOWLEDGE_SYNC_TASK_NAME}:")
    other = vendor_knowledge_sync_idempotency_key(_job(operation_id="op-2"))
    assert other != key_a


@pytest.mark.unit
def test_dispatcher_duplicate_enqueue_returns_same_task() -> None:
    queue = DocumentStoreTaskQueue(InMemoryDocumentStore())
    dispatcher = VendorKnowledgeSyncDispatcher(queue)
    job = _job()
    first = dispatcher.enqueue(job=job, run_id="run-1")
    second = dispatcher.enqueue(job=job, run_id="run-1")
    assert first.task_id == second.task_id
    stored = _job_from_handle(queue, first)
    assert stored == job
    assert "run_id" not in json.loads(encode_vendor_knowledge_sync_job(stored).decode("utf-8"))


@pytest.mark.unit
def test_handler_rejects_tenant_mismatch() -> None:
    coordinator = type("C", (), {})()
    queue = DocumentStoreTaskQueue(InMemoryDocumentStore())
    dispatcher = VendorKnowledgeSyncDispatcher(queue)
    handler = make_vendor_knowledge_sync_handler(
        lambda tenant, owner: coordinator,  # type: ignore[arg-type, return-value]
        dispatcher,
        retry_delays_seconds=(),
        sleeper=lambda _: None,
    )
    payload = encode_vendor_knowledge_sync_job(_job())
    tenant_mismatch = handler(
        tenant_id="other-tenant",
        run_id="run-1",
        payload=payload,
        execution_identity=_execution_identity(tenant_id="other-tenant"),
    )
    assert tenant_mismatch.success is False
    assert tenant_mismatch.error is not None
    assert tenant_mismatch.error.error_code == "vendor_knowledge_sync_tenant_mismatch"


@pytest.mark.unit
def test_handler_incremental_and_reconciliation_dispatch() -> None:
    class _Coordinator:
        def __init__(self) -> None:
            self.sync_calls: list[dict[str, Any]] = []
            self.reconcile_calls: list[dict[str, Any]] = []

        async def sync_once(self, *, binding_id: str, page_size: int) -> KnowledgeSyncRunResult:
            self.sync_calls.append({"binding_id": binding_id, "page_size": page_size})
            return _completed_result()

        async def reconcile_once(
            self,
            *,
            binding_id: str,
            page_size: int,
            restart: bool,
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
            return _completed_result(mode=KnowledgeSyncMode.RECONCILIATION)

    coordinator = _Coordinator()
    captured: list[tuple[str, str]] = []

    def _factory(tenant_id: str, owner_id: str) -> Any:
        captured.append((tenant_id, owner_id))
        return coordinator

    queue = DocumentStoreTaskQueue(InMemoryDocumentStore())
    dispatcher = VendorKnowledgeSyncDispatcher(queue)
    handler = make_vendor_knowledge_sync_handler(
        _factory,
        dispatcher,
        retry_delays_seconds=(),
        sleeper=lambda _: None,
    )
    incremental = handler(
        tenant_id="tenant-1",
        run_id="run-1",
        payload=encode_vendor_knowledge_sync_job(_job(page_size=25)),
        execution_identity=_execution_identity(),
    )
    assert incremental.success is True
    assert coordinator.sync_calls == [{"binding_id": "binding-1", "page_size": 25}]
    assert captured[0] == ("tenant-1", owner_id_for_sync_run("run-1"))

    reconciliation = handler(
        tenant_id="tenant-1",
        run_id="run-2",
        payload=encode_vendor_knowledge_sync_job(
            _job(
                mode=KnowledgeSyncMode.RECONCILIATION,
                restart=True,
                page_size=10,
            )
        ),
        execution_identity=_execution_identity(),
    )
    assert reconciliation.success is True
    assert coordinator.reconcile_calls == [
        {
            "binding_id": "binding-1",
            "page_size": 10,
            "restart": True,
            "operation_id": "op-1",
            "trigger_delivery_id": None,
        }
    ]


@pytest.mark.unit
def test_handler_error_normalization() -> None:
    class _Coordinator:
        def __init__(self, error: Exception) -> None:
            self._error = error

        async def sync_once(self, *, binding_id: str, page_size: int) -> KnowledgeSyncRunResult:
            raise self._error

        async def reconcile_once(
            self,
            *,
            binding_id: str,
            page_size: int,
            restart: bool,
        ) -> KnowledgeSyncRunResult:
            raise AssertionError("unused")

    queue = DocumentStoreTaskQueue(InMemoryDocumentStore())
    dispatcher = VendorKnowledgeSyncDispatcher(queue)
    handler = make_vendor_knowledge_sync_handler(
        lambda tenant, owner: _Coordinator(  # type: ignore[arg-type, return-value]
            VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.RATE_LIMITED,
                safe_message="provider temporarily unavailable",
                retryable=False,
            )
        ),
        dispatcher,
        retry_delays_seconds=(),
        sleeper=lambda _: None,
    )
    failed = handler(
        tenant_id="tenant-1",
        run_id="run-1",
        payload=encode_vendor_knowledge_sync_job(_job()),
        execution_identity=_execution_identity(),
    )
    assert failed.success is False
    assert failed.error is not None
    assert failed.error.error_code == "vendor_knowledge_sync_rate_limited"
    assert "provider temporarily unavailable" in failed.error.error_message

    boom_handler = make_vendor_knowledge_sync_handler(
        lambda tenant, owner: _Coordinator(  # type: ignore[arg-type, return-value]
            RuntimeError("raw provider boom SECRET")
        ),
        dispatcher,
        retry_delays_seconds=(),
        sleeper=lambda _: None,
    )
    unknown = boom_handler(
        tenant_id="tenant-1",
        run_id="run-1",
        payload=encode_vendor_knowledge_sync_job(_job()),
        execution_identity=_execution_identity(),
    )
    assert unknown.success is False
    assert unknown.error is not None
    assert unknown.error.error_code == "vendor_knowledge_sync_retry_exhausted"
    assert "SECRET" not in unknown.error.error_message


@pytest.mark.unit
def test_registry_and_worker_schedules_continuation() -> None:
    store = InMemoryDocumentStore()
    queue = DocumentStoreTaskQueue(store)
    registry = TaskExecutionRegistry()
    calls: list[dict[str, Any]] = []
    deliveries = [_sha("page-1"), _sha("page-2")]

    class _Coordinator:
        async def sync_once(self, *, binding_id: str, page_size: int) -> KnowledgeSyncRunResult:
            calls.append({"binding_id": binding_id, "page_size": page_size})
            delivery = deliveries[len(calls) - 1]
            return _completed_result(has_more=len(calls) == 1, delivery=delivery)

        async def reconcile_once(
            self,
            *,
            binding_id: str,
            page_size: int,
            restart: bool,
        ) -> KnowledgeSyncRunResult:
            raise AssertionError("reconcile must not run")

    dispatcher = VendorKnowledgeSyncDispatcher(queue)
    register_vendor_knowledge_sync_handler(
        registry,
        lambda tenant_id, owner_id: _Coordinator(),
        dispatcher,
        retry_delays_seconds=(),
        sleeper=lambda _: None,
    )
    with pytest.raises(ValueError, match="already registered"):
        register_vendor_knowledge_sync_handler(
            registry,
            lambda tenant_id, owner_id: _Coordinator(),
            dispatcher,
            retry_delays_seconds=(),
            sleeper=lambda _: None,
        )

    handle = dispatcher.enqueue(job=_job(page_size=7), run_id="run-1")
    worker = DocumentStoreTaskWorker(queue, registry, claim_limit=4)
    assert worker.drain_once() == 1
    assert calls == [{"binding_id": "binding-1", "page_size": 7}]
    result = queue.get_result(handle)
    assert result is not None
    assert result.status.value == "SUCCEEDED"
    assert result.output is not None
    output = VendorKnowledgeSyncWorkerOutput.model_validate(
        json.loads(result.output.decode("utf-8"))
    )
    assert output.has_more is True
    assert output.continuation_task_id is not None
    assert worker.drain_once() == 1
    assert len(calls) == 2


@pytest.mark.unit
def test_sync_handler_registry_requires_exact_dimensions_and_executable_handler() -> None:
    registry = VendorKnowledgeSyncHandlerRegistry()

    def handler() -> None:
        return None

    registry.register(
        provider_id="provider-a",
        integration_kind=IntegrationCategory.ISSUE_TRACKER,
        source_kind="issues",
        handler_ref="synthetic.sync.v1",
        handler=handler,
        registration_version="registration-7",
    )

    assert (
        registry.resolve_handler(
            provider_id="provider-a",
            integration_kind=IntegrationCategory.ISSUE_TRACKER,
            source_kind="issues",
            handler_ref="synthetic.sync.v1",
        )
        is handler
    )
    assert (
        registry.handler_registration_version(
            provider_id="provider-a",
            integration_kind=IntegrationCategory.ISSUE_TRACKER,
            source_kind="issues",
            handler_ref="synthetic.sync.v1",
        )
        == "registration-7"
    )
    assert (
        registry.resolve_handler(
            provider_id="provider-b",
            integration_kind=IntegrationCategory.ISSUE_TRACKER,
            source_kind="issues",
            handler_ref="synthetic.sync.v1",
        )
        is None
    )
    assert (
        registry.resolve_handler(
            provider_id="provider-a",
            integration_kind=IntegrationCategory.ISSUE_TRACKER,
            source_kind="documents",
            handler_ref="synthetic.sync.v1",
        )
        is None
    )

    inactive_handler = handler
    registry.register(
        provider_id="provider-a",
        integration_kind=IntegrationCategory.ISSUE_TRACKER,
        source_kind="broken",
        handler_ref="broken.sync.v1",
        handler=inactive_handler,
        active=False,
    )
    assert (
        registry.resolve_handler(
            provider_id="provider-a",
            integration_kind=IntegrationCategory.ISSUE_TRACKER,
            source_kind="broken",
            handler_ref="broken.sync.v1",
        )
        is None
    )
    assert registry.unregister(
        provider_id="provider-a",
        integration_kind=IntegrationCategory.ISSUE_TRACKER,
        source_kind="issues",
        handler_ref="synthetic.sync.v1",
    )


@pytest.mark.unit
def test_canonical_registration_rolls_back_executable_when_metadata_publish_fails() -> None:
    class _FailingRegistry(VendorKnowledgeSyncHandlerRegistry):
        def _publish_metadata(self, key, registration) -> None:
            raise RuntimeError("metadata publication failed")

    task_registry = TaskExecutionRegistry()
    handler_registry = _FailingRegistry(task_registry)
    dispatcher = VendorKnowledgeSyncDispatcher(
        DocumentStoreTaskQueue(InMemoryDocumentStore())
    )

    with pytest.raises(RuntimeError, match="metadata publication failed"):
        register_vendor_knowledge_sync_handler(
            task_registry,
            lambda _tenant_id, _owner_id: object(),
            dispatcher,
            handler_registry=handler_registry,
            provider_id="provider-a",
            integration_kind=IntegrationCategory.ISSUE_TRACKER,
            source_kind="issues",
            handler_ref="synthetic.sync.v1",
            registration_version="registration-1",
        )

    with pytest.raises(ValueError, match="not registered"):
        task_registry.get_handler(VENDOR_KNOWLEDGE_SYNC_TASK_NAME)
    assert handler_registry.resolve_registration(
        provider_id="provider-a",
        integration_kind=IntegrationCategory.ISSUE_TRACKER,
        source_kind="issues",
        handler_ref="synthetic.sync.v1",
    ) is None


@pytest.mark.unit
def test_canonical_registration_does_not_publish_metadata_when_executable_fails() -> None:
    task_registry = TaskExecutionRegistry()

    def existing_handler(*_args: object, **_kwargs: object) -> None:
        return None

    task_registry.register(VENDOR_KNOWLEDGE_SYNC_TASK_NAME, existing_handler)
    handler_registry = VendorKnowledgeSyncHandlerRegistry(task_registry)
    dispatcher = VendorKnowledgeSyncDispatcher(
        DocumentStoreTaskQueue(InMemoryDocumentStore())
    )

    with pytest.raises(ValueError, match="executable_mismatch"):
        register_vendor_knowledge_sync_handler(
            task_registry,
            lambda _tenant_id, _owner_id: object(),
            dispatcher,
            handler_registry=handler_registry,
            provider_id="provider-a",
            integration_kind=IntegrationCategory.ISSUE_TRACKER,
            source_kind="issues",
            handler_ref="synthetic.sync.v1",
            registration_version="registration-1",
        )

    assert task_registry.get_handler(VENDOR_KNOWLEDGE_SYNC_TASK_NAME) is existing_handler
    assert handler_registry.resolve_registration(
        provider_id="provider-a",
        integration_kind=IntegrationCategory.ISSUE_TRACKER,
        source_kind="issues",
        handler_ref="synthetic.sync.v1",
    ) is None


@pytest.mark.unit
def test_duplicate_canonical_registration_changes_neither_store() -> None:
    task_registry = TaskExecutionRegistry()
    handler_registry = VendorKnowledgeSyncHandlerRegistry(task_registry)
    dispatcher = VendorKnowledgeSyncDispatcher(
        DocumentStoreTaskQueue(InMemoryDocumentStore())
    )
    registration = {
        "handler_registry": handler_registry,
        "provider_id": "provider-a",
        "integration_kind": IntegrationCategory.ISSUE_TRACKER,
        "source_kind": "issues",
        "handler_ref": "synthetic.sync.v1",
        "registration_version": "registration-1",
    }
    register_vendor_knowledge_sync_handler(
        task_registry,
        lambda _tenant_id, _owner_id: object(),
        dispatcher,
        **registration,
    )
    handler = task_registry.get_handler(VENDOR_KNOWLEDGE_SYNC_TASK_NAME)
    view = handler_registry.resolve_registration(
        provider_id="provider-a",
        integration_kind=IntegrationCategory.ISSUE_TRACKER,
        source_kind="issues",
        handler_ref="synthetic.sync.v1",
    )

    with pytest.raises(ValueError, match="already_registered"):
        register_vendor_knowledge_sync_handler(
            task_registry,
            lambda _tenant_id, _owner_id: object(),
            dispatcher,
            **registration,
        )

    assert task_registry.get_handler(VENDOR_KNOWLEDGE_SYNC_TASK_NAME) is handler
    assert handler_registry.resolve_registration(
        provider_id="provider-a",
        integration_kind=IntegrationCategory.ISSUE_TRACKER,
        source_kind="issues",
        handler_ref="synthetic.sync.v1",
    ) == view


@pytest.mark.unit
def test_canonical_executable_configuration_mismatch_preserves_registration() -> None:
    task_registry = TaskExecutionRegistry()
    handler_registry = VendorKnowledgeSyncHandlerRegistry(task_registry)
    dispatcher = VendorKnowledgeSyncDispatcher(
        DocumentStoreTaskQueue(InMemoryDocumentStore())
    )
    first = register_vendor_knowledge_sync_executable(
        task_registry=task_registry,
        coordinator_factory=lambda _tenant_id, _owner_id: object(),  # type: ignore[return-value]
        dispatcher=dispatcher,
        registration_token="configuration-a",
        retry_delays_seconds=(),
        sleeper=lambda _: None,
    )
    register_vendor_knowledge_indexed_sync_dimension(
        handler_registry=handler_registry,
        provider_id="provider-a",
        integration_kind=IntegrationCategory.ISSUE_TRACKER,
        source_kind="issues",
        handler_ref="synthetic.sync.v1",
        registration_version="registration-a",
        executable_registration=first,
    )

    with pytest.raises(ValueError, match="executable_mismatch"):
        register_vendor_knowledge_sync_handler(
            task_registry=task_registry,
            coordinator_factory=lambda _tenant_id, _owner_id: object(),  # type: ignore[return-value]
            dispatcher=dispatcher,
            handler_registry=handler_registry,
            provider_id="provider-b",
            integration_kind=IntegrationCategory.COLLABORATION_SUITE,
            source_kind="documents",
            handler_ref="synthetic.sync.v1",
            registration_version="registration-b",
            registration_token="configuration-b",
            retry_delays_seconds=(),
            sleeper=lambda _: None,
        )

    assert task_registry.get_handler(VENDOR_KNOWLEDGE_SYNC_TASK_NAME) is first.handler
    assert handler_registry.resolve_registration(
        provider_id="provider-a",
        integration_kind=IntegrationCategory.ISSUE_TRACKER,
        source_kind="issues",
        handler_ref="synthetic.sync.v1",
    ) is not None
    assert handler_registry.resolve_registration(
        provider_id="provider-b",
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        source_kind="documents",
        handler_ref="synthetic.sync.v1",
    ) is None


@pytest.mark.unit
def test_shared_executable_removal_preserves_other_dimension_then_removes_owned_task() -> None:
    task_registry = TaskExecutionRegistry()
    handler_registry = VendorKnowledgeSyncHandlerRegistry(task_registry)
    dispatcher = VendorKnowledgeSyncDispatcher(
        DocumentStoreTaskQueue(InMemoryDocumentStore())
    )
    first = register_vendor_knowledge_sync_handler(
        task_registry=task_registry,
        coordinator_factory=lambda _tenant_id, _owner_id: object(),  # type: ignore[return-value]
        dispatcher=dispatcher,
        handler_registry=handler_registry,
        provider_id="provider-a",
        integration_kind=IntegrationCategory.ISSUE_TRACKER,
        source_kind="issues",
        handler_ref="synthetic.sync.v1",
        registration_version="registration-a",
        registration_token="configuration-a",
        retry_delays_seconds=(),
        sleeper=lambda _: None,
    )
    register_vendor_knowledge_sync_handler(
        task_registry=task_registry,
        handler_registry=handler_registry,
        provider_id="provider-b",
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        source_kind="documents",
        handler_ref="synthetic.sync.v1",
        registration_version="registration-b",
        executable_registration=first,
    )

    assert unregister_vendor_knowledge_sync_handler(
        handler_registry,
        provider_id="provider-a",
        integration_kind=IntegrationCategory.ISSUE_TRACKER,
        source_kind="issues",
        handler_ref="synthetic.sync.v1",
    )
    assert task_registry.get_handler(VENDOR_KNOWLEDGE_SYNC_TASK_NAME) is first.handler
    assert handler_registry.resolve_handler(
        provider_id="provider-b",
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        source_kind="documents",
        handler_ref="synthetic.sync.v1",
    ) is first.handler

    assert unregister_vendor_knowledge_sync_handler(
        handler_registry,
        provider_id="provider-b",
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        source_kind="documents",
        handler_ref="synthetic.sync.v1",
    )
    with pytest.raises(ValueError, match="not registered"):
        task_registry.get_handler(VENDOR_KNOWLEDGE_SYNC_TASK_NAME)
