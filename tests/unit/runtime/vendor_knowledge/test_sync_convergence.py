# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Convergence proof for parallel Vendor Knowledge sync path merge."""

from __future__ import annotations

import ast
import base64
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import intergrax.runtime.vendor_knowledge as vk
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.background_execution.identity_persistence import wire_background_execution_identity_persistence
from intergrax.queueing.contracts.task_queue import TaskHandle
from intergrax.queueing.providers.document_store import (
    DocumentStoreTaskQueue,
    DocumentStoreTaskWorker,
)
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.runtime.vendor_knowledge import sync_jobs, sync_task, sync_worker
from intergrax.runtime.vendor_knowledge.models import KnowledgeCursor
from intergrax.runtime.vendor_knowledge.sync_coordinator import VendorKnowledgeSyncCoordinator
from intergrax.runtime.vendor_knowledge.sync_document_store import (
    DocumentStoreKnowledgeRemoteItemStateRepository,
    DocumentStoreKnowledgeSourceLeaseRepository,
    DocumentStoreKnowledgeSyncCheckpointRepository,
)
from intergrax.runtime.vendor_knowledge.sync_jobs import (
    VENDOR_KNOWLEDGE_SYNC_JOB_SCHEMA,
    VENDOR_KNOWLEDGE_SYNC_TASK_NAME,
    VendorKnowledgeSyncJob,
    VendorKnowledgeSyncScheduler,
    decode_vendor_knowledge_sync_job,
    encode_vendor_knowledge_sync_job,
)
from intergrax.runtime.vendor_knowledge.sync_models import (
    KnowledgeSyncMode,
    KnowledgeSyncRunResult,
    KnowledgeSyncRunStatus,
)
from intergrax.runtime.vendor_knowledge.sync_runtime import build_vendor_knowledge_sync_runtime
from intergrax.runtime.vendor_knowledge.sync_task import (
    VendorKnowledgeSyncDispatcher,
    make_vendor_knowledge_sync_handler,
    owner_id_for_sync_run,
    register_vendor_knowledge_sync_handler,
)
from tests.unit.runtime.vendor_knowledge._sync_fakes import (
    IdempotentRecordingSink,
    RecordingBindingService,
    RecordingFacade,
    make_binding,
    make_change,
    make_page,
)

_PACKAGE = Path(__file__).resolve().parents[4] / "intergrax" / "runtime" / "vendor_knowledge"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _class_defs(module_name: str, class_name: str) -> list[str]:
    path = _PACKAGE / f"{module_name}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]


def _assign_targets(module_name: str, name: str) -> list[str]:
    path = _PACKAGE / f"{module_name}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    found.append(name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == name:
                found.append(name)
    return found


def _job_from_handle(queue: DocumentStoreTaskQueue, handle: TaskHandle) -> VendorKnowledgeSyncJob:
    row = queue._load(handle)  # noqa: SLF001
    assert row is not None
    return decode_vendor_knowledge_sync_job(base64.b64decode(str(row["payload_base64"])))


@pytest.mark.unit
def test_exactly_one_job_model_and_schema_definition() -> None:
    assert _class_defs("sync_jobs", "VendorKnowledgeSyncJob") == ["VendorKnowledgeSyncJob"]
    assert _class_defs("sync_task", "VendorKnowledgeSyncJob") == []
    assert _class_defs("sync_worker", "VendorKnowledgeSyncJob") == []
    assert _assign_targets("sync_jobs", "VENDOR_KNOWLEDGE_SYNC_TASK_NAME") == [
        "VENDOR_KNOWLEDGE_SYNC_TASK_NAME"
    ]
    assert _assign_targets("sync_jobs", "VENDOR_KNOWLEDGE_SYNC_JOB_SCHEMA") == [
        "VENDOR_KNOWLEDGE_SYNC_JOB_SCHEMA"
    ]
    assert _assign_targets("sync_task", "VENDOR_KNOWLEDGE_SYNC_TASK_NAME") == []
    assert _assign_targets("sync_task", "VENDOR_KNOWLEDGE_SYNC_JOB_SCHEMA") == []
    assert _class_defs("sync_worker", "VendorKnowledgeSyncWorkerOutput") == [
        "VendorKnowledgeSyncWorkerOutput"
    ]
    assert _class_defs("sync_task", "VendorKnowledgeSyncWorkerOutput") == []


@pytest.mark.unit
def test_sync_task_reexports_canonical_identity() -> None:
    assert sync_task.VendorKnowledgeSyncJob is sync_jobs.VendorKnowledgeSyncJob
    assert sync_task.VendorKnowledgeSyncWorkerOutput is sync_worker.VendorKnowledgeSyncWorkerOutput
    assert vk.VendorKnowledgeSyncJob is sync_jobs.VendorKnowledgeSyncJob
    assert vk.VendorKnowledgeSyncWorkerOutput is sync_worker.VendorKnowledgeSyncWorkerOutput
    assert vk.VENDOR_KNOWLEDGE_SYNC_TASK_NAME == VENDOR_KNOWLEDGE_SYNC_TASK_NAME
    assert vk.VENDOR_KNOWLEDGE_SYNC_JOB_SCHEMA == VENDOR_KNOWLEDGE_SYNC_JOB_SCHEMA


@pytest.mark.unit
def test_application_dispatcher_and_handler_use_canonical_contract() -> None:
    queue = DocumentStoreTaskQueue(InMemoryDocumentStore())
    dispatcher = VendorKnowledgeSyncDispatcher(queue)
    job = VendorKnowledgeSyncJob(
        tenant_id="tenant-1",
        binding_id="binding-1",
        operation_id="op-1",
        mode=KnowledgeSyncMode.INCREMENTAL,
        restart=False,
        page_size=10,
        trigger_delivery_id=None,
        recovery_attempt=0,
    )
    handle = dispatcher.enqueue(job=job, run_id="run-secret")
    decoded = _job_from_handle(queue, handle)
    assert decoded == job
    raw = json.loads(encode_vendor_knowledge_sync_job(decoded).decode("utf-8"))
    assert "run_id" not in raw
    assert "cursor" not in raw

    class _Coordinator:
        async def sync_once(self, *, binding_id: str, page_size: int) -> KnowledgeSyncRunResult:
            return KnowledgeSyncRunResult(
                status=KnowledgeSyncRunStatus.COMPLETED,
                mode=KnowledgeSyncMode.INCREMENTAL,
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

        async def reconcile_once(self, **kwargs: Any) -> KnowledgeSyncRunResult:
            raise AssertionError("unused")

    captured: list[tuple[str, str]] = []

    def _factory(tenant_id: str, owner_id: str) -> Any:
        captured.append((tenant_id, owner_id))
        return _Coordinator()

    handler = make_vendor_knowledge_sync_handler(
        _factory,
        dispatcher,
        retry_delays_seconds=(),
        sleeper=lambda _: None,
    )
    result = handler(
        tenant_id="tenant-1",
        run_id="run-secret",
        payload=encode_vendor_knowledge_sync_job(job),
    )
    assert result.success is True
    assert captured == [("tenant-1", owner_id_for_sync_run("run-secret"))]
    assert "run-secret" not in owner_id_for_sync_run("run-secret")


@pytest.mark.unit
def test_application_adapter_schedules_continuation_with_delivery_id() -> None:
    queue = DocumentStoreTaskQueue(InMemoryDocumentStore())
    dispatcher = VendorKnowledgeSyncDispatcher(queue)
    delivery = _sha("page-1")

    class _Coordinator:
        async def sync_once(self, *, binding_id: str, page_size: int) -> KnowledgeSyncRunResult:
            return KnowledgeSyncRunResult(
                status=KnowledgeSyncRunStatus.COMPLETED,
                mode=KnowledgeSyncMode.INCREMENTAL,
                tenant_id="tenant-1",
                binding_id=binding_id,
                delivery_id=delivery,
                changes_count=1,
                active_count=1,
                tombstone_count=0,
                checkpoint_advanced=True,
                has_more=True,
                retryable=False,
            )

        async def reconcile_once(self, **kwargs: Any) -> KnowledgeSyncRunResult:
            raise AssertionError("unused")

    handler = make_vendor_knowledge_sync_handler(
        lambda tenant, owner: _Coordinator(),
        dispatcher,
        retry_delays_seconds=(),
        sleeper=lambda _: None,
    )
    job = VendorKnowledgeSyncJob(
        tenant_id="tenant-1",
        binding_id="binding-1",
        operation_id="op-1",
        mode=KnowledgeSyncMode.INCREMENTAL,
        restart=False,
        page_size=5,
        trigger_delivery_id=None,
        recovery_attempt=0,
    )
    result = handler(
        tenant_id="tenant-1",
        run_id="run-1",
        payload=encode_vendor_knowledge_sync_job(job),
    )
    assert result.success is True
    assert result.output is not None
    assert result.output.has_more is True
    assert result.output.continuation_task_id is not None
    cont_job = _job_from_handle(
        queue,
        TaskHandle(
            task_id=result.output.continuation_task_id,
            provider="document_store",
            tenant_id="tenant-1",
        ),
    )
    assert cont_job.trigger_delivery_id == delivery
    assert "run_id" not in json.loads(encode_vendor_knowledge_sync_job(cont_job).decode("utf-8"))


@pytest.mark.unit
def test_legacy_run_id_payload_rejected() -> None:
    legacy = {
        "schema_version": VENDOR_KNOWLEDGE_SYNC_JOB_SCHEMA,
        "tenant_id": "tenant-1",
        "run_id": "run-1",
        "binding_id": "binding-1",
        "mode": "incremental",
        "page_size": 10,
        "restart": False,
    }
    with pytest.raises((ValidationError, ValueError)):
        decode_vendor_knowledge_sync_job(
            json.dumps(legacy, separators=(",", ":"), sort_keys=True).encode("utf-8")
        )


@pytest.mark.unit
def test_standalone_runtime_uses_same_job_format() -> None:
    runtime = build_vendor_knowledge_sync_runtime(
        document_store=InMemoryDocumentStore(),
        binding_service=RecordingBindingService(binding=make_binding()),  # type: ignore[arg-type]
        facade=RecordingFacade(
            default_page=make_page(
                changes=(make_change(remote_id="i1"),),
                proposed_checkpoint=KnowledgeCursor(value="cp1"),
            )
        ),  # type: ignore[arg-type]
        sink=IdempotentRecordingSink(),  # type: ignore[arg-type]
        owner_id="owner-1",
        tenant_id="tenant-1",
        retry_delays_seconds=(),
    )
    assert isinstance(runtime.scheduler, VendorKnowledgeSyncScheduler)
    handle = runtime.scheduler.enqueue_incremental(
        tenant_id="tenant-1",
        binding_id="binding-1",
        operation_id="op-runtime",
        run_id="run-runtime",
        page_size=10,
    )
    job = _job_from_handle(runtime.task_queue, handle)
    assert job.operation_id == "op-runtime"
    assert "run_id" not in json.loads(encode_vendor_knowledge_sync_job(job).decode("utf-8"))


@pytest.mark.unit
def test_lkw_style_composition_without_vendor_runtime() -> None:
    store = InMemoryDocumentStore()
    queue = DocumentStoreTaskQueue(store)
    registry = TaskExecutionRegistry()
    dispatcher = VendorKnowledgeSyncDispatcher(queue)
    facade = RecordingFacade(
        pages_by_cursor={
            None: make_page(
                changes=(make_change(remote_id="a"),),
                next_cursor=KnowledgeCursor(value="cp1"),
                proposed_checkpoint=KnowledgeCursor(value="cp1"),
                has_more=True,
            ),
            "cp1": make_page(
                changes=(make_change(remote_id="b"),),
                proposed_checkpoint=KnowledgeCursor(value="cp2"),
                has_more=False,
            ),
        }
    )
    binding_service = RecordingBindingService(binding=make_binding())
    sink = IdempotentRecordingSink()

    def _factory(tenant_id: str, owner_id: str) -> VendorKnowledgeSyncCoordinator:
        return VendorKnowledgeSyncCoordinator(
            tenant_id=tenant_id,
            owner_id=owner_id,
            binding_service=binding_service,  # type: ignore[arg-type]
            facade=facade,  # type: ignore[arg-type]
            lease_repository=DocumentStoreKnowledgeSourceLeaseRepository(store),
            checkpoint_repository=DocumentStoreKnowledgeSyncCheckpointRepository(store),
            item_state_repository=DocumentStoreKnowledgeRemoteItemStateRepository(store),
            sink=sink,  # type: ignore[arg-type]
            lease_ttl_seconds=60,
        )

    register_vendor_knowledge_sync_handler(
        registry,
        _factory,
        dispatcher,
        retry_delays_seconds=(),
        sleeper=lambda _: None,
    )
    worker = DocumentStoreTaskWorker(queue, registry, claim_limit=4, identity_persistence=wire_background_execution_identity_persistence(document_store=store))
    start = dispatcher.enqueue_incremental(
        tenant_id="tenant-1",
        binding_id="binding-1",
        operation_id="op-lkw",
        run_id="run-lkw",
        page_size=10,
    )
    assert worker.drain_once() == 1
    first = queue.get_result(start)
    assert first is not None
    assert first.status.value == "SUCCEEDED"
    assert first.output is not None
    out1 = sync_worker.VendorKnowledgeSyncWorkerOutput.model_validate(
        json.loads(first.output.decode("utf-8"))
    )
    assert out1.has_more is True
    assert out1.delivery_id is not None
    assert out1.continuation_task_id is not None
    assert worker.drain_once() == 1
    cont = queue.get_result(
        TaskHandle(
            task_id=out1.continuation_task_id,
            provider="document_store",
            tenant_id="tenant-1",
        )
    )
    assert cont is not None
    assert cont.status.value == "SUCCEEDED"
    assert cont.output is not None
    out2 = sync_worker.VendorKnowledgeSyncWorkerOutput.model_validate(
        json.loads(cont.output.decode("utf-8"))
    )
    assert out2.has_more is False
    assert out2.delivery_id is not None
    assert out1.delivery_id != out2.delivery_id
    checkpoint = DocumentStoreKnowledgeSyncCheckpointRepository(store).get(
        tenant_id="tenant-1",
        binding_id="binding-1",
    )
    assert checkpoint is not None
    assert checkpoint.cursor is not None
    assert checkpoint.cursor.value == "cp2"
    start_job = _job_from_handle(queue, start)
    assert "cursor" not in json.loads(encode_vendor_knowledge_sync_job(start_job).decode("utf-8"))
    assert "VendorKnowledgeSyncRuntime" not in type(worker).__name__
