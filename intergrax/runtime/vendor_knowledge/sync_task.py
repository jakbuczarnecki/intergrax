# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Queue payload, dispatcher and worker handler for Vendor Knowledge sync."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Callable
from typing import Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
)

from intergrax.queueing.contracts.task_queue import TaskHandle, TaskRequest
from intergrax.queueing.providers.document_store import DocumentStoreTaskQueue
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.runtime.vendor_knowledge.errors import VendorKnowledgeError
from intergrax.runtime.vendor_knowledge.sync_coordinator import VendorKnowledgeSyncCoordinator
from intergrax.runtime.vendor_knowledge.sync_models import (
    KnowledgeSyncMode,
    KnowledgeSyncRunResult,
    KnowledgeSyncRunStatus,
)
from intergrax.tools.execution_models import ToolExecutionResult

logger = logging.getLogger(__name__)

VENDOR_KNOWLEDGE_SYNC_TASK_NAME = "vendor_knowledge.sync.v1"
VENDOR_KNOWLEDGE_SYNC_JOB_SCHEMA = "vendor_knowledge.sync_job.v1"
_WORKER_OUTPUT_SCHEMA = "vendor_knowledge.sync_worker.v1"

CoordinatorFactory = Callable[[str, str], VendorKnowledgeSyncCoordinator]


def _require_non_empty(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be a non-empty string")
    return cleaned


def owner_id_for_sync_run(run_id: str) -> str:
    """Stable lease owner identity derived from queue run identity."""
    cleaned = _require_non_empty(run_id, field_name="run_id")
    return f"vendor_knowledge.sync:{cleaned}"


class VendorKnowledgeSyncJob(BaseModel):
    """Queue payload for one bounded sync page — identity and safe params only."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["vendor_knowledge.sync_job.v1"] = (
        VENDOR_KNOWLEDGE_SYNC_JOB_SCHEMA
    )
    tenant_id: str
    run_id: str
    binding_id: str
    mode: KnowledgeSyncMode
    page_size: int = Field(ge=1, le=1000)
    restart: bool = False

    @field_validator("tenant_id", "run_id", "binding_id")
    @classmethod
    def _non_empty_ids(cls, value: str, info: ValidationInfo) -> str:
        field_name = info.field_name or "field"
        return _require_non_empty(value, field_name=field_name)


def encode_vendor_knowledge_sync_job(job: VendorKnowledgeSyncJob) -> bytes:
    return json.dumps(
        job.model_dump(mode="json"),
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def decode_vendor_knowledge_sync_job(payload: bytes) -> VendorKnowledgeSyncJob:
    try:
        raw = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("vendor knowledge sync job payload is invalid") from exc
    if not isinstance(raw, dict):
        raise ValueError("vendor knowledge sync job must be a JSON object")
    return VendorKnowledgeSyncJob.model_validate(raw)


def vendor_knowledge_sync_idempotency_key(job: VendorKnowledgeSyncJob) -> str:
    canonical = json.dumps(
        {
            "schema_version": job.schema_version,
            "tenant_id": job.tenant_id,
            "run_id": job.run_id,
            "binding_id": job.binding_id,
            "mode": job.mode.value,
            "page_size": job.page_size,
            "restart": job.restart,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"vendor-knowledge-sync:v1:{digest}"


class VendorKnowledgeSyncDispatcher:
    """Enqueue one Vendor Knowledge sync page through DocumentStoreTaskQueue."""

    def __init__(self, task_queue: DocumentStoreTaskQueue) -> None:
        self._task_queue = task_queue

    def enqueue(self, job: VendorKnowledgeSyncJob) -> TaskHandle:
        return self._task_queue.enqueue(
            TaskRequest(
                tenant_id=job.tenant_id,
                run_id=job.run_id,
                task_name=VENDOR_KNOWLEDGE_SYNC_TASK_NAME,
                payload=encode_vendor_knowledge_sync_job(job),
                idempotency_key=vendor_knowledge_sync_idempotency_key(job),
            )
        )


class VendorKnowledgeSyncWorkerOutput(BaseModel):
    """Safe worker output — no cursors, tokens, secrets or provider payloads."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = _WORKER_OUTPUT_SCHEMA
    run_id: str
    binding_id: str
    status: KnowledgeSyncRunStatus
    mode: KnowledgeSyncMode
    delivery_id: str | None = None
    changes_count: int = Field(ge=0)
    active_count: int = Field(ge=0)
    tombstone_count: int = Field(ge=0)
    checkpoint_advanced: bool
    has_more: bool
    retryable: bool


def _run_coordinator(
    coordinator: VendorKnowledgeSyncCoordinator,
    job: VendorKnowledgeSyncJob,
) -> KnowledgeSyncRunResult:
    if job.mode is KnowledgeSyncMode.INCREMENTAL:
        coro = coordinator.sync_once(
            binding_id=job.binding_id,
            page_size=job.page_size,
        )
    elif job.mode is KnowledgeSyncMode.RECONCILIATION:
        coro = coordinator.reconcile_once(
            binding_id=job.binding_id,
            page_size=job.page_size,
            restart=job.restart,
        )
    else:
        raise ValueError(f"unsupported sync mode: {job.mode!r}")
    return asyncio.run(coro)


def _output_from_result(
    *,
    run_id: str,
    result: KnowledgeSyncRunResult,
) -> VendorKnowledgeSyncWorkerOutput:
    return VendorKnowledgeSyncWorkerOutput(
        run_id=run_id,
        binding_id=result.binding_id,
        status=result.status,
        mode=result.mode,
        delivery_id=result.delivery_id,
        changes_count=result.changes_count,
        active_count=result.active_count,
        tombstone_count=result.tombstone_count,
        checkpoint_advanced=result.checkpoint_advanced,
        has_more=result.has_more,
        retryable=result.retryable,
    )


def make_vendor_knowledge_sync_handler(
    coordinator_factory: CoordinatorFactory,
):
    """Build a TaskExecutionRegistry-compatible sync handler."""

    def handler(
        *,
        tenant_id: str,
        run_id: str,
        payload: bytes,
        idempotency_key: Optional[str] = None,
    ) -> ToolExecutionResult[VendorKnowledgeSyncWorkerOutput]:
        _ = idempotency_key
        try:
            job = decode_vendor_knowledge_sync_job(payload)
        except (ValueError, ValidationError, TypeError):
            return ToolExecutionResult.fail(
                "vendor_knowledge_sync_invalid_job",
                "vendor knowledge sync job payload is invalid",
            )
        if job.tenant_id != tenant_id:
            return ToolExecutionResult.fail(
                "vendor_knowledge_sync_tenant_mismatch",
                "vendor knowledge sync tenant mismatch",
            )
        if job.run_id != run_id:
            return ToolExecutionResult.fail(
                "vendor_knowledge_sync_run_mismatch",
                "vendor knowledge sync run mismatch",
            )

        owner_id = owner_id_for_sync_run(run_id)
        try:
            coordinator = coordinator_factory(tenant_id, owner_id)
            result = _run_coordinator(coordinator, job)
        except VendorKnowledgeError as exc:
            return ToolExecutionResult.fail(
                f"vendor_knowledge_sync_{exc.code.value}",
                exc.safe_message,
            )
        except Exception as exc:
            logger.error(
                "task_name=%s exception_class=%s",
                VENDOR_KNOWLEDGE_SYNC_TASK_NAME,
                type(exc).__name__,
            )
            return ToolExecutionResult.fail(
                "vendor_knowledge_sync_failed",
                "vendor knowledge sync failed",
            )

        return ToolExecutionResult.ok(
            _output_from_result(run_id=run_id, result=result)
        )

    return handler


def register_vendor_knowledge_sync_handler(
    registry: TaskExecutionRegistry,
    coordinator_factory: CoordinatorFactory,
) -> None:
    """Register exactly one handler under vendor_knowledge.sync.v1."""
    registry.register(
        VENDOR_KNOWLEDGE_SYNC_TASK_NAME,
        make_vendor_knowledge_sync_handler(coordinator_factory),
    )
