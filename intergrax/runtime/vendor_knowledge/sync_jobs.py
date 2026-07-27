# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Durable Vendor Knowledge sync job contract and scheduler."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from intergrax.queueing.contracts.task_queue import TaskHandle, TaskQueue, TaskRequest
from intergrax.runtime.vendor_knowledge.sync_models import KnowledgeSyncMode

VENDOR_KNOWLEDGE_SYNC_TASK_NAME = "vendor_knowledge.sync.v1"
VENDOR_KNOWLEDGE_SYNC_JOB_SCHEMA = "vendor_knowledge.sync_job.v1"

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


def _require_non_empty(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be a non-empty string")
    return cleaned


def _require_sha256_hex(value: str, *, field_name: str) -> str:
    cleaned = _require_non_empty(value, field_name=field_name)
    if _SHA256_HEX_RE.fullmatch(cleaned) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")
    return cleaned


class VendorKnowledgeSyncJob(BaseModel):
    """Queue payload for one durable sync page attempt — no provider cursor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["vendor_knowledge.sync_job.v1"] = (
        VENDOR_KNOWLEDGE_SYNC_JOB_SCHEMA
    )
    tenant_id: str
    binding_id: str
    operation_id: str
    mode: KnowledgeSyncMode
    restart: bool
    page_size: int = Field(ge=1, le=1000)
    trigger_delivery_id: str | None = None
    recovery_attempt: int = Field(ge=0)

    @field_validator("tenant_id", "binding_id", "operation_id")
    @classmethod
    def _non_empty_ids(cls, value: str, info: ValidationInfo) -> str:
        field_name = info.field_name or "field"
        return _require_non_empty(value, field_name=field_name)

    @field_validator("trigger_delivery_id")
    @classmethod
    def _optional_delivery(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _require_sha256_hex(value, field_name="trigger_delivery_id")

    @model_validator(mode="after")
    def _mode_restart_rules(self) -> VendorKnowledgeSyncJob:
        if self.mode is KnowledgeSyncMode.INCREMENTAL:
            if self.restart:
                raise ValueError("incremental sync job must set restart=False")
            return self
        if self.mode is KnowledgeSyncMode.RECONCILIATION:
            if self.trigger_delivery_id is None and not self.restart:
                raise ValueError(
                    "first reconciliation sync job must set restart=True"
                )
            if self.trigger_delivery_id is not None and self.restart:
                raise ValueError(
                    "continuation reconciliation sync job must set restart=False"
                )
            return self
        raise ValueError(f"unsupported sync mode: {self.mode!r}")


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
    if job.trigger_delivery_id is None:
        payload: dict[str, object] = {
            "schema_version": "vendor_knowledge.sync_idempotency.v1",
            "tenant_id": job.tenant_id,
            "operation_id": job.operation_id,
            "binding_id": job.binding_id,
            "mode": job.mode.value,
            "phase": "start",
            "restart": job.restart,
            "recovery_attempt": job.recovery_attempt,
        }
    else:
        payload = {
            "schema_version": "vendor_knowledge.sync_idempotency.v1",
            "tenant_id": job.tenant_id,
            "operation_id": job.operation_id,
            "binding_id": job.binding_id,
            "mode": job.mode.value,
            "phase": "continuation",
            "restart": job.restart,
            "trigger_delivery_id": job.trigger_delivery_id,
            "recovery_attempt": job.recovery_attempt,
        }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"{VENDOR_KNOWLEDGE_SYNC_TASK_NAME}:{digest}"


class VendorKnowledgeSyncScheduler:
    """Enqueue first, continuation and recovery sync jobs without provider cursors."""

    def __init__(self, *, task_queue: TaskQueue) -> None:
        self._task_queue = task_queue

    def enqueue_incremental(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        operation_id: str,
        run_id: str,
        page_size: int = 100,
    ) -> TaskHandle:
        job = VendorKnowledgeSyncJob(
            tenant_id=tenant_id,
            binding_id=binding_id,
            operation_id=operation_id,
            mode=KnowledgeSyncMode.INCREMENTAL,
            restart=False,
            page_size=page_size,
            trigger_delivery_id=None,
            recovery_attempt=0,
        )
        return self._enqueue(job=job, run_id=run_id)

    def enqueue_reconciliation(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        operation_id: str,
        run_id: str,
        page_size: int = 100,
    ) -> TaskHandle:
        job = VendorKnowledgeSyncJob(
            tenant_id=tenant_id,
            binding_id=binding_id,
            operation_id=operation_id,
            mode=KnowledgeSyncMode.RECONCILIATION,
            restart=True,
            page_size=page_size,
            trigger_delivery_id=None,
            recovery_attempt=0,
        )
        return self._enqueue(job=job, run_id=run_id)

    def enqueue_continuation(
        self,
        *,
        parent_job: VendorKnowledgeSyncJob,
        run_id: str,
        trigger_delivery_id: str,
    ) -> TaskHandle:
        job = VendorKnowledgeSyncJob(
            tenant_id=parent_job.tenant_id,
            binding_id=parent_job.binding_id,
            operation_id=parent_job.operation_id,
            mode=parent_job.mode,
            restart=False,
            page_size=parent_job.page_size,
            trigger_delivery_id=trigger_delivery_id,
            recovery_attempt=0,
        )
        return self._enqueue(job=job, run_id=run_id)

    def enqueue_recovery(
        self,
        *,
        interrupted_job: VendorKnowledgeSyncJob,
        run_id: str,
    ) -> TaskHandle:
        job = VendorKnowledgeSyncJob(
            tenant_id=interrupted_job.tenant_id,
            binding_id=interrupted_job.binding_id,
            operation_id=interrupted_job.operation_id,
            mode=interrupted_job.mode,
            restart=interrupted_job.restart,
            page_size=interrupted_job.page_size,
            trigger_delivery_id=interrupted_job.trigger_delivery_id,
            recovery_attempt=interrupted_job.recovery_attempt + 1,
        )
        return self._enqueue(job=job, run_id=run_id)

    def _enqueue(self, *, job: VendorKnowledgeSyncJob, run_id: str) -> TaskHandle:
        cleaned_run = _require_non_empty(run_id, field_name="run_id")
        return self._task_queue.enqueue(
            TaskRequest(
                tenant_id=job.tenant_id,
                run_id=cleaned_run,
                task_name=VENDOR_KNOWLEDGE_SYNC_TASK_NAME,
                payload=encode_vendor_knowledge_sync_job(job),
                idempotency_key=vendor_knowledge_sync_idempotency_key(job),
            )
        )
