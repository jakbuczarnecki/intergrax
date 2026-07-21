# © Artur Czarnecki. All rights reserved.

"""Managed workspace sync job contract for durable MessageBus enqueue."""

from __future__ import annotations

import base64
import json
from typing import Literal

from pydantic import BaseModel, Field

LKW_MANAGED_WORKSPACE_SYNC_TASK_NAME = "lkw.managed_workspace_sync.v1"
LKW_MANAGED_WORKSPACE_SYNC_SCHEMA = "lkw.managed_workspace_sync_job.v1"


class ManagedWorkspaceSyncJob(BaseModel):
    """Queue payload — identities only; worker reloads source from persistence."""

    schema_version: Literal["lkw.managed_workspace_sync_job.v1"] = (
        LKW_MANAGED_WORKSPACE_SYNC_SCHEMA
    )
    tenant_id: str = Field(..., min_length=1)
    workspace_id: str = Field(..., min_length=1)
    source_id: str = Field(..., min_length=1)
    operation_id: str = Field(..., min_length=1)
    operation_type: Literal["source_sync"] = "source_sync"


def encode_managed_workspace_sync_job(job: ManagedWorkspaceSyncJob) -> bytes:
    return json.dumps(
        job.model_dump(mode="json"),
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def decode_managed_workspace_sync_job(payload: bytes) -> ManagedWorkspaceSyncJob:
    raw = json.loads(payload.decode("utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("managed_workspace_sync_job must be a JSON object")
    return ManagedWorkspaceSyncJob.model_validate(raw)


def managed_workspace_sync_payload_base64(job: ManagedWorkspaceSyncJob) -> str:
    return base64.b64encode(encode_managed_workspace_sync_job(job)).decode("ascii")


def managed_workspace_sync_idempotency_key(job: ManagedWorkspaceSyncJob) -> str:
    return f"{LKW_MANAGED_WORKSPACE_SYNC_TASK_NAME}:{job.operation_id}"
