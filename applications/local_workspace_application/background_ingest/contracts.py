# © Artur Czarnecki. All rights reserved.

"""LKW background ingest job payload contract for message_bus enqueue (LKW.4A)."""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Literal

from pydantic import BaseModel, Field, field_validator

LKW_BACKGROUND_INGEST_TASK_NAME = "lkw.background_ingest.v1"
LKW_BACKGROUND_INGEST_SCHEMA_VERSION = "lkw.background_ingest_job.v1"
_IDEMPOTENCY_KEY_PREFIX = "lkw.background_ingest.v1:"


class LkwBackgroundIngestJob(BaseModel):
    schema_version: Literal["lkw.background_ingest_job.v1"] = LKW_BACKGROUND_INGEST_SCHEMA_VERSION
    tenant_id: str = Field(..., min_length=1)
    workspace_id: str = Field(..., min_length=1)
    collection_id: str = Field(..., min_length=1)
    source_paths: tuple[str, ...] = Field(..., min_length=1)
    requested_by: str = "background_ingest"
    run_id: str | None = None
    correlation_id: str | None = None
    reason: str | None = None
    priority: str = "normal"

    @field_validator("source_paths", mode="before")
    @classmethod
    def _normalize_source_paths(cls, value: object) -> tuple[str, ...]:
        if value is None:
            raise ValueError("source_paths must not be empty")
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple)):
            raise ValueError("source_paths must be a sequence")
        normalized = tuple(str(path).strip() for path in value)
        if len(normalized) < 1:
            raise ValueError("source_paths must not be empty")
        if any(not path for path in normalized):
            raise ValueError("source_paths must not contain blank strings")
        return normalized


def encode_background_ingest_job(job: LkwBackgroundIngestJob) -> bytes:
    return json.dumps(
        job.model_dump(mode="json"),
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def decode_background_ingest_job(payload: bytes) -> LkwBackgroundIngestJob:
    raw = json.loads(payload.decode("utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("background ingest job payload must be a JSON object")
    return LkwBackgroundIngestJob.model_validate(raw)


def background_ingest_idempotency_key(job: LkwBackgroundIngestJob) -> str:
    identity = {
        "tenant_id": job.tenant_id,
        "workspace_id": job.workspace_id,
        "collection_id": job.collection_id,
        "source_paths": sorted(job.source_paths),
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"{_IDEMPOTENCY_KEY_PREFIX}{digest[:32]}"


def background_ingest_payload_base64(job: LkwBackgroundIngestJob) -> str:
    return base64.b64encode(encode_background_ingest_job(job)).decode("ascii")
