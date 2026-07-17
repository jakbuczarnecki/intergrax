# © Artur Czarnecki. All rights reserved.

"""LKW background ingest job payload contract for message_bus enqueue (LKW.4A)."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

LKW_BACKGROUND_INGEST_TASK_NAME = "lkw.background_ingest.v1"
LKW_BACKGROUND_INGEST_SCHEMA_VERSION = "lkw.background_ingest_job.v1"
_IDEMPOTENCY_KEY_PREFIX = "lkw.background_ingest.v1:"
_CHANGE_TOKEN_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


class LkwBackgroundIngestJob(BaseModel):
    schema_version: Literal["lkw.background_ingest_job.v1"] = (
        LKW_BACKGROUND_INGEST_SCHEMA_VERSION
    )
    tenant_id: str = Field(..., min_length=1)
    workspace_id: str = Field(..., min_length=1)
    collection_id: str = Field(..., min_length=1)
    source_paths: tuple[str, ...] = Field(..., min_length=1)
    requested_by: str = "background_ingest"
    run_id: str | None = None
    correlation_id: str | None = None
    reason: str | None = None
    priority: str = "normal"
    change_token: str | None = None

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

    @field_validator("change_token")
    @classmethod
    def _validate_change_token(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("change_token must be a string")
        if not _CHANGE_TOKEN_PATTERN.fullmatch(value):
            raise ValueError(
                "change_token must match sha256:<64 lowercase hexadecimal characters>"
            )
        return value


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
    identity: dict[str, object] = {
        "tenant_id": job.tenant_id,
        "workspace_id": job.workspace_id,
        "collection_id": job.collection_id,
        "source_paths": sorted(job.source_paths),
    }
    if job.change_token is not None:
        identity["change_token"] = job.change_token
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"{_IDEMPOTENCY_KEY_PREFIX}{digest[:32]}"


def background_ingest_payload_base64(job: LkwBackgroundIngestJob) -> str:
    return base64.b64encode(encode_background_ingest_job(job)).decode("ascii")
