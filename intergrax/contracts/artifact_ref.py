# © Artur Czarnecki. All rights reserved.

"""Typed artifact references on agent run results (architecture §40.6 · ACP-PROD-6)."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ArtifactSensitivity(StrEnum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    PII = "pii"


class ArtifactProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    created_by_agent_id: str
    created_by_tool_id: str | None = None
    source_side_effect_id: str | None = None


class ArtifactRef(BaseModel):
    """Harness-registered artifact pointer — no secrets in uri query."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["artifact_ref.v1"] = "artifact_ref.v1"
    artifact_id: str
    type: str
    uri: str
    mime_type: str | None = None
    provenance: ArtifactProvenance
    retention_class: str = "default"
    sensitivity: ArtifactSensitivity = ArtifactSensitivity.INTERNAL
    checksum: str | None = None
    size_bytes: int | None = None
    created_at: datetime = Field(default_factory=_utc_now)
    trace_id: str = ""
    run_id: str = ""
    step_index: int | None = None


def artifact_ref_from_payload(
    payload: dict[str, Any],
    *,
    run_id: str,
    trace_id: str,
    agent_id: str,
    step_index: int | None = None,
) -> ArtifactRef:
    """Normalize loose StepOutcome artifact dicts into ``ArtifactRef``."""
    artifact_id = str(payload.get("artifact_id") or payload.get("id") or f"art_{uuid4().hex}")
    artifact_type = str(payload.get("type") or payload.get("kind") or "structured_json")
    uri = str(payload.get("uri") or payload.get("path") or f"memory://{artifact_id}")
    tool_id = payload.get("tool_id")
    return ArtifactRef(
        artifact_id=artifact_id,
        type=artifact_type,
        uri=uri,
        mime_type=payload.get("mime_type"),
        provenance=ArtifactProvenance(
            created_by_agent_id=agent_id,
            created_by_tool_id=str(tool_id) if tool_id else None,
            source_side_effect_id=(
                str(payload["source_side_effect_id"])
                if payload.get("source_side_effect_id")
                else None
            ),
        ),
        retention_class=str(payload.get("retention_class") or "default"),
        sensitivity=ArtifactSensitivity(
            str(payload.get("sensitivity") or ArtifactSensitivity.INTERNAL.value)
        ),
        checksum=payload.get("checksum"),
        size_bytes=payload.get("size_bytes"),
        trace_id=trace_id,
        run_id=run_id,
        step_index=step_index,
    )
