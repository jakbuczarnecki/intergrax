# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Neutral runtime-event export source models (OBS-EXPORT-2)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RuntimeEventExportSource(BaseModel):
    """Typed runtime-event source for deferred lifecycle wiring (OBS-EXPORT-2)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["runtime_event_export_source.v1"] = (
        "runtime_event_export_source.v1"
    )
    event_id: str
    run_id: str
    task_id: str
    attempt_id: str = ""
    execution_id: str = ""
    event_type: str
    agent_id: str = ""
    tenant_id: str = ""
    correlation_id: str = ""
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    execution_phase: str = ""
    parent_event_id: str = ""
    w3c_traceparent: str = ""
    w3c_tracestate: str = ""
    safe_payload: dict[str, str | int] = Field(default_factory=dict)
