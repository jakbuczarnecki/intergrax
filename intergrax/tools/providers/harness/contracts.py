# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class HarnessGetRunInput(BaseModel):
    run_id: str = Field(..., min_length=1)
    tenant_id: str = Field(..., min_length=1)


class HarnessRunMetadataOutput(BaseModel):
    run_id: str
    session_id: str = ""
    user_id: str = ""
    tenant_id: str = ""
    started_at_utc: str = ""
    duration_ms: int = 0
    llm_usage: dict[str, Any] = Field(default_factory=dict)
    error_type: str = ""
    error_message: str = ""


class HarnessGetRunOutput(BaseModel):
    metadata: HarnessRunMetadataOutput
    events: list[dict[str, Any]] = Field(default_factory=list)
    event_count: int = 0


class HarnessListRunsInput(BaseModel):
    tenant_id: str = Field(..., min_length=1)
    limit: int = Field(default=50, ge=1, le=200)


class HarnessRunSummaryOutput(BaseModel):
    run_id: str
    tenant_id: str = ""
    user_id: str = ""
    session_id: str = ""
    started_at_utc: str = ""
    duration_ms: int = 0
    event_count: int = 0


class HarnessListRunsOutput(BaseModel):
    runs: list[HarnessRunSummaryOutput] = Field(default_factory=list)
    total: int = 0


class HarnessGetRunCostInput(BaseModel):
    run_id: str = Field(..., min_length=1)
    tenant_id: str = Field(..., min_length=1)


class HarnessGetRunCostOutput(BaseModel):
    run_id: str
    tenant_id: str
    duration_ms: int = 0
    llm_usage: dict[str, Any] = Field(default_factory=dict)


class HarnessGetRunEventsInput(BaseModel):
    run_id: str = Field(..., min_length=1)
    tenant_id: str = Field(..., min_length=1)
    step: str = ""
    level: str = ""
    limit: int = Field(default=100, ge=1, le=500)


class HarnessRunEventOutput(BaseModel):
    event_id: str = ""
    step: str = ""
    level: str = ""
    message: str = ""
    ts_utc: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)


class HarnessGetRunEventsOutput(BaseModel):
    run_id: str
    events: list[HarnessRunEventOutput] = Field(default_factory=list)
    total: int = 0
