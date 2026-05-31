# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class MetricsQueryInstantInput(BaseModel):
    query: str = Field(..., min_length=1, description="PromQL instant query.")
    eval_time: Optional[float] = Field(default=None, description="Optional evaluation timestamp (epoch seconds).")


class MetricPointOutput(BaseModel):
    timestamp: float
    value: float


class MetricSeriesOutput(BaseModel):
    metric: dict[str, str] = Field(default_factory=dict)
    points: list[MetricPointOutput] = Field(default_factory=list)


class MetricsQueryInstantOutput(BaseModel):
    result_type: str = ""
    series: list[MetricSeriesOutput] = Field(default_factory=list)


class LogsSearchInput(BaseModel):
    query: str = Field(..., min_length=1, description="Log search query (Lucene query_string for Elasticsearch).")
    limit: int = Field(default=20, ge=1, le=100)


class LogHitOutput(BaseModel):
    id: str = ""
    message: str = ""
    timestamp: Optional[str] = None
    source: dict[str, Any] = Field(default_factory=dict)


class LogsSearchOutput(BaseModel):
    hits: list[LogHitOutput] = Field(default_factory=list)
    total: int = 0
    context_text: str = ""


class TracesQueryInput(BaseModel):
    limit: int = Field(default=20, ge=1, le=100)
    name: Optional[str] = Field(default=None, description="Optional trace name filter.")


class TraceRecordOutput(BaseModel):
    trace_id: str = ""
    name: str = ""
    timestamp: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TracesQueryOutput(BaseModel):
    traces: list[TraceRecordOutput] = Field(default_factory=list)
    total: int = 0


class ErrorsCaptureInput(BaseModel):
    message: str = Field(..., min_length=1, description="Error or diagnostic message to report.")
    level: str = Field(default="error", description="Severity: debug, info, warning, error, fatal.")
    tags: dict[str, str] = Field(default_factory=dict, description="Optional tags (tenant_id, run_id, …).")


class ErrorsCaptureOutput(BaseModel):
    event_id: str = ""
    provider: str = ""
