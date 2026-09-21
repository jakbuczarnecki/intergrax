# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Observability backend integration contract (§7.1.2, Phase M.6)."""

from __future__ import annotations

from typing import Optional, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, Field, field_validator

from intergrax.contracts.application_observability_attributes import (
    ObservabilityAttributeValue,
    coerce_observability_attribute_mapping,
)


class MetricPoint(BaseModel):
    timestamp: float
    value: float


class MetricSeries(BaseModel):
    metric: dict[str, str] = Field(default_factory=dict)
    points: Sequence[MetricPoint] = Field(default_factory=list)


class MetricQueryResult(BaseModel):
    result_type: str
    series: Sequence[MetricSeries] = Field(default_factory=list)


class TraceRecord(BaseModel):
    trace_id: str = ""
    name: str = ""
    timestamp: Optional[str] = None
    metadata: dict[str, ObservabilityAttributeValue] = Field(default_factory=dict)

    @field_validator("metadata", mode="before")
    @classmethod
    def _coerce_metadata(cls, value: object) -> dict[str, ObservabilityAttributeValue]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise TypeError("TraceRecord.metadata must be a mapping")
        return coerce_observability_attribute_mapping(value)


class TraceQueryResult(BaseModel):
    traces: Sequence[TraceRecord] = Field(default_factory=list)


@runtime_checkable
class ObservabilityBackend(Protocol):
    """
    Backend-agnostic metrics query facade.

    Implementations: prometheus, elasticsearch, otel, …
    """

    def query_instant(self, promql: str, *, eval_time: Optional[float] = None) -> MetricQueryResult:
        """Run an instant PromQL query (Prometheus ``/api/v1/query``)."""
        ...

    def query_range(
        self,
        promql: str,
        *,
        start: float,
        end: float,
        step: str = "15s",
    ) -> MetricQueryResult:
        """Run a range PromQL query (Prometheus ``/api/v1/query_range``)."""
        ...

    def query_traces(
        self,
        *,
        limit: int = 20,
        name: Optional[str] = None,
    ) -> TraceQueryResult:
        """Query recent traces/spans from the backend (Langfuse, OTEL, etc.)."""
        ...
