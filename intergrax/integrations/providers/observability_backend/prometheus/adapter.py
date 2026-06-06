# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Prometheus observability adapter — ``ObservabilityBackend`` facade (no HTTP here)."""

from __future__ import annotations

from typing import Optional

from intergrax.integrations.contracts.base import HealthStatus
from intergrax.integrations.contracts.observability_backend import MetricQueryResult, TraceQueryResult
from intergrax.integrations.providers.observability_backend.prometheus.client import PrometheusRestClient


class PrometheusObservabilityBackend:
    """
    Catalog facade over ``PrometheusRestClient``.

    Instantiate via ``create_prometheus_observability_backend()`` — not from agent code.
    """

    def __init__(self, client: PrometheusRestClient) -> None:
        self._client = client

    @property
    def rest_client(self) -> PrometheusRestClient:
        return self._client

    def query_instant(self, promql: str, *, eval_time: Optional[float] = None) -> MetricQueryResult:
        return self._client.query_instant(promql, eval_time=eval_time)

    def query_range(
        self,
        promql: str,
        *,
        start: float,
        end: float,
        step: str = "15s",
    ) -> MetricQueryResult:
        return self._client.query_range(promql, start=start, end=end, step=step)

    def query_traces(self, *, limit: int = 20, name: Optional[str] = None) -> TraceQueryResult:
        _ = limit, name
        return TraceQueryResult()

    def health(self) -> HealthStatus:
        return HealthStatus(
            slug="prometheus",
            healthy=bool(self._client.health()),
            detail="prometheus ready probe",
        )
