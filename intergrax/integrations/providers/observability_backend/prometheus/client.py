# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Prometheus HTTP API client — HTTP client injected from ``opens.py`` only."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.observability_backend import (
    MetricPoint,
    MetricQueryResult,
    MetricSeries,
)
from intergrax.integrations.providers.observability_backend.prometheus.config import PrometheusIntegrationConfig


def _parse_point(raw: object) -> MetricPoint | None:
    if not isinstance(raw, Sequence) or len(raw) < 2:
        return None
    try:
        timestamp = float(raw[0])
        value = float(raw[1])
    except (TypeError, ValueError):
        return None
    return MetricPoint(timestamp=timestamp, value=value)


def _metric_labels(raw: object) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    return {str(key): str(value) for key, value in raw.items()}


def _parse_query_data(data: Mapping[str, Any]) -> MetricQueryResult:
    result_type = str(data.get("resultType") or "unknown")
    raw_result = data.get("result")

    if result_type == "scalar":
        point = _parse_point(raw_result)
        series = [MetricSeries(metric={}, points=[point] if point else [])] if point else []
        return MetricQueryResult(result_type=result_type, series=series)

    if not isinstance(raw_result, list):
        return MetricQueryResult(result_type=result_type, series=[])

    series_list: list[MetricSeries] = []
    for item in raw_result:
        if not isinstance(item, dict):
            continue
        metric = _metric_labels(item.get("metric"))
        if result_type == "matrix":
            values = item.get("values")
            points: list[MetricPoint] = []
            if isinstance(values, list):
                for value in values:
                    parsed = _parse_point(value)
                    if parsed is not None:
                        points.append(parsed)
        else:
            point = _parse_point(item.get("value"))
            points = [point] if point is not None else []
        series_list.append(MetricSeries(metric=metric, points=points))

    return MetricQueryResult(result_type=result_type, series=series_list)


class PrometheusRestClient:
    """Minimal Prometheus query API client — sync HTTP via injected client."""

    def __init__(
        self,
        config: PrometheusIntegrationConfig,
        *,
        http_client: Any,
    ) -> None:
        if not config.base_url:
            raise IntegrationConfigurationError(
                "Prometheus base_url is required (INTERGRAX_PROMETHEUS_BASE_URL)"
            )
        self._config = config
        self._http_client = http_client

    @property
    def config(self) -> PrometheusIntegrationConfig:
        return self._config

    def query_instant(self, promql: str, *, eval_time: Optional[float] = None) -> MetricQueryResult:
        params: dict[str, str | float] = {"query": promql}
        if eval_time is not None:
            params["time"] = eval_time
        return self._query("/api/v1/query", params=params)

    def query_range(
        self,
        promql: str,
        *,
        start: float,
        end: float,
        step: str = "15s",
    ) -> MetricQueryResult:
        params: dict[str, str | float] = {
            "query": promql,
            "start": start,
            "end": end,
            "step": step,
        }
        return self._query("/api/v1/query_range", params=params)

    def _query(self, path: str, *, params: dict[str, str | float]) -> MetricQueryResult:
        response = self._http_client.get(path, params=params)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise IntegrationConfigurationError("Unexpected Prometheus query response")
        if payload.get("status") != "success":
            error_type = payload.get("errorType", "error")
            error = payload.get("error", "Prometheus query failed")
            raise IntegrationConfigurationError(f"Prometheus {error_type}: {error}")
        data = payload.get("data")
        if not isinstance(data, dict):
            raise IntegrationConfigurationError("Unexpected Prometheus query data")
        return _parse_query_data(data)
