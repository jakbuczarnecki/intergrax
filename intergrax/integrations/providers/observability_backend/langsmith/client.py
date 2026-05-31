# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""LangSmith REST client — runs/traces API."""

from __future__ import annotations

import time
from typing import Any, Mapping, Optional

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.observability_backend import (
    MetricPoint,
    MetricQueryResult,
    MetricSeries,
    TraceQueryResult,
    TraceRecord,
)
from intergrax.integrations.providers.observability_backend.langsmith.config import LangSmithIntegrationConfig


def _trace_rows(payload: Any, *, limit: int) -> TraceQueryResult:
    rows = payload if isinstance(payload, list) else (payload.get("runs") if isinstance(payload, dict) else [])
    traces: list[TraceRecord] = []
    for item in list(rows or [])[:limit]:
        if not isinstance(item, dict):
            continue
        traces.append(
            TraceRecord(
                trace_id=str(item.get("id") or item.get("trace_id") or ""),
                name=str(item.get("name") or item.get("run_type") or ""),
                timestamp=str(item.get("start_time") or item.get("created_at") or "") or None,
                metadata={k: v for k, v in item.items() if k not in {"id", "trace_id", "name", "start_time", "created_at"}},
            )
        )
    return TraceQueryResult(traces=traces)


class LangSmithRestClient:
    """LangSmith REST API v1 client for runs and session metrics."""

    def __init__(self, config: LangSmithIntegrationConfig, *, http_client: Any) -> None:
        if not config.api_key:
            raise IntegrationConfigurationError("LangSmith api_key is required (INTERGRAX_LANGSMITH_API_KEY)")
        self._config = config
        self._http = http_client

    @property
    def config(self) -> LangSmithIntegrationConfig:
        return self._config

    def query_instant(self, promql: str, *, eval_time: Optional[float] = None) -> MetricQueryResult:
        _ = promql
        params: dict[str, object] = {}
        if self._config.project:
            params["name"] = self._config.project
        response = self._http.get("/api/v1/sessions", params=params)
        response.raise_for_status()
        payload = response.json()
        count = len(payload) if isinstance(payload, list) else int((payload or {}).get("total", 0) or 0)
        ts = float(eval_time if eval_time is not None else time.time())
        return MetricQueryResult(
            result_type="vector",
            series=[MetricSeries(metric={"provider": "langsmith"}, points=[MetricPoint(timestamp=ts, value=float(count))])],
        )

    def query_range(
        self,
        promql: str,
        *,
        start: float,
        end: float,
        step: str = "15s",
    ) -> MetricQueryResult:
        _ = promql, end, step
        instant = self.query_instant(promql, eval_time=start)
        return instant

    def query_traces(self, *, limit: int = 20, name: Optional[str] = None) -> TraceQueryResult:
        params: dict[str, object] = {"limit": max(1, int(limit))}
        if name:
            params["filter"] = f'eq(name, "{name}")'
        elif self._config.project:
            params["project_name"] = self._config.project
        response = self._http.get("/api/v1/runs/query", params=params)
        if response.status_code == 404:
            response = self._http.get("/api/v1/runs", params={"limit": limit})
        response.raise_for_status()
        return _trace_rows(response.json(), limit=limit)

    def get_run(self, run_id: str) -> Mapping[str, Any]:
        response = self._http.get(f"/api/v1/runs/{run_id}")
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise IntegrationConfigurationError("Unexpected LangSmith run response")
        return payload
