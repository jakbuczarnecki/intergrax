# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Braintrust REST client — eval logs and metrics."""

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
from intergrax.integrations.providers.observability_backend.braintrust.config import BraintrustIntegrationConfig


class BraintrustRestClient:
    """Braintrust project logs and experiment metrics client."""

    def __init__(self, config: BraintrustIntegrationConfig, *, http_client: Any) -> None:
        if not config.api_key:
            raise IntegrationConfigurationError("Braintrust api_key is required (INTERGRAX_BRAINTRUST_API_KEY)")
        self._config = config
        self._http = http_client

    @property
    def config(self) -> BraintrustIntegrationConfig:
        return self._config

    def query_instant(self, promql: str, *, eval_time: Optional[float] = None) -> MetricQueryResult:
        _ = promql
        path = "/v1/experiment/metrics"
        if self._config.project:
            path = f"/v1/project/{self._config.project}/metrics"
        response = self._http.get(path)
        response.raise_for_status()
        payload = response.json()
        value = float(payload.get("count") or len(payload.get("data") or [])) if isinstance(payload, dict) else 0.0
        ts = float(eval_time if eval_time is not None else time.time())
        return MetricQueryResult(
            result_type="vector",
            series=[MetricSeries(metric={"provider": "braintrust"}, points=[MetricPoint(timestamp=ts, value=value)])],
        )

    def query_range(
        self,
        promql: str,
        *,
        start: float,
        end: float,
        step: str = "15s",
    ) -> MetricQueryResult:
        return self.query_instant(promql, eval_time=start)

    def query_traces(self, *, limit: int = 20, name: Optional[str] = None) -> TraceQueryResult:
        params: dict[str, object] = {"limit": max(1, int(limit))}
        if name:
            params["name"] = name
        response = self._http.get("/v1/project/logs", params=params)
        response.raise_for_status()
        payload = response.json()
        rows = payload if isinstance(payload, list) else (payload.get("data") if isinstance(payload, dict) else [])
        traces: list[TraceRecord] = []
        for item in list(rows or [])[:limit]:
            if not isinstance(item, dict):
                continue
            traces.append(
                TraceRecord(
                    trace_id=str(item.get("id") or ""),
                    name=str(item.get("name") or item.get("event") or ""),
                    timestamp=str(item.get("created") or "") or None,
                    metadata=dict(item),
                )
            )
        return TraceQueryResult(traces=traces)

    def log_eval(
        self,
        *,
        name: str,
        score: float,
        metadata: Optional[Mapping[str, Any]] = None,
        project: Optional[str] = None,
    ) -> str:
        project_id = project or self._config.project
        payload: dict[str, object] = {
            "events": [
                {
                    "input": {"name": name},
                    "scores": {"eval": float(score)},
                    "metadata": dict(metadata or {}),
                }
            ]
        }
        path = "/v1/project/logs"
        if project_id:
            path = f"/v1/project/{project_id}/logs"
        response = self._http.post(path, json=payload)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            return str(data.get("id") or data.get("row_ids", [""])[0] if data.get("row_ids") else "")
        return ""
