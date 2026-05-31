# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""OpenSearch REST client — search + index API."""

from __future__ import annotations

import time
from typing import Any, Mapping, Optional
from urllib.parse import quote

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.observability_backend import (
    MetricPoint,
    MetricQueryResult,
    MetricSeries,
)
from intergrax.integrations.providers.observability_backend.opensearch.config import OpenSearchIntegrationConfig


class OpenSearchRestClient:
    """OpenSearch ``_search`` and index management client."""

    def __init__(self, config: OpenSearchIntegrationConfig, *, http_client: Any) -> None:
        if not config.base_url:
            raise IntegrationConfigurationError("OpenSearch base_url is required (INTERGRAX_OPENSEARCH_URL)")
        self._config = config
        self._http = http_client

    @property
    def config(self) -> OpenSearchIntegrationConfig:
        return self._config

    def _index_path(self, index: Optional[str] = None) -> str:
        name = index or self._config.index
        return f"/{quote(name, safe='*,.-')}"

    def query_instant(self, promql: str, *, eval_time: Optional[float] = None) -> MetricQueryResult:
        body = {
            "size": 0,
            "query": {"query_string": {"query": promql or "*"}},
            "aggs": {"count": {"value_count": {"field": "_id"}}},
        }
        response = self._http.post(f"{self._index_path()}/_search", json=body)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise IntegrationConfigurationError("Unexpected OpenSearch search response")
        aggregations = payload.get("aggregations") or {}
        count_obj = aggregations.get("count") if isinstance(aggregations, dict) else {}
        value = count_obj.get("value") if isinstance(count_obj, dict) else 0
        ts = float(eval_time if eval_time is not None else time.time())
        return MetricQueryResult(
            result_type="vector",
            series=[MetricSeries(metric={"provider": "opensearch"}, points=[MetricPoint(timestamp=ts, value=float(value or 0))])],
        )

    def query_range(
        self,
        promql: str,
        *,
        start: float,
        end: float,
        step: str = "15s",
    ) -> MetricQueryResult:
        _ = promql, step
        body = {
            "size": 0,
            "query": {"range": {self._config.timestamp_field: {"gte": int(start * 1000), "lte": int(end * 1000)}}},
            "aggs": {
                "timeline": {
                    "date_histogram": {
                        "field": self._config.timestamp_field,
                        "fixed_interval": "1h",
                    }
                }
            },
        }
        response = self._http.post(f"{self._index_path()}/_search", json=body)
        response.raise_for_status()
        payload = response.json()
        points: list[MetricPoint] = []
        aggregations = payload.get("aggregations") if isinstance(payload, dict) else {}
        timeline = aggregations.get("timeline") if isinstance(aggregations, dict) else {}
        buckets = timeline.get("buckets") if isinstance(timeline, dict) else []
        for bucket in buckets or []:
            if not isinstance(bucket, dict):
                continue
            key = bucket.get("key")
            if isinstance(key, (int, float)):
                points.append(MetricPoint(timestamp=float(key) / 1000.0, value=float(bucket.get("doc_count", 0))))
        return MetricQueryResult(
            result_type="matrix",
            series=[MetricSeries(metric={"provider": "opensearch"}, points=points)],
        )

    def index_document(self, *, index: str, document: Mapping[str, Any], doc_id: Optional[str] = None) -> str:
        path = self._index_path(index)
        if doc_id:
            response = self._http.put(f"{path}/_doc/{quote(doc_id, safe='')}", json=dict(document))
        else:
            response = self._http.post(f"{path}/_doc", json=dict(document))
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            return str(payload.get("_id") or doc_id or "")
        return str(doc_id or "")

    def ensure_index(self, index: str) -> bool:
        path = self._index_path(index)
        head = self._http.head(path)
        if head.status_code == 200:
            return False
        response = self._http.put(path, json={"settings": {"number_of_shards": 1}})
        response.raise_for_status()
        return True
