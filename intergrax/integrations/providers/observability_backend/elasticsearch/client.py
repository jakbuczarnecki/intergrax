# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Elasticsearch search client — HTTP client injected from ``opens.py`` only."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional
from urllib.parse import quote

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.observability_backend import (
    MetricPoint,
    MetricQueryResult,
    MetricSeries,
)
from intergrax.integrations.providers.observability_backend.elasticsearch.config import ElasticsearchIntegrationConfig

_ELASTICSEARCH_OBSERVABILITY_PROVIDER_ID = "elasticsearch"
_RETRIABLE_HTTP_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
_NON_RETRIABLE_HTTP_STATUS_CODES = frozenset({400, 401, 403, 404})


@dataclass(frozen=True)
class ElasticsearchDeliveryErrorDetail:
    """Safe provider-owned diagnostics for Elasticsearch delivery failures."""

    provider_id: str
    operation: str
    index: str
    status_code: int | None
    reason: str
    retriable: bool

    def __str__(self) -> str:
        parts = [
            f"provider_id={self.provider_id}",
            f"operation={self.operation}",
            f"index={self.index}",
            f"status_code={self.status_code}",
            f"reason={self.reason}",
            f"retriable={self.retriable}",
        ]
        return " ".join(parts)


class ElasticsearchDeliveryError(RuntimeError):
    """Classified Elasticsearch/OpenSearch observability delivery failure."""

    def __init__(self, detail: ElasticsearchDeliveryErrorDetail) -> None:
        self.detail = detail
        super().__init__(str(detail))

    def __str__(self) -> str:
        return f"Elasticsearch delivery failed: {self.detail}"


def _exception_module_name(exc: BaseException) -> str:
    return type(exc).__module__


def _exception_type_name(exc: BaseException) -> str:
    return type(exc).__name__


def _is_timeout_error(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    if _exception_module_name(exc).startswith("httpx"):
        return "Timeout" in _exception_type_name(exc)
    return False


def _is_connection_error(exc: BaseException) -> bool:
    name = _exception_type_name(exc)
    if name in {"ConnectError", "ConnectTimeout", "NetworkError", "RemoteProtocolError"}:
        return True
    return _exception_module_name(exc).startswith("httpx") and "Connect" in name


def _http_status_code(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)
    if response is None:
        return None
    status_code = getattr(response, "status_code", None)
    return int(status_code) if isinstance(status_code, int) else None


def _delivery_reason(exc: BaseException, *, status_code: int | None) -> str:
    if isinstance(exc, IntegrationConfigurationError):
        return "invalid_config"
    if isinstance(exc, (TypeError, ValueError)):
        return "invalid_payload_shape"
    if status_code is not None:
        return f"http_status_{status_code}"
    if _is_timeout_error(exc):
        return "timeout"
    if _is_connection_error(exc):
        return "connection_error"
    return "delivery_failed"


def _is_retriable_delivery_failure(exc: BaseException, *, status_code: int | None) -> bool:
    if status_code in _RETRIABLE_HTTP_STATUS_CODES:
        return True
    if status_code in _NON_RETRIABLE_HTTP_STATUS_CODES:
        return False
    if _is_timeout_error(exc) or _is_connection_error(exc):
        return True
    if isinstance(exc, IntegrationConfigurationError):
        return False
    if isinstance(exc, (TypeError, ValueError)):
        return False
    if status_code is not None and status_code >= 500:
        return True
    return False


def classify_elasticsearch_delivery_error(
    exc: BaseException,
    *,
    operation: str,
    index: str,
    provider_id: str = _ELASTICSEARCH_OBSERVABILITY_PROVIDER_ID,
) -> ElasticsearchDeliveryError:
    """Map a transport/client failure to safe provider-owned delivery diagnostics."""
    if isinstance(exc, ElasticsearchDeliveryError):
        return exc
    status_code = _http_status_code(exc)
    detail = ElasticsearchDeliveryErrorDetail(
        provider_id=provider_id,
        operation=operation,
        index=index,
        status_code=status_code,
        reason=_delivery_reason(exc, status_code=status_code),
        retriable=_is_retriable_delivery_failure(exc, status_code=status_code),
    )
    return ElasticsearchDeliveryError(detail)


def _parse_instant(payload: Mapping[str, Any], *, eval_time: Optional[float]) -> MetricQueryResult:
    aggregations = payload.get("aggregations")
    if not isinstance(aggregations, dict):
        raise IntegrationConfigurationError("Unexpected Elasticsearch instant search response")
    count_obj = aggregations.get("count")
    value = count_obj.get("value") if isinstance(count_obj, dict) else 0
    timestamp = float(eval_time if eval_time is not None else time.time())
    point = MetricPoint(timestamp=timestamp, value=float(value or 0))
    return MetricQueryResult(
        result_type="vector",
        series=[MetricSeries(metric={}, points=[point])],
    )


def _parse_range(payload: Mapping[str, Any]) -> MetricQueryResult:
    aggregations = payload.get("aggregations")
    if not isinstance(aggregations, dict):
        raise IntegrationConfigurationError("Unexpected Elasticsearch range search response")
    timeline = aggregations.get("timeline")
    buckets = timeline.get("buckets") if isinstance(timeline, dict) else None
    if not isinstance(buckets, list):
        return MetricQueryResult(result_type="matrix", series=[])
    points: list[MetricPoint] = []
    for bucket in buckets:
        if not isinstance(bucket, dict):
            continue
        key = bucket.get("key")
        doc_count = bucket.get("doc_count", 0)
        if isinstance(key, (int, float)):
            timestamp = float(key) / 1000.0
        else:
            continue
        points.append(MetricPoint(timestamp=timestamp, value=float(doc_count)))
    return MetricQueryResult(
        result_type="matrix",
        series=[MetricSeries(metric={}, points=points)],
    )


class ElasticsearchRestClient:
    """Minimal Elasticsearch ``_search`` client — sync HTTP via injected client."""

    def __init__(
        self,
        config: ElasticsearchIntegrationConfig,
        *,
        http_client: Any,
    ) -> None:
        if not config.base_url:
            raise IntegrationConfigurationError(
                "Elasticsearch base_url is required (INTERGRAX_ELASTICSEARCH_URL)"
            )
        self._config = config
        self._http_client = http_client

    @property
    def config(self) -> ElasticsearchIntegrationConfig:
        return self._config

    def query_instant(self, promql: str, *, eval_time: Optional[float] = None) -> MetricQueryResult:
        filters: list[dict[str, Any]] = []
        if eval_time is not None:
            epoch_ms = int(eval_time * 1000)
            filters.append(
                {
                    "range": {
                        self._config.timestamp_field: {
                            "gte": epoch_ms,
                            "lte": epoch_ms,
                            "format": "epoch_millis",
                        }
                    }
                }
            )
        body = self._search_body(promql, filters=filters, aggs={"count": {"value_count": {"field": "_id"}}})
        payload = self._search(body)
        return _parse_instant(payload, eval_time=eval_time)

    def query_range(
        self,
        promql: str,
        *,
        start: float,
        end: float,
        step: str = "15s",
    ) -> MetricQueryResult:
        filters = [
            {
                "range": {
                    self._config.timestamp_field: {
                        "gte": int(start * 1000),
                        "lte": int(end * 1000),
                        "format": "epoch_millis",
                    }
                }
            }
        ]
        body = self._search_body(
            promql,
            filters=filters,
            aggs={
                "timeline": {
                    "date_histogram": {
                        "field": self._config.timestamp_field,
                        "fixed_interval": step,
                        "min_doc_count": 0,
                        "extended_bounds": {
                            "min": int(start * 1000),
                            "max": int(end * 1000),
                        },
                    }
                }
            },
        )
        payload = self._search(body)
        return _parse_range(payload)

    def _search_body(
        self,
        query: str,
        *,
        filters: list[dict[str, Any]],
        aggs: dict[str, Any],
    ) -> dict[str, Any]:
        bool_query: dict[str, Any] = {
            "must": [{"query_string": {"query": query or "*"}}],
        }
        if filters:
            bool_query["filter"] = filters
        return {
            "size": 0,
            "query": {"bool": bool_query},
            "aggs": aggs,
        }

    def _index_path(self, index: Optional[str] = None) -> str:
        name = index or self._config.index
        return f"/{quote(name, safe='*,.-')}"

    def index_document(
        self,
        *,
        index: str,
        document: Mapping[str, Any],
        doc_id: Optional[str] = None,
    ) -> str:
        target_index = index or self._config.index
        try:
            path = self._index_path(index)
            if doc_id:
                response = self._http_client.put(
                    f"{path}/_doc/{quote(doc_id, safe='')}",
                    json=dict(document),
                )
            else:
                response = self._http_client.post(f"{path}/_doc", json=dict(document))
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict):
                return str(payload.get("_id") or doc_id or "")
            return str(doc_id or "")
        except ElasticsearchDeliveryError:
            raise
        except Exception as exc:
            raise classify_elasticsearch_delivery_error(
                exc,
                operation="index_document",
                index=target_index,
            ) from exc

    def _search(self, body: dict[str, Any]) -> dict[str, Any]:
        index = quote(self._config.index, safe="*,.-")
        response = self._http_client.post(f"/{index}/_search", json=body)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise IntegrationConfigurationError("Unexpected Elasticsearch search response")
        return payload
