# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Any

from intergrax.tools.providers.observability.contracts import (
    LogHitOutput,
    LogsSearchInput,
    LogsSearchOutput,
    MetricPointOutput,
    MetricSeriesOutput,
    MetricsQueryInstantInput,
    MetricsQueryInstantOutput,
)
from intergrax.tools.registry.wiring import ToolWiringContext

METRICS_QUERY_INSTANT_TOOL_ID = "metrics.query_instant"
LOGS_SEARCH_TOOL_ID = "logs.search"


def metrics_query_instant(
    ctx: ToolWiringContext,
    params: MetricsQueryInstantInput,
) -> MetricsQueryInstantOutput:
    backend = ctx.observability_backend
    if backend is None:
        raise RuntimeError("observability_backend_not_configured")

    result = backend.query_instant(params.query, eval_time=params.eval_time)
    series = [
        MetricSeriesOutput(
            metric=dict(item.metric or {}),
            points=[
                MetricPointOutput(timestamp=point.timestamp, value=point.value)
                for point in item.points
            ],
        )
        for item in result.series
    ]
    return MetricsQueryInstantOutput(result_type=result.result_type, series=series)


def logs_search(ctx: ToolWiringContext, params: LogsSearchInput) -> LogsSearchOutput:
    backend = ctx.observability_backend
    if backend is None:
        raise RuntimeError("observability_backend_not_configured")

    rest_client = getattr(backend, "rest_client", None)
    if rest_client is None:
        raise RuntimeError("observability_backend_does_not_support_log_search")

    payload = _search_log_documents(rest_client, params.query, params.limit)
    hits = _parse_log_hits(payload, limit=params.limit)
    context_text = "\n".join(hit.message for hit in hits if hit.message).strip()
    total = _extract_total_hits(payload)
    return LogsSearchOutput(hits=hits, total=total, context_text=context_text)


def _search_log_documents(rest_client: Any, query: str, limit: int) -> dict[str, Any]:
    index = rest_client.config.index
    body = {
        "size": limit,
        "query": {"query_string": {"query": query or "*"}},
        "sort": [{rest_client.config.timestamp_field: {"order": "desc"}}],
    }
    from urllib.parse import quote

    response = rest_client._http_client.post(f"/{quote(index, safe='*,.-')}/_search", json=body)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("invalid_elasticsearch_response")
    return payload


def _extract_total_hits(payload: dict[str, Any]) -> int:
    hits_obj = payload.get("hits")
    if not isinstance(hits_obj, dict):
        return 0
    total = hits_obj.get("total")
    if isinstance(total, dict):
        return int(total.get("value", 0) or 0)
    if isinstance(total, int):
        return total
    return len(hits_obj.get("hits") or [])


def _parse_log_hits(payload: dict[str, Any], *, limit: int) -> list[LogHitOutput]:
    hits_obj = payload.get("hits")
    if not isinstance(hits_obj, dict):
        return []
    raw_hits = hits_obj.get("hits") or []
    results: list[LogHitOutput] = []
    for item in raw_hits[:limit]:
        if not isinstance(item, dict):
            continue
        source = item.get("_source")
        if not isinstance(source, dict):
            source = {}
        message = str(
            source.get("message")
            or source.get("log")
            or source.get("@message")
            or item.get("_id")
            or ""
        )
        timestamp = source.get("@timestamp") or source.get("timestamp")
        results.append(
            LogHitOutput(
                id=str(item.get("_id") or ""),
                message=message,
                timestamp=str(timestamp) if timestamp is not None else None,
                source=source,
            )
        )
    return results
