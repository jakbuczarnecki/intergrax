# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pytest

from intergrax.integrations.contracts.observability_backend import MetricPoint, MetricQueryResult, MetricSeries
from intergrax.tools.providers.observability.contracts import (
    LogsSearchInput,
    LogsTailInput,
    MetricsQueryInstantInput,
    MetricsQueryRangeInput,
)
from intergrax.tools.providers.observability.service import (
    logs_search,
    logs_tail,
    metrics_query_instant,
    metrics_query_range,
)
from intergrax.tools.registry.bootstrap import register_default_tools, reset_default_tools_bootstrap
from intergrax.tools.registry.catalog import clear_tool_catalog, get_bundle, list_catalog_tool_ids
from intergrax.tools.registry.factory import build_registry_from_profile
from intergrax.tools.registry.profile import ToolProfile
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


class FakePrometheusBackend:
    def query_instant(self, promql: str, *, eval_time: Optional[float] = None) -> MetricQueryResult:
        return MetricQueryResult(
            result_type="vector",
            series=[
                MetricSeries(
                    metric={"__name__": "up", "job": "api"},
                    points=[MetricPoint(timestamp=1710000000.0, value=1.0)],
                )
            ],
        )

    def query_range(
        self,
        promql: str,
        *,
        start: float,
        end: float,
        step: str = "15s",
    ) -> MetricQueryResult:
        return MetricQueryResult(
            result_type="matrix",
            series=[
                MetricSeries(
                    metric={"__name__": "requests_total"},
                    points=[
                        MetricPoint(timestamp=start, value=1.0),
                        MetricPoint(timestamp=end, value=2.0),
                    ],
                )
            ],
        )


@dataclass
class FakeElasticsearchConfig:
    index: str = "logs-*"
    timestamp_field: str = "@timestamp"


@dataclass
class FakeHttpClient:
    last_body: dict[str, Any] = field(default_factory=dict)
    response_payload: dict[str, Any] = field(default_factory=dict)

    def post(self, path: str, json: dict[str, Any]) -> "FakeHttpResponse":
        self.last_body = json
        return FakeHttpResponse(self.response_payload)


@dataclass
class FakeHttpResponse:
    payload: dict[str, Any]

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


@dataclass
class FakeElasticsearchRestClient:
    config: FakeElasticsearchConfig = field(default_factory=FakeElasticsearchConfig)
    _http_client: FakeHttpClient = field(default_factory=FakeHttpClient)

    @property
    def rest_client(self) -> "FakeElasticsearchRestClient":
        return self


class FakeElasticsearchObservabilityBackend:
    def __init__(self, client: FakeElasticsearchRestClient) -> None:
        self.rest_client = client

    def query_instant(self, promql: str, *, eval_time: Optional[float] = None) -> MetricQueryResult:
        raise NotImplementedError


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_tool_catalog()
    reset_default_tools_bootstrap()
    yield
    clear_tool_catalog()
    reset_default_tools_bootstrap()


def test_metrics_query_instant() -> None:
    ctx = ToolWiringContext(observability_backend=FakePrometheusBackend())
    out = metrics_query_instant(ctx, MetricsQueryInstantInput(query='up{job="api"}'))
    assert out.result_type == "vector"
    assert len(out.series) == 1
    assert out.series[0].metric["job"] == "api"
    assert out.series[0].points[0].value == 1.0


def test_metrics_query_range() -> None:
    ctx = ToolWiringContext(observability_backend=FakePrometheusBackend())
    out = metrics_query_range(
        ctx,
        MetricsQueryRangeInput(query="rate(requests_total[5m])", start=1710000000.0, end=1710003600.0),
    )
    assert out.result_type == "matrix"
    assert len(out.series[0].points) == 2


def test_logs_search() -> None:
    http = FakeHttpClient(
        response_payload={
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_id": "log-1",
                        "_source": {
                            "@timestamp": "2026-05-30T10:00:00Z",
                            "message": "Connection timeout",
                        },
                    }
                ],
            }
        }
    )
    client = FakeElasticsearchRestClient(_http_client=http)
    backend = FakeElasticsearchObservabilityBackend(client)
    ctx = ToolWiringContext(observability_backend=backend)

    out = logs_search(ctx, LogsSearchInput(query="timeout", limit=10))

    assert out.total == 1
    assert len(out.hits) == 1
    assert out.hits[0].message == "Connection timeout"
    assert "Connection timeout" in out.context_text
    assert http.last_body["size"] == 10


def test_logs_tail() -> None:
    http = FakeHttpClient(
        response_payload={
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_id": "log-2",
                        "_source": {"@timestamp": "2026-05-30T11:00:00Z", "message": "Tail line"},
                    }
                ],
            }
        }
    )
    client = FakeElasticsearchRestClient(_http_client=http)
    backend = FakeElasticsearchObservabilityBackend(client)
    ctx = ToolWiringContext(observability_backend=backend)
    out = logs_tail(ctx, LogsTailInput(limit=5))
    assert out.total == 1
    assert out.hits[0].message == "Tail line"


def test_observability_backend_not_configured() -> None:
    with pytest.raises(RuntimeError, match="observability_backend_not_configured"):
        metrics_query_instant(ToolWiringContext(), MetricsQueryInstantInput(query="up"))


def test_observability_tools_registered_in_catalog() -> None:
    register_default_tools()
    ids = list_catalog_tool_ids()
    assert "metrics.query_instant" in ids
    assert "logs.search" in ids
    assert "errors.capture" in ids
    assert "observability.query_traces" in ids
    assert get_bundle("observability").tool_ids == (
        "metrics.query_instant",
        "metrics.query_range",
        "logs.search",
        "logs.tail",
        "observability.query_traces",
        "errors.capture",
    )


def test_errors_capture_tool() -> None:
    class _SentryLike:
        def capture_message(self, message: str, *, level: str) -> str:
            return "evt-99"

    from intergrax.tools.providers.observability.contracts import ErrorsCaptureInput
    from intergrax.tools.providers.observability.service import errors_capture

    ctx = ToolWiringContext(observability_backend=_SentryLike())
    out = errors_capture(ctx, ErrorsCaptureInput(message="agent loop failed", level="error"))
    assert out.event_id == "evt-99"


def test_build_registry_enables_observability_bundle() -> None:
    register_default_tools()
    ctx = ToolWiringContext(observability_backend=FakePrometheusBackend())
    registry = build_registry_from_profile(ToolProfile(enabled_bundles=["observability"]), ctx=ctx)
    assert registry.has("metrics.query_instant")
    assert registry.has("logs.search")
