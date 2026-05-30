# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Elasticsearch observability adapter — ``ObservabilityBackend`` facade (no HTTP here)."""

from __future__ import annotations

from typing import Optional

from intergrax.integrations.contracts.observability_backend import MetricQueryResult
from intergrax.integrations.providers.elasticsearch.client import ElasticsearchRestClient


class ElasticsearchObservabilityBackend:
    """
    Catalog facade over ``ElasticsearchRestClient``.

    The ``promql`` parameters carry Lucene ``query_string`` syntax for Elasticsearch.
    Instantiate via ``create_elasticsearch_observability_backend()`` — not from agent code.
    """

    def __init__(self, client: ElasticsearchRestClient) -> None:
        self._client = client

    @property
    def rest_client(self) -> ElasticsearchRestClient:
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
