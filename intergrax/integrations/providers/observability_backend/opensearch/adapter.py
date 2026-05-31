# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""OpenSearch observability adapter."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from intergrax.integrations.contracts.observability_backend import MetricQueryResult, TraceQueryResult
from intergrax.integrations.providers.observability_backend.opensearch.client import OpenSearchRestClient


class OpenSearchObservabilityBackend:
    """Catalog facade over ``OpenSearchRestClient``."""

    def __init__(self, client: OpenSearchRestClient) -> None:
        self._client = client

    @property
    def rest_client(self) -> OpenSearchRestClient:
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

    def index_document(self, *, index: str, document: Mapping[str, Any], doc_id: Optional[str] = None) -> str:
        return self._client.index_document(index=index, document=document, doc_id=doc_id)

    def ensure_index(self, index: str) -> bool:
        return self._client.ensure_index(index)
