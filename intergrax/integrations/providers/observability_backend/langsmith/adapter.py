# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""LangSmith observability adapter."""

from __future__ import annotations

from typing import Optional

from intergrax.integrations.contracts.observability_backend import MetricQueryResult, TraceQueryResult
from intergrax.integrations.providers.observability_backend.langsmith.client import LangSmithRestClient


class _LangSmithObservabilityBackend:
    """Catalog facade over ``LangSmithRestClient``."""

    def __init__(self, client: LangSmithRestClient) -> None:
        self._client = client

    @property
    def rest_client(self) -> LangSmithRestClient:
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
        return self._client.query_traces(limit=limit, name=name)
