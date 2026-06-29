# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Braintrust observability adapter."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from intergrax.integrations.contracts.observability_backend import MetricQueryResult, TraceQueryResult
from intergrax.integrations.providers.observability_backend.braintrust.client import BraintrustRestClient


class _BraintrustObservabilityBackend:
    """Catalog facade over ``BraintrustRestClient``."""

    def __init__(self, client: BraintrustRestClient) -> None:
        self._client = client

    @property
    def rest_client(self) -> BraintrustRestClient:
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

    def log_eval(
        self,
        *,
        name: str,
        score: float,
        metadata: Optional[Mapping[str, Any]] = None,
        project: Optional[str] = None,
    ) -> str:
        return self._client.log_eval(name=name, score=score, metadata=metadata, project=project)
