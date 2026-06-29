# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shared observability catalog query surface for provider integrations."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.observability_backend import MetricQueryResult, TraceQueryResult


@runtime_checkable
class ObservabilityCatalogClient(Protocol):
    """Typed catalog query client — metrics and traces without vendor SDK imports."""

    def query_instant(self, promql: str, *, eval_time: float | None = None) -> MetricQueryResult:
        """Instant vector query."""

    def query_range(
        self,
        promql: str,
        *,
        start: float,
        end: float,
        step: str = "15s",
    ) -> MetricQueryResult:
        """Range matrix query."""

    def query_traces(
        self,
        *,
        limit: int = 20,
        name: str | None = None,
    ) -> TraceQueryResult:
        """Trace listing query."""


def require_observability_catalog_client(
    owner: object,
    client: ObservabilityCatalogClient | None,
) -> ObservabilityCatalogClient:
    if client is None:
        raise IntegrationConfigurationError(
            f"{type(owner).__name__} requires a catalog client for query operations",
        )
    return client
