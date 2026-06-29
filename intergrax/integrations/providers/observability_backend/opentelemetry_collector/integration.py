# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Opentelemetry Collector observability vendor integration (INTEGRATIONS-2C · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import HealthStatus, IntegrationConfigurationError
from intergrax.integrations.providers.observability_backend._catalog_client import (
    ObservabilityCatalogClient,
    require_observability_catalog_client,
)
from intergrax.integrations.contracts.observability_backend import MetricQueryResult, ObservabilityBackend, TraceQueryResult
from intergrax.runtime.integrations.observability import (
    ObservabilityVendorIntegrationConfig,
    ObservabilityVendorIntegrationContract,
    ObservabilityVendorPayload,
    ObservabilityVendorSignal,
)

OPENTELEMETRY_COLLECTOR_OBSERVABILITY_PROVIDER_ID = "opentelemetry_collector"

_OPENTELEMETRY_COLLECTOR_SUPPORTED_SIGNALS: tuple[ObservabilityVendorSignal, ...] = (
    ObservabilityVendorSignal.EVENTS,
    ObservabilityVendorSignal.LOGS,
    ObservabilityVendorSignal.TRACES,
    ObservabilityVendorSignal.METRICS,
)

OPENTELEMETRY_COLLECTOR_SUPPORTED_SIGNALS = _OPENTELEMETRY_COLLECTOR_SUPPORTED_SIGNALS

class OpenTelemetryCollectorObservabilityIntegrationConfig(ObservabilityVendorIntegrationConfig):
    """Typed config for Opentelemetry Collector observability vendor integration."""

    pass


@runtime_checkable
class OpenTelemetryCollectorObservabilityTransport(Protocol):
    """Injectable delivery facade — no vendor SDK or network I/O in the integration class."""

    async def send_observability_payload(self, payload: ObservabilityVendorPayload) -> None:
        """Deliver a policy-sanitized vendor payload to Opentelemetry Collector."""


class OpenTelemetryCollectorObservabilityIntegration(ObservabilityVendorIntegrationContract):
    """
    Single public Opentelemetry Collector observability entrypoint.

    Legacy catalog factory (create_opentelemetry_collector_observability_backend) owns catalog query behavior; legacy factories use from_client().
    """

    config: OpenTelemetryCollectorObservabilityIntegrationConfig = OpenTelemetryCollectorObservabilityIntegrationConfig()
    _transport: OpenTelemetryCollectorObservabilityTransport | None = PrivateAttr(default=None)
    _client: ObservabilityCatalogClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: ObservabilityCatalogClient,
        *,
        enabled: bool = True,
        supported_signals: tuple[ObservabilityVendorSignal, ...] | None = None,
    ) -> OpenTelemetryCollectorObservabilityIntegration:
        signals = supported_signals or OPENTELEMETRY_COLLECTOR_SUPPORTED_SIGNALS
        integration = cls.for_provider(
            provider_id=OPENTELEMETRY_COLLECTOR_OBSERVABILITY_PROVIDER_ID,
            supported_signals=signals,
            display_name="Opentelemetry Collector",
            config=OpenTelemetryCollectorObservabilityIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> ObservabilityCatalogClient | None:
        return self._client

    def query_instant(self, promql: str, *, eval_time: float | None = None) -> MetricQueryResult:
        return self._require_client().query_instant(promql, eval_time=eval_time)

    def query_range(
        self,
        promql: str,
        *,
        start: float,
        end: float,
        step: str = "15s",
    ) -> MetricQueryResult:
        return self._require_client().query_range(
            promql,
            start=start,
            end=end,
            step=step,
        )

    def query_traces(
        self,
        *,
        limit: int = 20,
        name: str | None = None,
    ) -> TraceQueryResult:
        return self._require_client().query_traces(limit=limit, name=name)

    def health(self) -> HealthStatus:
        client = self._require_client()
        health_fn = getattr(client, "health", None)
        if callable(health_fn):
            result = health_fn()
            if isinstance(result, HealthStatus):
                return result
            return HealthStatus(
                slug="opentelemetry_collector",
                healthy=bool(result),
                detail="opentelemetry_collector ready probe",
            )
        raise IntegrationConfigurationError(
            f"{type(self).__name__} catalog client does not support health",
        )

    def _require_client(self) -> ObservabilityCatalogClient:
        return require_observability_catalog_client(self, self._client)


    @classmethod
    def from_transport(
        cls,
        transport: OpenTelemetryCollectorObservabilityTransport,
        *,
        enabled: bool = False,
        supported_signals: tuple[ObservabilityVendorSignal, ...] | None = None,
    ) -> OpenTelemetryCollectorObservabilityIntegration:
        integration = cls.for_provider(
            provider_id=OPENTELEMETRY_COLLECTOR_OBSERVABILITY_PROVIDER_ID,
            supported_signals=supported_signals or OPENTELEMETRY_COLLECTOR_SUPPORTED_SIGNALS,
            display_name="Opentelemetry Collector",
            config=OpenTelemetryCollectorObservabilityIntegrationConfig(enabled=enabled),
        )
        integration._transport = transport
        return integration

    @property
    def transport(self) -> OpenTelemetryCollectorObservabilityTransport | None:
        return self._transport

    async def deliver_payload(self, payload: ObservabilityVendorPayload) -> None:
        if self._transport is None:
            msg = "OpenTelemetryCollectorObservabilityIntegration requires an injected transport for delivery"
            raise RuntimeError(msg)
        await self._transport.send_observability_payload(payload)


ObservabilityBackend.register(OpenTelemetryCollectorObservabilityIntegration)
