# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""OpenTelemetry Collector observability vendor integration (INTEGRATIONS-2C)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

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


class OpentelemetryCollectorObservabilityIntegrationConfig(ObservabilityVendorIntegrationConfig):
    """Typed config for OpenTelemetry Collector observability vendor integration."""

    pass


@runtime_checkable
class OpentelemetryCollectorObservabilityTransport(Protocol):
    """Injectable delivery facade — no vendor SDK or network I/O in the integration class."""

    async def send_observability_payload(self, payload: ObservabilityVendorPayload) -> None:
        """Deliver a policy-sanitized vendor payload to OpenTelemetry Collector."""


class OpentelemetryCollectorObservabilityIntegration(ObservabilityVendorIntegrationContract):
    """
    OpenTelemetry Collector observability vendor integration.

    Consumes only policy-sanitized ObservabilityExportEnvelope records via map_envelope().
    The legacy ObservabilityBackend query facade (create_opentelemetry_collector_observability_backend)
    remains separate and backward-compatible.
    """

    config: OpentelemetryCollectorObservabilityIntegrationConfig = OpentelemetryCollectorObservabilityIntegrationConfig()
    _transport: OpentelemetryCollectorObservabilityTransport | None = PrivateAttr(default=None)

    @classmethod
    def from_transport(
        cls,
        transport: OpentelemetryCollectorObservabilityTransport,
        *,
        enabled: bool = False,
        supported_signals: tuple[ObservabilityVendorSignal, ...] | None = None,
    ) -> OpentelemetryCollectorObservabilityIntegration:
        integration = cls.for_provider(
            provider_id=OPENTELEMETRY_COLLECTOR_OBSERVABILITY_PROVIDER_ID,
            supported_signals=supported_signals or _OPENTELEMETRY_COLLECTOR_SUPPORTED_SIGNALS,
            display_name="OpenTelemetry Collector",
            config=OpentelemetryCollectorObservabilityIntegrationConfig(enabled=enabled),
        )
        integration._transport = transport
        return integration

    @property
    def transport(self) -> OpentelemetryCollectorObservabilityTransport | None:
        return self._transport

    async def deliver_payload(self, payload: ObservabilityVendorPayload) -> None:
        if self._transport is None:
            msg = "OpentelemetryCollectorObservabilityIntegration requires an injected transport for delivery"
            raise RuntimeError(msg)
        await self._transport.send_observability_payload(payload)
