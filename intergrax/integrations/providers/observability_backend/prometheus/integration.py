# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Prometheus observability vendor integration (INTEGRATIONS-2C)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.observability import (
    ObservabilityVendorIntegrationConfig,
    ObservabilityVendorIntegrationContract,
    ObservabilityVendorPayload,
    ObservabilityVendorSignal,
)

PROMETHEUS_OBSERVABILITY_PROVIDER_ID = "prometheus"

_PROMETHEUS_SUPPORTED_SIGNALS: tuple[ObservabilityVendorSignal, ...] = (
    ObservabilityVendorSignal.EVENTS,
    ObservabilityVendorSignal.LOGS,
    ObservabilityVendorSignal.TRACES,
    ObservabilityVendorSignal.METRICS,
)

PROMETHEUS_SUPPORTED_SIGNALS = _PROMETHEUS_SUPPORTED_SIGNALS


class PrometheusObservabilityIntegrationConfig(ObservabilityVendorIntegrationConfig):
    """Typed config for Prometheus observability vendor integration."""

    pass


@runtime_checkable
class PrometheusObservabilityTransport(Protocol):
    """Injectable delivery facade — no vendor SDK or network I/O in the integration class."""

    async def send_observability_payload(self, payload: ObservabilityVendorPayload) -> None:
        """Deliver a policy-sanitized vendor payload to Prometheus."""


class PrometheusObservabilityIntegration(ObservabilityVendorIntegrationContract):
    """
    Prometheus observability vendor integration.

    Consumes only policy-sanitized ObservabilityExportEnvelope records via map_envelope().
    The legacy ObservabilityBackend query facade (create_prometheus_observability_backend)
    remains separate and backward-compatible.
    """

    config: PrometheusObservabilityIntegrationConfig = PrometheusObservabilityIntegrationConfig()
    _transport: PrometheusObservabilityTransport | None = PrivateAttr(default=None)

    @classmethod
    def from_transport(
        cls,
        transport: PrometheusObservabilityTransport,
        *,
        enabled: bool = False,
        supported_signals: tuple[ObservabilityVendorSignal, ...] | None = None,
    ) -> PrometheusObservabilityIntegration:
        integration = cls.for_provider(
            provider_id=PROMETHEUS_OBSERVABILITY_PROVIDER_ID,
            supported_signals=supported_signals or _PROMETHEUS_SUPPORTED_SIGNALS,
            display_name="Prometheus",
            config=PrometheusObservabilityIntegrationConfig(enabled=enabled),
        )
        integration._transport = transport
        return integration

    @property
    def transport(self) -> PrometheusObservabilityTransport | None:
        return self._transport

    async def deliver_payload(self, payload: ObservabilityVendorPayload) -> None:
        if self._transport is None:
            msg = "PrometheusObservabilityIntegration requires an injected transport for delivery"
            raise RuntimeError(msg)
        await self._transport.send_observability_payload(payload)
