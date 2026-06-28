# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Grafana observability vendor integration (INTEGRATIONS-2C)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.observability import (
    ObservabilityVendorIntegrationConfig,
    ObservabilityVendorIntegrationContract,
    ObservabilityVendorPayload,
    ObservabilityVendorSignal,
)

GRAFANA_OBSERVABILITY_PROVIDER_ID = "grafana"

_GRAFANA_SUPPORTED_SIGNALS: tuple[ObservabilityVendorSignal, ...] = (
    ObservabilityVendorSignal.EVENTS,
    ObservabilityVendorSignal.LOGS,
    ObservabilityVendorSignal.TRACES,
    ObservabilityVendorSignal.METRICS,
)

GRAFANA_SUPPORTED_SIGNALS = _GRAFANA_SUPPORTED_SIGNALS


class GrafanaObservabilityIntegrationConfig(ObservabilityVendorIntegrationConfig):
    """Typed config for Grafana observability vendor integration."""

    pass


@runtime_checkable
class GrafanaObservabilityTransport(Protocol):
    """Injectable delivery facade — no vendor SDK or network I/O in the integration class."""

    async def send_observability_payload(self, payload: ObservabilityVendorPayload) -> None:
        """Deliver a policy-sanitized vendor payload to Grafana."""


class GrafanaObservabilityIntegration(ObservabilityVendorIntegrationContract):
    """
    Grafana observability vendor integration.

    Consumes only policy-sanitized ObservabilityExportEnvelope records via map_envelope().
    The legacy ObservabilityBackend query facade (create_grafana_observability_backend)
    remains separate and backward-compatible.
    """

    config: GrafanaObservabilityIntegrationConfig = GrafanaObservabilityIntegrationConfig()
    _transport: GrafanaObservabilityTransport | None = PrivateAttr(default=None)

    @classmethod
    def from_transport(
        cls,
        transport: GrafanaObservabilityTransport,
        *,
        enabled: bool = False,
        supported_signals: tuple[ObservabilityVendorSignal, ...] | None = None,
    ) -> GrafanaObservabilityIntegration:
        integration = cls.for_provider(
            provider_id=GRAFANA_OBSERVABILITY_PROVIDER_ID,
            supported_signals=supported_signals or _GRAFANA_SUPPORTED_SIGNALS,
            display_name="Grafana",
            config=GrafanaObservabilityIntegrationConfig(enabled=enabled),
        )
        integration._transport = transport
        return integration

    @property
    def transport(self) -> GrafanaObservabilityTransport | None:
        return self._transport

    async def deliver_payload(self, payload: ObservabilityVendorPayload) -> None:
        if self._transport is None:
            msg = "GrafanaObservabilityIntegration requires an injected transport for delivery"
            raise RuntimeError(msg)
        await self._transport.send_observability_payload(payload)
