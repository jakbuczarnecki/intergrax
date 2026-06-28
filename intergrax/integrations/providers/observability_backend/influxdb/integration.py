# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Influxdb observability vendor integration (INTEGRATIONS-2C)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.observability import (
    ObservabilityVendorIntegrationConfig,
    ObservabilityVendorIntegrationContract,
    ObservabilityVendorPayload,
    ObservabilityVendorSignal,
)

INFLUXDB_OBSERVABILITY_PROVIDER_ID = "influxdb"

_INFLUXDB_SUPPORTED_SIGNALS: tuple[ObservabilityVendorSignal, ...] = (
    ObservabilityVendorSignal.EVENTS,
    ObservabilityVendorSignal.LOGS,
    ObservabilityVendorSignal.TRACES,
    ObservabilityVendorSignal.METRICS,
)

INFLUXDB_SUPPORTED_SIGNALS = _INFLUXDB_SUPPORTED_SIGNALS


class InfluxdbObservabilityIntegrationConfig(ObservabilityVendorIntegrationConfig):
    """Typed config for Influxdb observability vendor integration."""

    pass


@runtime_checkable
class InfluxdbObservabilityTransport(Protocol):
    """Injectable delivery facade — no vendor SDK or network I/O in the integration class."""

    async def send_observability_payload(self, payload: ObservabilityVendorPayload) -> None:
        """Deliver a policy-sanitized vendor payload to Influxdb."""


class InfluxdbObservabilityIntegration(ObservabilityVendorIntegrationContract):
    """
    Influxdb observability vendor integration.

    Consumes only policy-sanitized ObservabilityExportEnvelope records via map_envelope().
    The legacy ObservabilityBackend query facade (create_influxdb_observability_backend)
    remains separate and backward-compatible.
    """

    config: InfluxdbObservabilityIntegrationConfig = InfluxdbObservabilityIntegrationConfig()
    _transport: InfluxdbObservabilityTransport | None = PrivateAttr(default=None)

    @classmethod
    def from_transport(
        cls,
        transport: InfluxdbObservabilityTransport,
        *,
        enabled: bool = False,
        supported_signals: tuple[ObservabilityVendorSignal, ...] | None = None,
    ) -> InfluxdbObservabilityIntegration:
        integration = cls.for_provider(
            provider_id=INFLUXDB_OBSERVABILITY_PROVIDER_ID,
            supported_signals=supported_signals or _INFLUXDB_SUPPORTED_SIGNALS,
            display_name="Influxdb",
            config=InfluxdbObservabilityIntegrationConfig(enabled=enabled),
        )
        integration._transport = transport
        return integration

    @property
    def transport(self) -> InfluxdbObservabilityTransport | None:
        return self._transport

    async def deliver_payload(self, payload: ObservabilityVendorPayload) -> None:
        if self._transport is None:
            msg = "InfluxdbObservabilityIntegration requires an injected transport for delivery"
            raise RuntimeError(msg)
        await self._transport.send_observability_payload(payload)
