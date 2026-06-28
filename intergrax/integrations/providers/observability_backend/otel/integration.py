# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""OTel observability vendor integration (INTEGRATIONS-2C)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.observability import (
    ObservabilityVendorIntegrationConfig,
    ObservabilityVendorIntegrationContract,
    ObservabilityVendorPayload,
    ObservabilityVendorSignal,
)

OTEL_OBSERVABILITY_PROVIDER_ID = "otel"

_OTEL_SUPPORTED_SIGNALS: tuple[ObservabilityVendorSignal, ...] = (
    ObservabilityVendorSignal.EVENTS,
    ObservabilityVendorSignal.LOGS,
    ObservabilityVendorSignal.TRACES,
    ObservabilityVendorSignal.METRICS,
)

OTEL_SUPPORTED_SIGNALS = _OTEL_SUPPORTED_SIGNALS


class OtelObservabilityIntegrationConfig(ObservabilityVendorIntegrationConfig):
    """Typed config for OTel observability vendor integration."""

    pass


@runtime_checkable
class OtelObservabilityTransport(Protocol):
    """Injectable delivery facade — no vendor SDK or network I/O in the integration class."""

    async def send_observability_payload(self, payload: ObservabilityVendorPayload) -> None:
        """Deliver a policy-sanitized vendor payload to OTel."""


class OtelObservabilityIntegration(ObservabilityVendorIntegrationContract):
    """
    OTel observability vendor integration.

    Consumes only policy-sanitized ObservabilityExportEnvelope records via map_envelope().
    The legacy ObservabilityBackend query facade (create_otel_observability_backend)
    remains separate and backward-compatible.
    """

    config: OtelObservabilityIntegrationConfig = OtelObservabilityIntegrationConfig()
    _transport: OtelObservabilityTransport | None = PrivateAttr(default=None)

    @classmethod
    def from_transport(
        cls,
        transport: OtelObservabilityTransport,
        *,
        enabled: bool = False,
        supported_signals: tuple[ObservabilityVendorSignal, ...] | None = None,
    ) -> OtelObservabilityIntegration:
        integration = cls.for_provider(
            provider_id=OTEL_OBSERVABILITY_PROVIDER_ID,
            supported_signals=supported_signals or _OTEL_SUPPORTED_SIGNALS,
            display_name="OTel",
            config=OtelObservabilityIntegrationConfig(enabled=enabled),
        )
        integration._transport = transport
        return integration

    @property
    def transport(self) -> OtelObservabilityTransport | None:
        return self._transport

    async def deliver_payload(self, payload: ObservabilityVendorPayload) -> None:
        if self._transport is None:
            msg = "OtelObservabilityIntegration requires an injected transport for delivery"
            raise RuntimeError(msg)
        await self._transport.send_observability_payload(payload)
