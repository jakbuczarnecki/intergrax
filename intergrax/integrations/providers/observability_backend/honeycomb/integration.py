# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Honeycomb observability vendor integration (INTEGRATIONS-2C)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.observability import (
    ObservabilityVendorIntegrationConfig,
    ObservabilityVendorIntegrationContract,
    ObservabilityVendorPayload,
    ObservabilityVendorSignal,
)

HONEYCOMB_OBSERVABILITY_PROVIDER_ID = "honeycomb"

_HONEYCOMB_SUPPORTED_SIGNALS: tuple[ObservabilityVendorSignal, ...] = (
    ObservabilityVendorSignal.EVENTS,
    ObservabilityVendorSignal.LOGS,
    ObservabilityVendorSignal.TRACES,
    ObservabilityVendorSignal.METRICS,
)

HONEYCOMB_SUPPORTED_SIGNALS = _HONEYCOMB_SUPPORTED_SIGNALS


class HoneycombObservabilityIntegrationConfig(ObservabilityVendorIntegrationConfig):
    """Typed config for Honeycomb observability vendor integration."""

    pass


@runtime_checkable
class HoneycombObservabilityTransport(Protocol):
    """Injectable delivery facade — no vendor SDK or network I/O in the integration class."""

    async def send_observability_payload(self, payload: ObservabilityVendorPayload) -> None:
        """Deliver a policy-sanitized vendor payload to Honeycomb."""


class HoneycombObservabilityIntegration(ObservabilityVendorIntegrationContract):
    """
    Honeycomb observability vendor integration.

    Consumes only policy-sanitized ObservabilityExportEnvelope records via map_envelope().
    The legacy ObservabilityBackend query facade (create_honeycomb_observability_backend)
    remains separate and backward-compatible.
    """

    config: HoneycombObservabilityIntegrationConfig = HoneycombObservabilityIntegrationConfig()
    _transport: HoneycombObservabilityTransport | None = PrivateAttr(default=None)

    @classmethod
    def from_transport(
        cls,
        transport: HoneycombObservabilityTransport,
        *,
        enabled: bool = False,
        supported_signals: tuple[ObservabilityVendorSignal, ...] | None = None,
    ) -> HoneycombObservabilityIntegration:
        integration = cls.for_provider(
            provider_id=HONEYCOMB_OBSERVABILITY_PROVIDER_ID,
            supported_signals=supported_signals or _HONEYCOMB_SUPPORTED_SIGNALS,
            display_name="Honeycomb",
            config=HoneycombObservabilityIntegrationConfig(enabled=enabled),
        )
        integration._transport = transport
        return integration

    @property
    def transport(self) -> HoneycombObservabilityTransport | None:
        return self._transport

    async def deliver_payload(self, payload: ObservabilityVendorPayload) -> None:
        if self._transport is None:
            msg = "HoneycombObservabilityIntegration requires an injected transport for delivery"
            raise RuntimeError(msg)
        await self._transport.send_observability_payload(payload)
