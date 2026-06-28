# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Helicone observability vendor integration (INTEGRATIONS-2C)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.observability import (
    ObservabilityVendorIntegrationConfig,
    ObservabilityVendorIntegrationContract,
    ObservabilityVendorPayload,
    ObservabilityVendorSignal,
)

HELICONE_OBSERVABILITY_PROVIDER_ID = "helicone"

_HELICONE_SUPPORTED_SIGNALS: tuple[ObservabilityVendorSignal, ...] = (
    ObservabilityVendorSignal.EVENTS,
    ObservabilityVendorSignal.TRACES,
    ObservabilityVendorSignal.LLM_EVENTS,
)

HELICONE_SUPPORTED_SIGNALS = _HELICONE_SUPPORTED_SIGNALS


class HeliconeObservabilityIntegrationConfig(ObservabilityVendorIntegrationConfig):
    """Typed config for Helicone observability vendor integration."""

    pass


@runtime_checkable
class HeliconeObservabilityTransport(Protocol):
    """Injectable delivery facade — no vendor SDK or network I/O in the integration class."""

    async def send_observability_payload(self, payload: ObservabilityVendorPayload) -> None:
        """Deliver a policy-sanitized vendor payload to Helicone."""


class HeliconeObservabilityIntegration(ObservabilityVendorIntegrationContract):
    """
    Helicone observability vendor integration.

    Consumes only policy-sanitized ObservabilityExportEnvelope records via map_envelope().
    The legacy ObservabilityBackend query facade (create_helicone_observability_backend)
    remains separate and backward-compatible.
    """

    config: HeliconeObservabilityIntegrationConfig = HeliconeObservabilityIntegrationConfig()
    _transport: HeliconeObservabilityTransport | None = PrivateAttr(default=None)

    @classmethod
    def from_transport(
        cls,
        transport: HeliconeObservabilityTransport,
        *,
        enabled: bool = False,
        supported_signals: tuple[ObservabilityVendorSignal, ...] | None = None,
    ) -> HeliconeObservabilityIntegration:
        integration = cls.for_provider(
            provider_id=HELICONE_OBSERVABILITY_PROVIDER_ID,
            supported_signals=supported_signals or _HELICONE_SUPPORTED_SIGNALS,
            display_name="Helicone",
            config=HeliconeObservabilityIntegrationConfig(enabled=enabled),
        )
        integration._transport = transport
        return integration

    @property
    def transport(self) -> HeliconeObservabilityTransport | None:
        return self._transport

    async def deliver_payload(self, payload: ObservabilityVendorPayload) -> None:
        if self._transport is None:
            msg = "HeliconeObservabilityIntegration requires an injected transport for delivery"
            raise RuntimeError(msg)
        await self._transport.send_observability_payload(payload)
