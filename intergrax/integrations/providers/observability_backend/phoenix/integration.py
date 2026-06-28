# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Phoenix observability vendor integration (INTEGRATIONS-2C)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.observability import (
    ObservabilityVendorIntegrationConfig,
    ObservabilityVendorIntegrationContract,
    ObservabilityVendorPayload,
    ObservabilityVendorSignal,
)

PHOENIX_OBSERVABILITY_PROVIDER_ID = "phoenix"

_PHOENIX_SUPPORTED_SIGNALS: tuple[ObservabilityVendorSignal, ...] = (
    ObservabilityVendorSignal.EVENTS,
    ObservabilityVendorSignal.TRACES,
    ObservabilityVendorSignal.LLM_EVENTS,
)

PHOENIX_SUPPORTED_SIGNALS = _PHOENIX_SUPPORTED_SIGNALS


class PhoenixObservabilityIntegrationConfig(ObservabilityVendorIntegrationConfig):
    """Typed config for Phoenix observability vendor integration."""

    pass


@runtime_checkable
class PhoenixObservabilityTransport(Protocol):
    """Injectable delivery facade — no vendor SDK or network I/O in the integration class."""

    async def send_observability_payload(self, payload: ObservabilityVendorPayload) -> None:
        """Deliver a policy-sanitized vendor payload to Phoenix."""


class PhoenixObservabilityIntegration(ObservabilityVendorIntegrationContract):
    """
    Phoenix observability vendor integration.

    Consumes only policy-sanitized ObservabilityExportEnvelope records via map_envelope().
    The legacy ObservabilityBackend query facade (create_phoenix_observability_backend)
    remains separate and backward-compatible.
    """

    config: PhoenixObservabilityIntegrationConfig = PhoenixObservabilityIntegrationConfig()
    _transport: PhoenixObservabilityTransport | None = PrivateAttr(default=None)

    @classmethod
    def from_transport(
        cls,
        transport: PhoenixObservabilityTransport,
        *,
        enabled: bool = False,
        supported_signals: tuple[ObservabilityVendorSignal, ...] | None = None,
    ) -> PhoenixObservabilityIntegration:
        integration = cls.for_provider(
            provider_id=PHOENIX_OBSERVABILITY_PROVIDER_ID,
            supported_signals=supported_signals or _PHOENIX_SUPPORTED_SIGNALS,
            display_name="Phoenix",
            config=PhoenixObservabilityIntegrationConfig(enabled=enabled),
        )
        integration._transport = transport
        return integration

    @property
    def transport(self) -> PhoenixObservabilityTransport | None:
        return self._transport

    async def deliver_payload(self, payload: ObservabilityVendorPayload) -> None:
        if self._transport is None:
            msg = "PhoenixObservabilityIntegration requires an injected transport for delivery"
            raise RuntimeError(msg)
        await self._transport.send_observability_payload(payload)
