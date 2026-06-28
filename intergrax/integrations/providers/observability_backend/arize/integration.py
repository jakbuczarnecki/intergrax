# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Arize observability vendor integration (INTEGRATIONS-2C)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.observability import (
    ObservabilityVendorIntegrationConfig,
    ObservabilityVendorIntegrationContract,
    ObservabilityVendorPayload,
    ObservabilityVendorSignal,
)

ARIZE_OBSERVABILITY_PROVIDER_ID = "arize"

_ARIZE_SUPPORTED_SIGNALS: tuple[ObservabilityVendorSignal, ...] = (
    ObservabilityVendorSignal.EVENTS,
    ObservabilityVendorSignal.TRACES,
    ObservabilityVendorSignal.LLM_EVENTS,
)

ARIZE_SUPPORTED_SIGNALS = _ARIZE_SUPPORTED_SIGNALS


class ArizeObservabilityIntegrationConfig(ObservabilityVendorIntegrationConfig):
    """Typed config for Arize observability vendor integration."""

    pass


@runtime_checkable
class ArizeObservabilityTransport(Protocol):
    """Injectable delivery facade — no vendor SDK or network I/O in the integration class."""

    async def send_observability_payload(self, payload: ObservabilityVendorPayload) -> None:
        """Deliver a policy-sanitized vendor payload to Arize."""


class ArizeObservabilityIntegration(ObservabilityVendorIntegrationContract):
    """
    Arize observability vendor integration.

    Consumes only policy-sanitized ObservabilityExportEnvelope records via map_envelope().
    The legacy ObservabilityBackend query facade (create_arize_observability_backend)
    remains separate and backward-compatible.
    """

    config: ArizeObservabilityIntegrationConfig = ArizeObservabilityIntegrationConfig()
    _transport: ArizeObservabilityTransport | None = PrivateAttr(default=None)

    @classmethod
    def from_transport(
        cls,
        transport: ArizeObservabilityTransport,
        *,
        enabled: bool = False,
        supported_signals: tuple[ObservabilityVendorSignal, ...] | None = None,
    ) -> ArizeObservabilityIntegration:
        integration = cls.for_provider(
            provider_id=ARIZE_OBSERVABILITY_PROVIDER_ID,
            supported_signals=supported_signals or _ARIZE_SUPPORTED_SIGNALS,
            display_name="Arize",
            config=ArizeObservabilityIntegrationConfig(enabled=enabled),
        )
        integration._transport = transport
        return integration

    @property
    def transport(self) -> ArizeObservabilityTransport | None:
        return self._transport

    async def deliver_payload(self, payload: ObservabilityVendorPayload) -> None:
        if self._transport is None:
            msg = "ArizeObservabilityIntegration requires an injected transport for delivery"
            raise RuntimeError(msg)
        await self._transport.send_observability_payload(payload)
