# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tempo observability vendor integration (INTEGRATIONS-2C)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.observability import (
    ObservabilityVendorIntegrationConfig,
    ObservabilityVendorIntegrationContract,
    ObservabilityVendorPayload,
    ObservabilityVendorSignal,
)

TEMPO_OBSERVABILITY_PROVIDER_ID = "tempo"

_TEMPO_SUPPORTED_SIGNALS: tuple[ObservabilityVendorSignal, ...] = (
    ObservabilityVendorSignal.EVENTS,
    ObservabilityVendorSignal.LOGS,
    ObservabilityVendorSignal.TRACES,
    ObservabilityVendorSignal.METRICS,
)

TEMPO_SUPPORTED_SIGNALS = _TEMPO_SUPPORTED_SIGNALS


class TempoObservabilityIntegrationConfig(ObservabilityVendorIntegrationConfig):
    """Typed config for Tempo observability vendor integration."""

    pass


@runtime_checkable
class TempoObservabilityTransport(Protocol):
    """Injectable delivery facade — no vendor SDK or network I/O in the integration class."""

    async def send_observability_payload(self, payload: ObservabilityVendorPayload) -> None:
        """Deliver a policy-sanitized vendor payload to Tempo."""


class TempoObservabilityIntegration(ObservabilityVendorIntegrationContract):
    """
    Tempo observability vendor integration.

    Consumes only policy-sanitized ObservabilityExportEnvelope records via map_envelope().
    The legacy ObservabilityBackend query facade (create_tempo_observability_backend)
    remains separate and backward-compatible.
    """

    config: TempoObservabilityIntegrationConfig = TempoObservabilityIntegrationConfig()
    _transport: TempoObservabilityTransport | None = PrivateAttr(default=None)

    @classmethod
    def from_transport(
        cls,
        transport: TempoObservabilityTransport,
        *,
        enabled: bool = False,
        supported_signals: tuple[ObservabilityVendorSignal, ...] | None = None,
    ) -> TempoObservabilityIntegration:
        integration = cls.for_provider(
            provider_id=TEMPO_OBSERVABILITY_PROVIDER_ID,
            supported_signals=supported_signals or _TEMPO_SUPPORTED_SIGNALS,
            display_name="Tempo",
            config=TempoObservabilityIntegrationConfig(enabled=enabled),
        )
        integration._transport = transport
        return integration

    @property
    def transport(self) -> TempoObservabilityTransport | None:
        return self._transport

    async def deliver_payload(self, payload: ObservabilityVendorPayload) -> None:
        if self._transport is None:
            msg = "TempoObservabilityIntegration requires an injected transport for delivery"
            raise RuntimeError(msg)
        await self._transport.send_observability_payload(payload)
