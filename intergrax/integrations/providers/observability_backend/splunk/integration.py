# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Splunk observability vendor integration (INTEGRATIONS-2C)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.observability import (
    ObservabilityVendorIntegrationConfig,
    ObservabilityVendorIntegrationContract,
    ObservabilityVendorPayload,
    ObservabilityVendorSignal,
)

SPLUNK_OBSERVABILITY_PROVIDER_ID = "splunk"

_SPLUNK_SUPPORTED_SIGNALS: tuple[ObservabilityVendorSignal, ...] = (
    ObservabilityVendorSignal.EVENTS,
    ObservabilityVendorSignal.LOGS,
    ObservabilityVendorSignal.TRACES,
    ObservabilityVendorSignal.METRICS,
)

SPLUNK_SUPPORTED_SIGNALS = _SPLUNK_SUPPORTED_SIGNALS


class SplunkObservabilityIntegrationConfig(ObservabilityVendorIntegrationConfig):
    """Typed config for Splunk observability vendor integration."""

    pass


@runtime_checkable
class SplunkObservabilityTransport(Protocol):
    """Injectable delivery facade — no vendor SDK or network I/O in the integration class."""

    async def send_observability_payload(self, payload: ObservabilityVendorPayload) -> None:
        """Deliver a policy-sanitized vendor payload to Splunk."""


class SplunkObservabilityIntegration(ObservabilityVendorIntegrationContract):
    """
    Splunk observability vendor integration.

    Consumes only policy-sanitized ObservabilityExportEnvelope records via map_envelope().
    The legacy ObservabilityBackend query facade (create_splunk_observability_backend)
    remains separate and backward-compatible.
    """

    config: SplunkObservabilityIntegrationConfig = SplunkObservabilityIntegrationConfig()
    _transport: SplunkObservabilityTransport | None = PrivateAttr(default=None)

    @classmethod
    def from_transport(
        cls,
        transport: SplunkObservabilityTransport,
        *,
        enabled: bool = False,
        supported_signals: tuple[ObservabilityVendorSignal, ...] | None = None,
    ) -> SplunkObservabilityIntegration:
        integration = cls.for_provider(
            provider_id=SPLUNK_OBSERVABILITY_PROVIDER_ID,
            supported_signals=supported_signals or _SPLUNK_SUPPORTED_SIGNALS,
            display_name="Splunk",
            config=SplunkObservabilityIntegrationConfig(enabled=enabled),
        )
        integration._transport = transport
        return integration

    @property
    def transport(self) -> SplunkObservabilityTransport | None:
        return self._transport

    async def deliver_payload(self, payload: ObservabilityVendorPayload) -> None:
        if self._transport is None:
            msg = "SplunkObservabilityIntegration requires an injected transport for delivery"
            raise RuntimeError(msg)
        await self._transport.send_observability_payload(payload)
