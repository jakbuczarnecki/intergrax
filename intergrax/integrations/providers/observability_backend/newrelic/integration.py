# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""New Relic observability vendor integration (INTEGRATIONS-2C)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.observability import (
    ObservabilityVendorIntegrationConfig,
    ObservabilityVendorIntegrationContract,
    ObservabilityVendorPayload,
    ObservabilityVendorSignal,
)

NEWRELIC_OBSERVABILITY_PROVIDER_ID = "newrelic"

_NEWRELIC_SUPPORTED_SIGNALS: tuple[ObservabilityVendorSignal, ...] = (
    ObservabilityVendorSignal.EVENTS,
    ObservabilityVendorSignal.LOGS,
    ObservabilityVendorSignal.TRACES,
    ObservabilityVendorSignal.METRICS,
)

NEWRELIC_SUPPORTED_SIGNALS = _NEWRELIC_SUPPORTED_SIGNALS


class NewrelicObservabilityIntegrationConfig(ObservabilityVendorIntegrationConfig):
    """Typed config for New Relic observability vendor integration."""

    pass


@runtime_checkable
class NewrelicObservabilityTransport(Protocol):
    """Injectable delivery facade — no vendor SDK or network I/O in the integration class."""

    async def send_observability_payload(self, payload: ObservabilityVendorPayload) -> None:
        """Deliver a policy-sanitized vendor payload to New Relic."""


class NewrelicObservabilityIntegration(ObservabilityVendorIntegrationContract):
    """
    New Relic observability vendor integration.

    Consumes only policy-sanitized ObservabilityExportEnvelope records via map_envelope().
    The legacy ObservabilityBackend query facade (create_newrelic_observability_backend)
    remains separate and backward-compatible.
    """

    config: NewrelicObservabilityIntegrationConfig = NewrelicObservabilityIntegrationConfig()
    _transport: NewrelicObservabilityTransport | None = PrivateAttr(default=None)

    @classmethod
    def from_transport(
        cls,
        transport: NewrelicObservabilityTransport,
        *,
        enabled: bool = False,
        supported_signals: tuple[ObservabilityVendorSignal, ...] | None = None,
    ) -> NewrelicObservabilityIntegration:
        integration = cls.for_provider(
            provider_id=NEWRELIC_OBSERVABILITY_PROVIDER_ID,
            supported_signals=supported_signals or _NEWRELIC_SUPPORTED_SIGNALS,
            display_name="New Relic",
            config=NewrelicObservabilityIntegrationConfig(enabled=enabled),
        )
        integration._transport = transport
        return integration

    @property
    def transport(self) -> NewrelicObservabilityTransport | None:
        return self._transport

    async def deliver_payload(self, payload: ObservabilityVendorPayload) -> None:
        if self._transport is None:
            msg = "NewrelicObservabilityIntegration requires an injected transport for delivery"
            raise RuntimeError(msg)
        await self._transport.send_observability_payload(payload)
