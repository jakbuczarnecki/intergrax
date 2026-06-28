# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""SigNoz observability vendor integration (INTEGRATIONS-2C)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.observability import (
    ObservabilityVendorIntegrationConfig,
    ObservabilityVendorIntegrationContract,
    ObservabilityVendorPayload,
    ObservabilityVendorSignal,
)

SIGNOZ_OBSERVABILITY_PROVIDER_ID = "signoz"

_SIGNOZ_SUPPORTED_SIGNALS: tuple[ObservabilityVendorSignal, ...] = (
    ObservabilityVendorSignal.EVENTS,
    ObservabilityVendorSignal.LOGS,
    ObservabilityVendorSignal.TRACES,
    ObservabilityVendorSignal.METRICS,
)

SIGNOZ_SUPPORTED_SIGNALS = _SIGNOZ_SUPPORTED_SIGNALS


class SignozObservabilityIntegrationConfig(ObservabilityVendorIntegrationConfig):
    """Typed config for SigNoz observability vendor integration."""

    pass


@runtime_checkable
class SignozObservabilityTransport(Protocol):
    """Injectable delivery facade — no vendor SDK or network I/O in the integration class."""

    async def send_observability_payload(self, payload: ObservabilityVendorPayload) -> None:
        """Deliver a policy-sanitized vendor payload to SigNoz."""


class SignozObservabilityIntegration(ObservabilityVendorIntegrationContract):
    """
    SigNoz observability vendor integration.

    Consumes only policy-sanitized ObservabilityExportEnvelope records via map_envelope().
    The legacy ObservabilityBackend query facade (create_signoz_observability_backend)
    remains separate and backward-compatible.
    """

    config: SignozObservabilityIntegrationConfig = SignozObservabilityIntegrationConfig()
    _transport: SignozObservabilityTransport | None = PrivateAttr(default=None)

    @classmethod
    def from_transport(
        cls,
        transport: SignozObservabilityTransport,
        *,
        enabled: bool = False,
        supported_signals: tuple[ObservabilityVendorSignal, ...] | None = None,
    ) -> SignozObservabilityIntegration:
        integration = cls.for_provider(
            provider_id=SIGNOZ_OBSERVABILITY_PROVIDER_ID,
            supported_signals=supported_signals or _SIGNOZ_SUPPORTED_SIGNALS,
            display_name="SigNoz",
            config=SignozObservabilityIntegrationConfig(enabled=enabled),
        )
        integration._transport = transport
        return integration

    @property
    def transport(self) -> SignozObservabilityTransport | None:
        return self._transport

    async def deliver_payload(self, payload: ObservabilityVendorPayload) -> None:
        if self._transport is None:
            msg = "SignozObservabilityIntegration requires an injected transport for delivery"
            raise RuntimeError(msg)
        await self._transport.send_observability_payload(payload)
