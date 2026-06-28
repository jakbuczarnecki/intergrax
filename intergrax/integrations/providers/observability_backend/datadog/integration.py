# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Datadog observability vendor integration (INTEGRATIONS-2C)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.observability import (
    ObservabilityVendorIntegrationConfig,
    ObservabilityVendorIntegrationContract,
    ObservabilityVendorPayload,
    ObservabilityVendorSignal,
)

DATADOG_OBSERVABILITY_PROVIDER_ID = "datadog"

_DATADOG_SUPPORTED_SIGNALS: tuple[ObservabilityVendorSignal, ...] = (
    ObservabilityVendorSignal.EVENTS,
    ObservabilityVendorSignal.LOGS,
    ObservabilityVendorSignal.TRACES,
    ObservabilityVendorSignal.METRICS,
)

DATADOG_SUPPORTED_SIGNALS = _DATADOG_SUPPORTED_SIGNALS


class DatadogObservabilityIntegrationConfig(ObservabilityVendorIntegrationConfig):
    """Typed config for Datadog observability vendor integration."""

    pass


@runtime_checkable
class DatadogObservabilityTransport(Protocol):
    """Injectable delivery facade — no vendor SDK or network I/O in the integration class."""

    async def send_observability_payload(self, payload: ObservabilityVendorPayload) -> None:
        """Deliver a policy-sanitized vendor payload to Datadog."""


class DatadogObservabilityIntegration(ObservabilityVendorIntegrationContract):
    """
    Datadog observability vendor integration.

    Consumes only policy-sanitized ObservabilityExportEnvelope records via map_envelope().
    The legacy ObservabilityBackend query facade (create_datadog_observability_backend)
    remains separate and backward-compatible.
    """

    config: DatadogObservabilityIntegrationConfig = DatadogObservabilityIntegrationConfig()
    _transport: DatadogObservabilityTransport | None = PrivateAttr(default=None)

    @classmethod
    def from_transport(
        cls,
        transport: DatadogObservabilityTransport,
        *,
        enabled: bool = False,
        supported_signals: tuple[ObservabilityVendorSignal, ...] | None = None,
    ) -> DatadogObservabilityIntegration:
        integration = cls.for_provider(
            provider_id=DATADOG_OBSERVABILITY_PROVIDER_ID,
            supported_signals=supported_signals or _DATADOG_SUPPORTED_SIGNALS,
            display_name="Datadog",
            config=DatadogObservabilityIntegrationConfig(enabled=enabled),
        )
        integration._transport = transport
        return integration

    @property
    def transport(self) -> DatadogObservabilityTransport | None:
        return self._transport

    async def deliver_payload(self, payload: ObservabilityVendorPayload) -> None:
        if self._transport is None:
            msg = "DatadogObservabilityIntegration requires an injected transport for delivery"
            raise RuntimeError(msg)
        await self._transport.send_observability_payload(payload)
