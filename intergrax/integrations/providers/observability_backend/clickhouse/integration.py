# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Clickhouse observability vendor integration (INTEGRATIONS-2C)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.observability import (
    ObservabilityVendorIntegrationConfig,
    ObservabilityVendorIntegrationContract,
    ObservabilityVendorPayload,
    ObservabilityVendorSignal,
)

CLICKHOUSE_OBSERVABILITY_PROVIDER_ID = "clickhouse"

_CLICKHOUSE_SUPPORTED_SIGNALS: tuple[ObservabilityVendorSignal, ...] = (
    ObservabilityVendorSignal.EVENTS,
    ObservabilityVendorSignal.LOGS,
    ObservabilityVendorSignal.TRACES,
    ObservabilityVendorSignal.METRICS,
)

CLICKHOUSE_SUPPORTED_SIGNALS = _CLICKHOUSE_SUPPORTED_SIGNALS


class ClickhouseObservabilityIntegrationConfig(ObservabilityVendorIntegrationConfig):
    """Typed config for Clickhouse observability vendor integration."""

    pass


@runtime_checkable
class ClickhouseObservabilityTransport(Protocol):
    """Injectable delivery facade — no vendor SDK or network I/O in the integration class."""

    async def send_observability_payload(self, payload: ObservabilityVendorPayload) -> None:
        """Deliver a policy-sanitized vendor payload to Clickhouse."""


class ClickhouseObservabilityIntegration(ObservabilityVendorIntegrationContract):
    """
    Clickhouse observability vendor integration.

    Consumes only policy-sanitized ObservabilityExportEnvelope records via map_envelope().
    The legacy ObservabilityBackend query facade (create_clickhouse_observability_backend)
    remains separate and backward-compatible.
    """

    config: ClickhouseObservabilityIntegrationConfig = ClickhouseObservabilityIntegrationConfig()
    _transport: ClickhouseObservabilityTransport | None = PrivateAttr(default=None)

    @classmethod
    def from_transport(
        cls,
        transport: ClickhouseObservabilityTransport,
        *,
        enabled: bool = False,
        supported_signals: tuple[ObservabilityVendorSignal, ...] | None = None,
    ) -> ClickhouseObservabilityIntegration:
        integration = cls.for_provider(
            provider_id=CLICKHOUSE_OBSERVABILITY_PROVIDER_ID,
            supported_signals=supported_signals or _CLICKHOUSE_SUPPORTED_SIGNALS,
            display_name="Clickhouse",
            config=ClickhouseObservabilityIntegrationConfig(enabled=enabled),
        )
        integration._transport = transport
        return integration

    @property
    def transport(self) -> ClickhouseObservabilityTransport | None:
        return self._transport

    async def deliver_payload(self, payload: ObservabilityVendorPayload) -> None:
        if self._transport is None:
            msg = "ClickhouseObservabilityIntegration requires an injected transport for delivery"
            raise RuntimeError(msg)
        await self._transport.send_observability_payload(payload)
