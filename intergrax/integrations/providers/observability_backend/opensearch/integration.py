# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""OpenSearch observability vendor integration (INTEGRATIONS-2C)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.observability import (
    ObservabilityVendorIntegrationConfig,
    ObservabilityVendorIntegrationContract,
    ObservabilityVendorPayload,
    ObservabilityVendorSignal,
)

OPENSEARCH_OBSERVABILITY_PROVIDER_ID = "opensearch"

_OPENSEARCH_SUPPORTED_SIGNALS: tuple[ObservabilityVendorSignal, ...] = (
    ObservabilityVendorSignal.EVENTS,
    ObservabilityVendorSignal.LOGS,
    ObservabilityVendorSignal.TRACES,
    ObservabilityVendorSignal.METRICS,
)

OPENSEARCH_SUPPORTED_SIGNALS = _OPENSEARCH_SUPPORTED_SIGNALS


class OpensearchObservabilityIntegrationConfig(ObservabilityVendorIntegrationConfig):
    """Typed config for OpenSearch observability vendor integration."""

    pass


@runtime_checkable
class OpensearchObservabilityTransport(Protocol):
    """Injectable delivery facade — no vendor SDK or network I/O in the integration class."""

    async def send_observability_payload(self, payload: ObservabilityVendorPayload) -> None:
        """Deliver a policy-sanitized vendor payload to OpenSearch."""


class OpensearchObservabilityIntegration(ObservabilityVendorIntegrationContract):
    """
    OpenSearch observability vendor integration.

    Consumes only policy-sanitized ObservabilityExportEnvelope records via map_envelope().
    The legacy ObservabilityBackend query facade (create_opensearch_observability_backend)
    remains separate and backward-compatible.
    """

    config: OpensearchObservabilityIntegrationConfig = OpensearchObservabilityIntegrationConfig()
    _transport: OpensearchObservabilityTransport | None = PrivateAttr(default=None)

    @classmethod
    def from_transport(
        cls,
        transport: OpensearchObservabilityTransport,
        *,
        enabled: bool = False,
        supported_signals: tuple[ObservabilityVendorSignal, ...] | None = None,
    ) -> OpensearchObservabilityIntegration:
        integration = cls.for_provider(
            provider_id=OPENSEARCH_OBSERVABILITY_PROVIDER_ID,
            supported_signals=supported_signals or _OPENSEARCH_SUPPORTED_SIGNALS,
            display_name="OpenSearch",
            config=OpensearchObservabilityIntegrationConfig(enabled=enabled),
        )
        integration._transport = transport
        return integration

    @property
    def transport(self) -> OpensearchObservabilityTransport | None:
        return self._transport

    async def deliver_payload(self, payload: ObservabilityVendorPayload) -> None:
        if self._transport is None:
            msg = "OpensearchObservabilityIntegration requires an injected transport for delivery"
            raise RuntimeError(msg)
        await self._transport.send_observability_payload(payload)
