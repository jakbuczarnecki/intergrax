# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Langsmith observability vendor integration (INTEGRATIONS-2C)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.observability import (
    ObservabilityVendorIntegrationConfig,
    ObservabilityVendorIntegrationContract,
    ObservabilityVendorPayload,
    ObservabilityVendorSignal,
)

LANGSMITH_OBSERVABILITY_PROVIDER_ID = "langsmith"

_LANGSMITH_SUPPORTED_SIGNALS: tuple[ObservabilityVendorSignal, ...] = (
    ObservabilityVendorSignal.EVENTS,
    ObservabilityVendorSignal.TRACES,
    ObservabilityVendorSignal.LLM_EVENTS,
)

LANGSMITH_SUPPORTED_SIGNALS = _LANGSMITH_SUPPORTED_SIGNALS


class LangsmithObservabilityIntegrationConfig(ObservabilityVendorIntegrationConfig):
    """Typed config for Langsmith observability vendor integration."""

    pass


@runtime_checkable
class LangsmithObservabilityTransport(Protocol):
    """Injectable delivery facade — no vendor SDK or network I/O in the integration class."""

    async def send_observability_payload(self, payload: ObservabilityVendorPayload) -> None:
        """Deliver a policy-sanitized vendor payload to Langsmith."""


class LangsmithObservabilityIntegration(ObservabilityVendorIntegrationContract):
    """
    Langsmith observability vendor integration.

    Consumes only policy-sanitized ObservabilityExportEnvelope records via map_envelope().
    The legacy ObservabilityBackend query facade (create_langsmith_observability_backend)
    remains separate and backward-compatible.
    """

    config: LangsmithObservabilityIntegrationConfig = LangsmithObservabilityIntegrationConfig()
    _transport: LangsmithObservabilityTransport | None = PrivateAttr(default=None)

    @classmethod
    def from_transport(
        cls,
        transport: LangsmithObservabilityTransport,
        *,
        enabled: bool = False,
        supported_signals: tuple[ObservabilityVendorSignal, ...] | None = None,
    ) -> LangsmithObservabilityIntegration:
        integration = cls.for_provider(
            provider_id=LANGSMITH_OBSERVABILITY_PROVIDER_ID,
            supported_signals=supported_signals or _LANGSMITH_SUPPORTED_SIGNALS,
            display_name="Langsmith",
            config=LangsmithObservabilityIntegrationConfig(enabled=enabled),
        )
        integration._transport = transport
        return integration

    @property
    def transport(self) -> LangsmithObservabilityTransport | None:
        return self._transport

    async def deliver_payload(self, payload: ObservabilityVendorPayload) -> None:
        if self._transport is None:
            msg = "LangsmithObservabilityIntegration requires an injected transport for delivery"
            raise RuntimeError(msg)
        await self._transport.send_observability_payload(payload)
