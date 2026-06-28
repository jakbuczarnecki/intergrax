# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Langfuse observability vendor integration (INTEGRATIONS-2B pilot)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.observability import (
    ObservabilityVendorIntegrationConfig,
    ObservabilityVendorIntegrationContract,
    ObservabilityVendorPayload,
    ObservabilityVendorSignal,
)

LANGFUSE_OBSERVABILITY_PROVIDER_ID = "langfuse"

_LANGFUSE_SUPPORTED_SIGNALS: tuple[ObservabilityVendorSignal, ...] = (
    ObservabilityVendorSignal.EVENTS,
    ObservabilityVendorSignal.TRACES,
    ObservabilityVendorSignal.LLM_EVENTS,
)

LANGFUSE_SUPPORTED_SIGNALS = _LANGFUSE_SUPPORTED_SIGNALS


class LangfuseObservabilityIntegrationConfig(ObservabilityVendorIntegrationConfig):
    """Typed config for Langfuse observability vendor integration."""

    pass


@runtime_checkable
class LangfuseObservabilityTransport(Protocol):
    """Injectable delivery facade — no vendor SDK or network I/O in the integration class."""

    async def send_observability_payload(self, payload: ObservabilityVendorPayload) -> None:
        """Deliver a policy-sanitized vendor payload to Langfuse."""


class LangfuseObservabilityIntegration(ObservabilityVendorIntegrationContract):
    """
    Langfuse observability vendor integration.

    Consumes only policy-sanitized ObservabilityExportEnvelope records via map_envelope().
    The legacy ObservabilityBackend query facade (create_langfuse_observability_backend)
    remains separate and backward-compatible.
    """

    config: LangfuseObservabilityIntegrationConfig = LangfuseObservabilityIntegrationConfig()
    _transport: LangfuseObservabilityTransport | None = PrivateAttr(default=None)

    @classmethod
    def from_transport(
        cls,
        transport: LangfuseObservabilityTransport,
        *,
        enabled: bool = False,
        supported_signals: tuple[ObservabilityVendorSignal, ...] | None = None,
    ) -> LangfuseObservabilityIntegration:
        integration = cls.for_provider(
            provider_id=LANGFUSE_OBSERVABILITY_PROVIDER_ID,
            supported_signals=supported_signals or _LANGFUSE_SUPPORTED_SIGNALS,
            display_name="Langfuse",
            config=LangfuseObservabilityIntegrationConfig(enabled=enabled),
        )
        integration._transport = transport
        return integration

    @property
    def transport(self) -> LangfuseObservabilityTransport | None:
        return self._transport

    async def deliver_payload(self, payload: ObservabilityVendorPayload) -> None:
        if self._transport is None:
            msg = "LangfuseObservabilityIntegration requires an injected transport for delivery"
            raise RuntimeError(msg)
        await self._transport.send_observability_payload(payload)
