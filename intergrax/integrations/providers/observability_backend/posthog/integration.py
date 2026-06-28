# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Posthog observability vendor integration (INTEGRATIONS-2C)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.observability import (
    ObservabilityVendorIntegrationConfig,
    ObservabilityVendorIntegrationContract,
    ObservabilityVendorPayload,
    ObservabilityVendorSignal,
)

POSTHOG_OBSERVABILITY_PROVIDER_ID = "posthog"

_POSTHOG_SUPPORTED_SIGNALS: tuple[ObservabilityVendorSignal, ...] = (
    ObservabilityVendorSignal.EVENTS,
    ObservabilityVendorSignal.LOGS,
    ObservabilityVendorSignal.TRACES,
    ObservabilityVendorSignal.METRICS,
)

POSTHOG_SUPPORTED_SIGNALS = _POSTHOG_SUPPORTED_SIGNALS


class PosthogObservabilityIntegrationConfig(ObservabilityVendorIntegrationConfig):
    """Typed config for Posthog observability vendor integration."""

    pass


@runtime_checkable
class PosthogObservabilityTransport(Protocol):
    """Injectable delivery facade — no vendor SDK or network I/O in the integration class."""

    async def send_observability_payload(self, payload: ObservabilityVendorPayload) -> None:
        """Deliver a policy-sanitized vendor payload to Posthog."""


class PosthogObservabilityIntegration(ObservabilityVendorIntegrationContract):
    """
    Posthog observability vendor integration.

    Consumes only policy-sanitized ObservabilityExportEnvelope records via map_envelope().
    The legacy ObservabilityBackend query facade (create_posthog_observability_backend)
    remains separate and backward-compatible.
    """

    config: PosthogObservabilityIntegrationConfig = PosthogObservabilityIntegrationConfig()
    _transport: PosthogObservabilityTransport | None = PrivateAttr(default=None)

    @classmethod
    def from_transport(
        cls,
        transport: PosthogObservabilityTransport,
        *,
        enabled: bool = False,
        supported_signals: tuple[ObservabilityVendorSignal, ...] | None = None,
    ) -> PosthogObservabilityIntegration:
        integration = cls.for_provider(
            provider_id=POSTHOG_OBSERVABILITY_PROVIDER_ID,
            supported_signals=supported_signals or _POSTHOG_SUPPORTED_SIGNALS,
            display_name="Posthog",
            config=PosthogObservabilityIntegrationConfig(enabled=enabled),
        )
        integration._transport = transport
        return integration

    @property
    def transport(self) -> PosthogObservabilityTransport | None:
        return self._transport

    async def deliver_payload(self, payload: ObservabilityVendorPayload) -> None:
        if self._transport is None:
            msg = "PosthogObservabilityIntegration requires an injected transport for delivery"
            raise RuntimeError(msg)
        await self._transport.send_observability_payload(payload)
