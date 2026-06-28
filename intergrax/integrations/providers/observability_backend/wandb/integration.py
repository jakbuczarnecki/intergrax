# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""W&B observability vendor integration (INTEGRATIONS-2C)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.observability import (
    ObservabilityVendorIntegrationConfig,
    ObservabilityVendorIntegrationContract,
    ObservabilityVendorPayload,
    ObservabilityVendorSignal,
)

WANDB_OBSERVABILITY_PROVIDER_ID = "wandb"

_WANDB_SUPPORTED_SIGNALS: tuple[ObservabilityVendorSignal, ...] = (
    ObservabilityVendorSignal.EVENTS,
    ObservabilityVendorSignal.TRACES,
    ObservabilityVendorSignal.LLM_EVENTS,
)

WANDB_SUPPORTED_SIGNALS = _WANDB_SUPPORTED_SIGNALS


class WandbObservabilityIntegrationConfig(ObservabilityVendorIntegrationConfig):
    """Typed config for W&B observability vendor integration."""

    pass


@runtime_checkable
class WandbObservabilityTransport(Protocol):
    """Injectable delivery facade — no vendor SDK or network I/O in the integration class."""

    async def send_observability_payload(self, payload: ObservabilityVendorPayload) -> None:
        """Deliver a policy-sanitized vendor payload to W&B."""


class WandbObservabilityIntegration(ObservabilityVendorIntegrationContract):
    """
    W&B observability vendor integration.

    Consumes only policy-sanitized ObservabilityExportEnvelope records via map_envelope().
    The legacy ObservabilityBackend query facade (create_wandb_observability_backend)
    remains separate and backward-compatible.
    """

    config: WandbObservabilityIntegrationConfig = WandbObservabilityIntegrationConfig()
    _transport: WandbObservabilityTransport | None = PrivateAttr(default=None)

    @classmethod
    def from_transport(
        cls,
        transport: WandbObservabilityTransport,
        *,
        enabled: bool = False,
        supported_signals: tuple[ObservabilityVendorSignal, ...] | None = None,
    ) -> WandbObservabilityIntegration:
        integration = cls.for_provider(
            provider_id=WANDB_OBSERVABILITY_PROVIDER_ID,
            supported_signals=supported_signals or _WANDB_SUPPORTED_SIGNALS,
            display_name="W&B",
            config=WandbObservabilityIntegrationConfig(enabled=enabled),
        )
        integration._transport = transport
        return integration

    @property
    def transport(self) -> WandbObservabilityTransport | None:
        return self._transport

    async def deliver_payload(self, payload: ObservabilityVendorPayload) -> None:
        if self._transport is None:
            msg = "WandbObservabilityIntegration requires an injected transport for delivery"
            raise RuntimeError(msg)
        await self._transport.send_observability_payload(payload)
