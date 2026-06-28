# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""MLflow observability vendor integration (INTEGRATIONS-2C)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.observability import (
    ObservabilityVendorIntegrationConfig,
    ObservabilityVendorIntegrationContract,
    ObservabilityVendorPayload,
    ObservabilityVendorSignal,
)

MLFLOW_OBSERVABILITY_PROVIDER_ID = "mlflow"

_MLFLOW_SUPPORTED_SIGNALS: tuple[ObservabilityVendorSignal, ...] = (
    ObservabilityVendorSignal.EVENTS,
    ObservabilityVendorSignal.LOGS,
    ObservabilityVendorSignal.TRACES,
    ObservabilityVendorSignal.METRICS,
)

MLFLOW_SUPPORTED_SIGNALS = _MLFLOW_SUPPORTED_SIGNALS


class MlflowObservabilityIntegrationConfig(ObservabilityVendorIntegrationConfig):
    """Typed config for MLflow observability vendor integration."""

    pass


@runtime_checkable
class MlflowObservabilityTransport(Protocol):
    """Injectable delivery facade — no vendor SDK or network I/O in the integration class."""

    async def send_observability_payload(self, payload: ObservabilityVendorPayload) -> None:
        """Deliver a policy-sanitized vendor payload to MLflow."""


class MlflowObservabilityIntegration(ObservabilityVendorIntegrationContract):
    """
    MLflow observability vendor integration.

    Consumes only policy-sanitized ObservabilityExportEnvelope records via map_envelope().
    The legacy ObservabilityBackend query facade (create_mlflow_observability_backend)
    remains separate and backward-compatible.
    """

    config: MlflowObservabilityIntegrationConfig = MlflowObservabilityIntegrationConfig()
    _transport: MlflowObservabilityTransport | None = PrivateAttr(default=None)

    @classmethod
    def from_transport(
        cls,
        transport: MlflowObservabilityTransport,
        *,
        enabled: bool = False,
        supported_signals: tuple[ObservabilityVendorSignal, ...] | None = None,
    ) -> MlflowObservabilityIntegration:
        integration = cls.for_provider(
            provider_id=MLFLOW_OBSERVABILITY_PROVIDER_ID,
            supported_signals=supported_signals or _MLFLOW_SUPPORTED_SIGNALS,
            display_name="MLflow",
            config=MlflowObservabilityIntegrationConfig(enabled=enabled),
        )
        integration._transport = transport
        return integration

    @property
    def transport(self) -> MlflowObservabilityTransport | None:
        return self._transport

    async def deliver_payload(self, payload: ObservabilityVendorPayload) -> None:
        if self._transport is None:
            msg = "MlflowObservabilityIntegration requires an injected transport for delivery"
            raise RuntimeError(msg)
        await self._transport.send_observability_payload(payload)
