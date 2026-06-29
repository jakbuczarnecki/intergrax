# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Mlflow observability vendor integration (INTEGRATIONS-2C · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.observability_backend._catalog_client import (
    ObservabilityCatalogClient,
    require_observability_catalog_client,
)
from intergrax.integrations.contracts.observability_backend import MetricQueryResult, ObservabilityBackend, TraceQueryResult
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
    """Typed config for Mlflow observability vendor integration."""

    pass


@runtime_checkable
class MlflowObservabilityTransport(Protocol):
    """Injectable delivery facade — no vendor SDK or network I/O in the integration class."""

    async def send_observability_payload(self, payload: ObservabilityVendorPayload) -> None:
        """Deliver a policy-sanitized vendor payload to Mlflow."""


class MlflowObservabilityIntegration(ObservabilityVendorIntegrationContract):
    """
    Single public Mlflow observability entrypoint.

    Legacy catalog factory (create_mlflow_observability_backend) owns catalog query behavior; legacy factories use from_client().
    """

    config: MlflowObservabilityIntegrationConfig = MlflowObservabilityIntegrationConfig()
    _transport: MlflowObservabilityTransport | None = PrivateAttr(default=None)
    _client: ObservabilityCatalogClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: ObservabilityCatalogClient,
        *,
        enabled: bool = True,
        supported_signals: tuple[ObservabilityVendorSignal, ...] | None = None,
    ) -> MlflowObservabilityIntegration:
        signals = supported_signals or MLFLOW_SUPPORTED_SIGNALS
        integration = cls.for_provider(
            provider_id=MLFLOW_OBSERVABILITY_PROVIDER_ID,
            supported_signals=signals,
            display_name="Mlflow",
            config=MlflowObservabilityIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> ObservabilityCatalogClient | None:
        return self._client

    def query_instant(self, promql: str, *, eval_time: float | None = None) -> MetricQueryResult:
        return self._require_client().query_instant(promql, eval_time=eval_time)

    def query_range(
        self,
        promql: str,
        *,
        start: float,
        end: float,
        step: str = "15s",
    ) -> MetricQueryResult:
        return self._require_client().query_range(
            promql,
            start=start,
            end=end,
            step=step,
        )

    def query_traces(
        self,
        *,
        limit: int = 20,
        name: str | None = None,
    ) -> TraceQueryResult:
        return self._require_client().query_traces(limit=limit, name=name)


    def _require_client(self) -> ObservabilityCatalogClient:
        return require_observability_catalog_client(self, self._client)


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
            supported_signals=supported_signals or MLFLOW_SUPPORTED_SIGNALS,
            display_name="Mlflow",
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


ObservabilityBackend.register(MlflowObservabilityIntegration)
