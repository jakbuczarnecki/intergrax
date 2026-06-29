# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Opensearch observability vendor integration (INTEGRATIONS-2C · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.observability_backend import MetricQueryResult, ObservabilityBackend, TraceQueryResult
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
    """Typed config for Opensearch observability vendor integration."""

    pass


@runtime_checkable
class OpensearchObservabilityTransport(Protocol):
    """Injectable delivery facade — no vendor SDK or network I/O in the integration class."""

    async def send_observability_payload(self, payload: ObservabilityVendorPayload) -> None:
        """Deliver a policy-sanitized vendor payload to Opensearch."""


class OpensearchObservabilityIntegration(ObservabilityVendorIntegrationContract):
    """
    Single public Opensearch observability entrypoint.

    Legacy catalog factory (create_opensearch_observability_backend) delegates to this class via from_backend().
    """

    config: OpensearchObservabilityIntegrationConfig = OpensearchObservabilityIntegrationConfig()
    _transport: OpensearchObservabilityTransport | None = PrivateAttr(default=None)
    _backend: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_backend(
        cls,
        backend: Any,
        *,
        enabled: bool = True,
        supported_signals: tuple[ObservabilityVendorSignal, ...] | None = None,
    ) -> OpensearchObservabilityIntegration:
        signals = supported_signals or OPENSEARCH_SUPPORTED_SIGNALS
        integration = cls.for_provider(
            provider_id=OPENSEARCH_OBSERVABILITY_PROVIDER_ID,
            supported_signals=signals,
            display_name="Opensearch",
            config=OpensearchObservabilityIntegrationConfig(enabled=enabled),
        )
        integration._backend = backend
        return integration

    @property
    def backend(self) -> Any | None:
        return self._backend

    def query_instant(self, promql: str, *, eval_time: float | None = None) -> MetricQueryResult:
        return self._require_runtime().query_instant(promql, eval_time=eval_time)

    def query_range(
        self,
        promql: str,
        *,
        start: float,
        end: float,
        step: str = "15s",
    ) -> MetricQueryResult:
        return self._require_runtime().query_range(
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
        return self._require_runtime().query_traces(limit=limit, name=name)


    def _require_runtime(self) -> Any:
        private = object.__getattribute__(self, "__pydantic_private__")
        runtime = private.get("_runtime")
        if runtime is None:
            runtime = private.get("_backend")
        if runtime is None:
            runtime = private.get("_inner")
        if runtime is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a runtime delegate for catalog operations",
            )
        return runtime


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
            supported_signals=supported_signals or OPENSEARCH_SUPPORTED_SIGNALS,
            display_name="Opensearch",
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

    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            private = object.__getattribute__(self, "__pydantic_private__")
            if name in private:
                return private[name]
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self._require_runtime(), name)

ObservabilityBackend.register(OpensearchObservabilityIntegration)
