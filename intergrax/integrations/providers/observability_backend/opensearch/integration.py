# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Opensearch observability vendor integration (INTEGRATIONS-2C · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

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
from intergrax.utils import attribute_access


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

    Legacy catalog factory (create_opensearch_observability_backend) owns catalog query behavior; legacy factories use from_client().
    """

    config: OpensearchObservabilityIntegrationConfig = OpensearchObservabilityIntegrationConfig()
    _transport: OpensearchObservabilityTransport | None = PrivateAttr(default=None)
    _client: ObservabilityCatalogClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: ObservabilityCatalogClient,
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

    @property
    def rest_client(self) -> ObservabilityCatalogClient:
        return self._require_client()

    def index_document(
        self,
        *,
        index: str,
        document: Mapping[str, Any],
        doc_id: str | None = None,
    ) -> str:
        client = self._require_client()
        index_document = attribute_access.optional(client, "index_document", None)
        if not callable(index_document):
            raise IntegrationConfigurationError(
                f"{type(self).__name__} catalog client does not support index_document",
            )
        return str(index_document(index=index, document=document, doc_id=doc_id))

    def ensure_index(self, index: str) -> bool:
        client = self._require_client()
        ensure_index = attribute_access.optional(client, "ensure_index", None)
        if not callable(ensure_index):
            raise IntegrationConfigurationError(
                f"{type(self).__name__} catalog client does not support ensure_index",
            )
        return bool(ensure_index(index))

    def _require_client(self) -> ObservabilityCatalogClient:
        return require_observability_catalog_client(self, self._client)


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


ObservabilityBackend.register(OpensearchObservabilityIntegration)
