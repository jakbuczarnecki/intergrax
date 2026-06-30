# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Elasticsearch/OpenSearch observability export transport (OBS-VENDOR-4A)."""

from __future__ import annotations

import asyncio
from typing import Any, Mapping, Protocol, runtime_checkable

from intergrax.integrations.providers.observability_backend.elasticsearch.client import (
    ElasticsearchDeliveryError,
    ElasticsearchRestClient,
    classify_elasticsearch_delivery_error,
)
from intergrax.integrations.providers.observability_backend.elasticsearch.config import DEFAULT_TIMESTAMP_FIELD
from intergrax.runtime.integrations.observability import ObservabilityVendorPayload
from intergrax.runtime.observability.export_attributes import ObservabilityAttributeValue

_INTERGRAX_DOC_PREFIX = "intergrax."


def _set_optional_string(doc: dict[str, Any], key: str, value: str) -> None:
    if value:
        doc[key] = value


def _attribute_value_to_document(value: ObservabilityAttributeValue) -> Any:
    if isinstance(value, list):
        return list(value)
    return value


def map_vendor_payload_to_elasticsearch_document(
    payload: ObservabilityVendorPayload,
    *,
    timestamp_field: str = DEFAULT_TIMESTAMP_FIELD,
) -> dict[str, Any]:
    """Map a policy-safe vendor payload to an Elasticsearch/OpenSearch index document."""
    prefix = _INTERGRAX_DOC_PREFIX
    document: dict[str, Any] = {
        timestamp_field: payload.recorded_at.isoformat(),
        f"{prefix}schema_id": payload.schema_id,
        f"{prefix}provider_id": payload.provider_id,
        f"{prefix}integration_id": payload.integration_id,
        f"{prefix}integration_kind": payload.integration_kind,
        f"{prefix}record_type": payload.record_type,
    }
    _set_optional_string(document, f"{prefix}run_id", payload.run_id)
    _set_optional_string(document, f"{prefix}task_id", payload.task_id)
    _set_optional_string(document, f"{prefix}agent_id", payload.agent_id)
    _set_optional_string(document, f"{prefix}capability", payload.capability)
    _set_optional_string(document, f"{prefix}event_type", payload.event_type)
    _set_optional_string(document, f"{prefix}status", payload.status)
    _set_optional_string(document, f"{prefix}tool_id", payload.tool_id)
    _set_optional_string(document, f"{prefix}artifact_ref", payload.artifact_ref)
    _set_optional_string(document, f"{prefix}sha256", payload.sha256)
    _set_optional_string(document, f"{prefix}safe_relative_path", payload.safe_relative_path)
    _set_optional_string(document, f"{prefix}schema_id_source", payload.schema_id_source)
    _set_optional_string(document, f"{prefix}tenant_id", payload.tenant_id)
    _set_optional_string(document, f"{prefix}workspace_id", payload.workspace_id)
    _set_optional_string(document, f"{prefix}source_schema_id", payload.source_schema_id)
    _set_optional_string(document, f"{prefix}correlation_id", payload.correlation_id)
    _set_optional_string(document, f"{prefix}event_id", payload.event_id)
    if payload.latency_ms is not None:
        document[f"{prefix}latency_ms"] = payload.latency_ms
    if payload.counts:
        document[f"{prefix}counts"] = dict(payload.counts)
    sanitized = payload.sanitized_application_attributes
    if sanitized is not None:
        if sanitized.namespace:
            document[f"{prefix}application.namespace"] = sanitized.namespace
        for key, value in sorted(sanitized.attributes.items()):
            document[key] = _attribute_value_to_document(value)
    return document


@runtime_checkable
class ElasticsearchObservabilityIndexer(Protocol):
    """Provider-owned indexer facade used by the observability transport."""

    def index_document(
        self,
        *,
        index: str,
        document: Mapping[str, Any],
        doc_id: str | None = None,
    ) -> str:
        """Index one observability document."""


class ElasticsearchHttpObservabilityTransport:
    """Index policy-safe observability payloads via provider-owned HTTP client."""

    def __init__(
        self,
        client: ElasticsearchObservabilityIndexer,
        *,
        index: str | None = None,
        timestamp_field: str | None = None,
    ) -> None:
        self._client = client
        if isinstance(client, ElasticsearchRestClient):
            self._index = index or client.config.index
            self._timestamp_field = timestamp_field or client.config.timestamp_field
        else:
            if index is None:
                msg = "index is required when client is not ElasticsearchRestClient"
                raise ValueError(msg)
            self._index = index
            self._timestamp_field = timestamp_field or DEFAULT_TIMESTAMP_FIELD

    async def send_observability_payload(self, payload: ObservabilityVendorPayload) -> None:
        document = map_vendor_payload_to_elasticsearch_document(
            payload,
            timestamp_field=self._timestamp_field,
        )
        try:
            await asyncio.to_thread(
                self._client.index_document,
                index=self._index,
                document=document,
            )
        except ElasticsearchDeliveryError:
            raise
        except Exception as exc:
            raise classify_elasticsearch_delivery_error(
                exc,
                operation="send_observability_payload",
                index=self._index,
            ) from exc
