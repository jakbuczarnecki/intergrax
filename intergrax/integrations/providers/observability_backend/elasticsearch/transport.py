# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Elasticsearch/OpenSearch observability export transport (OBS-VENDOR-4A)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

from intergrax.integrations.providers.observability_backend.elasticsearch.client import (
    ElasticsearchDeliveryError,
    ElasticsearchDeliveryErrorDetail,
    ElasticsearchRestClient,
    classify_elasticsearch_delivery_error,
)
from intergrax.integrations.providers.observability_backend.elasticsearch.config import (
    DEFAULT_TIMESTAMP_FIELD,
    ElasticsearchRetryPolicy,
)
from intergrax.runtime.integrations.observability import ObservabilityVendorPayload
from intergrax.runtime.observability.export_attributes import ObservabilityAttributeValue

_INTERGRAX_DOC_PREFIX = "intergrax."
_ELASTICSEARCH_OBSERVABILITY_PROVIDER_ID = "elasticsearch"
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ElasticsearchFailedDeliveryRecord:
    """Safe provider-owned diagnostics for ultimately failed observability delivery."""

    provider_id: str
    operation: str
    index: str
    status_code: int | None
    reason: str
    retriable: bool
    attempts: int
    exhausted: bool


@runtime_checkable
class ElasticsearchFailedDeliverySink(Protocol):
    """Provider-owned hook invoked when observability delivery ultimately fails."""

    def record_failed_delivery(self, record: ElasticsearchFailedDeliveryRecord) -> None:
        """Record one failed delivery using safe diagnostic metadata only."""


class NoOpElasticsearchFailedDeliverySink:
    """Default failed-delivery sink that discards records."""

    def record_failed_delivery(self, record: ElasticsearchFailedDeliveryRecord) -> None:
        return None


def _failed_delivery_record_from_error(
    error: ElasticsearchDeliveryError,
    *,
    attempts: int,
    exhausted: bool,
) -> ElasticsearchFailedDeliveryRecord:
    detail = error.detail
    return ElasticsearchFailedDeliveryRecord(
        provider_id=detail.provider_id,
        operation=detail.operation,
        index=detail.index,
        status_code=detail.status_code,
        reason=detail.reason,
        retriable=detail.retriable,
        attempts=attempts,
        exhausted=exhausted,
    )


def compute_elasticsearch_retry_backoff_seconds(
    *,
    retry_after_failure_number: int,
    previous_backoff_seconds: float,
    policy: ElasticsearchRetryPolicy,
) -> float:
    """Return sleep duration before the next attempt after a retriable failure."""
    if retry_after_failure_number <= 0:
        return 0.0
    if retry_after_failure_number == 1:
        return policy.initial_backoff_seconds
    return min(previous_backoff_seconds * 2, policy.max_backoff_seconds)


def _transport_delivery_error(error: ElasticsearchDeliveryError) -> ElasticsearchDeliveryError:
    detail = error.detail
    if detail.operation == "send_observability_payload":
        return error
    return ElasticsearchDeliveryError(
        ElasticsearchDeliveryErrorDetail(
            provider_id=detail.provider_id,
            operation="send_observability_payload",
            index=detail.index,
            status_code=detail.status_code,
            reason=detail.reason,
            retriable=detail.retriable,
        )
    )


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
        retry_policy: ElasticsearchRetryPolicy | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        failed_delivery_sink: ElasticsearchFailedDeliverySink | None = None,
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
        self._retry_policy = retry_policy or ElasticsearchRetryPolicy()
        self._sleep = sleep or asyncio.sleep
        self._failed_delivery_sink = (
            failed_delivery_sink if failed_delivery_sink is not None else NoOpElasticsearchFailedDeliverySink()
        )

    def _invoke_failed_delivery_sink(
        self,
        error: ElasticsearchDeliveryError,
        *,
        attempts: int,
        exhausted: bool,
    ) -> None:
        record = _failed_delivery_record_from_error(
            error,
            attempts=attempts,
            exhausted=exhausted,
        )
        try:
            self._failed_delivery_sink.record_failed_delivery(record)
        except Exception:
            _LOGGER.exception(
                "Elasticsearch failed-delivery sink raised; original delivery error is preserved",
            )

    async def send_observability_payload(self, payload: ObservabilityVendorPayload) -> None:
        document = map_vendor_payload_to_elasticsearch_document(
            payload,
            timestamp_field=self._timestamp_field,
        )
        policy = self._retry_policy
        max_attempts = policy.effective_max_attempts()
        previous_backoff_seconds = 0.0
        last_error: ElasticsearchDeliveryError | None = None

        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                retry_after_failure_number = attempt - 1
                delay = compute_elasticsearch_retry_backoff_seconds(
                    retry_after_failure_number=retry_after_failure_number,
                    previous_backoff_seconds=previous_backoff_seconds,
                    policy=policy,
                )
                await self._sleep(delay)
                previous_backoff_seconds = delay

            try:
                await asyncio.to_thread(
                    self._client.index_document,
                    index=self._index,
                    document=document,
                )
                return
            except ElasticsearchDeliveryError as exc:
                transport_error = _transport_delivery_error(exc)
                if not transport_error.detail.retriable:
                    self._invoke_failed_delivery_sink(
                        transport_error,
                        attempts=attempt,
                        exhausted=True,
                    )
                    raise transport_error
                last_error = transport_error
            except Exception as exc:
                classified = classify_elasticsearch_delivery_error(
                    exc,
                    operation="send_observability_payload",
                    index=self._index,
                    provider_id=_ELASTICSEARCH_OBSERVABILITY_PROVIDER_ID,
                )
                if not classified.detail.retriable:
                    self._invoke_failed_delivery_sink(
                        classified,
                        attempts=attempt,
                        exhausted=True,
                    )
                    raise classified
                last_error = classified

            if attempt >= max_attempts and last_error is not None:
                final_error = _transport_delivery_error(last_error)
                self._invoke_failed_delivery_sink(
                    final_error,
                    attempts=attempt,
                    exhausted=True,
                )
                raise final_error
