# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Elasticsearch observability delivery error classification (OBS-VENDOR-6A)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from intergrax.integrations.providers.observability_backend.elasticsearch.client import (
    ElasticsearchDeliveryError,
    ElasticsearchDeliveryErrorDetail,
    ElasticsearchRestClient,
    classify_elasticsearch_delivery_error,
)
from intergrax.integrations.providers.observability_backend.elasticsearch.config import ElasticsearchIntegrationConfig
from intergrax.integrations.providers.observability_backend.elasticsearch.config import (
    ElasticsearchRetryPolicy,
)
from intergrax.integrations.providers.observability_backend.elasticsearch.transport import (
    ElasticsearchHttpObservabilityTransport,
    map_vendor_payload_to_elasticsearch_document,
)
from intergrax.runtime.integrations.observability import ObservabilityVendorPayload

pytestmark = pytest.mark.unit

_UNSAFE_PROMPT = "RAW_PROMPT_DO_NOT_LEAK_abc123"
_UNSAFE_SECRET = "secret-api-key-should-not-appear"
_UNSAFE_PATH = "C:\\Users\\secret\\document.pdf"


def _elasticsearch_config() -> ElasticsearchIntegrationConfig:
    return ElasticsearchIntegrationConfig(
        base_url="http://elasticsearch.local:9200",
        index="observability-events",
    )


def _http_status_error(status_code: int, *, response_text: str = "") -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://elasticsearch.local:9200/observability-events/_doc")
    response = httpx.Response(
        status_code,
        request=request,
        text=response_text or f'{{"error":"status {status_code}"}}',
    )
    return httpx.HTTPStatusError("delivery failed", request=request, response=response)


def _vendor_payload() -> ObservabilityVendorPayload:
    return ObservabilityVendorPayload(
        provider_id="elasticsearch",
        integration_id="elasticsearch",
        integration_kind="observability_backend",
        record_type="runtime_event",
        recorded_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        run_id="run-1",
        event_id="event-1",
    )


def _unsafe_document() -> dict[str, Any]:
    return {
        "@timestamp": "2026-01-01T00:00:00+00:00",
        "intergrax.run_id": "run-1",
        "intergrax.prompt": _UNSAFE_PROMPT,
        "intergrax.secret": _UNSAFE_SECRET,
        "intergrax.path": _UNSAFE_PATH,
    }


def test_index_document_success_still_works() -> None:
    http = MagicMock()
    response = MagicMock()
    response.json.return_value = {"_id": "doc-42"}
    response.raise_for_status.return_value = None
    http.post.return_value = response
    client = ElasticsearchRestClient(_elasticsearch_config(), http_client=http)

    doc_id = client.index_document(
        index="observability-events",
        document={"intergrax.run_id": "run-1"},
    )

    assert doc_id == "doc-42"
    http.post.assert_called_once()


@pytest.mark.parametrize("status_code", [429, 503])
def test_http_rate_limit_and_unavailable_are_retriable(status_code: int) -> None:
    http = MagicMock()
    http.post.side_effect = _http_status_error(status_code)
    client = ElasticsearchRestClient(_elasticsearch_config(), http_client=http)

    with pytest.raises(ElasticsearchDeliveryError) as exc_info:
        client.index_document(index="observability-events", document={"intergrax.run_id": "run-1"})

    detail = exc_info.value.detail
    assert detail.status_code == status_code
    assert detail.retriable is True
    assert detail.operation == "index_document"
    assert detail.index == "observability-events"
    assert detail.provider_id == "elasticsearch"


def test_http_400_is_non_retriable() -> None:
    http = MagicMock()
    http.post.side_effect = _http_status_error(400, response_text='{"error":"mapper_parsing_exception"}')
    client = ElasticsearchRestClient(_elasticsearch_config(), http_client=http)

    with pytest.raises(ElasticsearchDeliveryError) as exc_info:
        client.index_document(index="observability-events", document={"intergrax.run_id": "run-1"})

    detail = exc_info.value.detail
    assert detail.status_code == 400
    assert detail.retriable is False
    assert detail.reason == "http_status_400"


@pytest.mark.parametrize(
    ("side_effect", "expected_reason"),
    [
        (httpx.ReadTimeout("read timeout"), "timeout"),
        (httpx.ConnectError("connection refused"), "connection_error"),
    ],
)
def test_timeout_and_connection_errors_are_retriable(
    side_effect: Exception,
    expected_reason: str,
) -> None:
    http = MagicMock()
    http.post.side_effect = side_effect
    client = ElasticsearchRestClient(_elasticsearch_config(), http_client=http)

    with pytest.raises(ElasticsearchDeliveryError) as exc_info:
        client.index_document(index="observability-events", document={"intergrax.run_id": "run-1"})

    detail = exc_info.value.detail
    assert detail.retriable is True
    assert detail.reason == expected_reason
    assert detail.status_code is None


def test_safe_diagnostic_detail_does_not_include_document_body() -> None:
    detail = ElasticsearchDeliveryErrorDetail(
        provider_id="elasticsearch",
        operation="index_document",
        index="observability-events",
        status_code=503,
        reason="http_status_503",
        retriable=True,
    )
    rendered = str(detail)

    for unsafe in (_UNSAFE_PROMPT, _UNSAFE_SECRET, _UNSAFE_PATH, "prompt", "chunks"):
        assert unsafe not in rendered


def test_exception_message_does_not_leak_unsafe_document_fields() -> None:
    http = MagicMock()
    http.post.side_effect = _http_status_error(400, response_text=f'failed to index {_UNSAFE_PROMPT}')
    client = ElasticsearchRestClient(_elasticsearch_config(), http_client=http)

    with pytest.raises(ElasticsearchDeliveryError) as exc_info:
        client.index_document(index="observability-events", document=_unsafe_document())

    message = str(exc_info.value)
    for unsafe in (_UNSAFE_PROMPT, _UNSAFE_SECRET, _UNSAFE_PATH):
        assert unsafe not in message
    assert exc_info.value.detail.reason == "http_status_400"


class RecordingIndexer:
    def __init__(self, *, error: BaseException | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._error = error

    def index_document(
        self,
        *,
        index: str,
        document: dict[str, Any],
        doc_id: str | None = None,
    ) -> str:
        self.calls.append({"index": index, "document": dict(document), "doc_id": doc_id})
        if self._error is not None:
            raise self._error
        return "generated-id"


@pytest.mark.asyncio
async def test_transport_maps_payload_before_delivery_on_failure() -> None:
    payload = _vendor_payload()
    expected_document = map_vendor_payload_to_elasticsearch_document(payload)
    indexer = RecordingIndexer(error=_http_status_error(503))
    transport = ElasticsearchHttpObservabilityTransport(
        indexer,
        index="observability-events",
        retry_policy=ElasticsearchRetryPolicy(max_attempts=1),
    )

    with pytest.raises(ElasticsearchDeliveryError) as exc_info:
        await transport.send_observability_payload(payload)

    assert len(indexer.calls) == 1
    assert indexer.calls[0]["document"] == expected_document
    assert indexer.calls[0]["index"] == "observability-events"
    assert exc_info.value.detail.operation == "send_observability_payload"
    assert exc_info.value.detail.retriable is True


def test_classify_elasticsearch_delivery_error_is_idempotent() -> None:
    original = classify_elasticsearch_delivery_error(
        _http_status_error(429),
        operation="index_document",
        index="observability-events",
    )

    classified = classify_elasticsearch_delivery_error(
        original,
        operation="index_document",
        index="observability-events",
    )

    assert classified is original


@pytest.mark.asyncio
async def test_transport_propagates_client_classified_error_without_rewrap() -> None:
    client_error = ElasticsearchDeliveryError(
        ElasticsearchDeliveryErrorDetail(
            provider_id="elasticsearch",
            operation="index_document",
            index="observability-events",
            status_code=503,
            reason="http_status_503",
            retriable=True,
        )
    )
    indexer = RecordingIndexer(error=client_error)
    transport = ElasticsearchHttpObservabilityTransport(
        indexer,
        index="observability-events",
        retry_policy=ElasticsearchRetryPolicy(max_attempts=1),
    )

    with pytest.raises(ElasticsearchDeliveryError) as exc_info:
        await transport.send_observability_payload(_vendor_payload())

    assert exc_info.value.detail.operation == "send_observability_payload"
    assert exc_info.value.detail.status_code == 503
    assert exc_info.value.detail.retriable is True


def test_index_document_success_path_uses_asyncio_to_thread_wrapper() -> None:
    indexer = RecordingIndexer()
    transport = ElasticsearchHttpObservabilityTransport(indexer, index="observability-events")

    asyncio.run(transport.send_observability_payload(_vendor_payload()))

    assert len(indexer.calls) == 1
    assert indexer.calls[0]["document"]["intergrax.run_id"] == "run-1"
