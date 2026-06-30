# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Elasticsearch observability retry/backoff (OBS-VENDOR-6B)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx
import pytest

from intergrax.integrations.providers.observability_backend.elasticsearch.client import (
    ElasticsearchDeliveryError,
    ElasticsearchDeliveryErrorDetail,
    classify_elasticsearch_delivery_error,
)
from intergrax.integrations.providers.observability_backend.elasticsearch.config import (
    ElasticsearchRetryPolicy,
)
from intergrax.integrations.providers.observability_backend.elasticsearch.transport import (
    ElasticsearchHttpObservabilityTransport,
    compute_elasticsearch_retry_backoff_seconds,
)
from intergrax.runtime.integrations.observability import ObservabilityVendorPayload

pytestmark = pytest.mark.unit

_UNSAFE_PROMPT = "RAW_PROMPT_DO_NOT_LEAK_abc123"
_UNSAFE_SECRET = "secret-api-key-should-not-appear"
_UNSAFE_PATH = "C:\\Users\\secret\\document.pdf"


def _http_status_error(status_code: int, *, response_text: str = "") -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://elasticsearch.local:9200/observability-events/_doc")
    response = httpx.Response(
        status_code,
        request=request,
        text=response_text or f'{{"error":"status {status_code}"}}',
    )
    return httpx.HTTPStatusError("delivery failed", request=request, response=response)


def _delivery_error(
    *,
    status_code: int | None,
    retriable: bool,
    reason: str,
) -> ElasticsearchDeliveryError:
    return ElasticsearchDeliveryError(
        ElasticsearchDeliveryErrorDetail(
            provider_id="elasticsearch",
            operation="index_document",
            index="observability-events",
            status_code=status_code,
            reason=reason,
            retriable=retriable,
        )
    )


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


class RecordingSleep:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


class FlakyIndexer:
    def __init__(self, outcomes: list[BaseException | str]) -> None:
        self._outcomes = list(outcomes)
        self.calls = 0
        self.documents: list[dict[str, Any]] = []

    def index_document(
        self,
        *,
        index: str,
        document: dict[str, Any],
        doc_id: str | None = None,
    ) -> str:
        self.calls += 1
        self.documents.append(dict(document))
        if not self._outcomes:
            return "generated-id"
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _transport(
    indexer: FlakyIndexer,
    *,
    retry_policy: ElasticsearchRetryPolicy | None = None,
    sleep: RecordingSleep | None = None,
) -> tuple[ElasticsearchHttpObservabilityTransport, RecordingSleep]:
    active_sleep = sleep or RecordingSleep()
    transport = ElasticsearchHttpObservabilityTransport(
        indexer,
        index="observability-events",
        retry_policy=retry_policy,
        sleep=active_sleep,
    )
    return transport, active_sleep


@pytest.mark.asyncio
async def test_success_on_first_attempt_calls_index_document_once() -> None:
    indexer = FlakyIndexer(["doc-1"])
    transport, sleep = _transport(indexer)

    await transport.send_observability_payload(_vendor_payload())

    assert indexer.calls == 1
    assert sleep.delays == []


@pytest.mark.asyncio
async def test_http_503_retriable_error_retries_and_then_succeeds() -> None:
    indexer = FlakyIndexer(
        [
            _delivery_error(status_code=503, retriable=True, reason="http_status_503"),
            _delivery_error(status_code=503, retriable=True, reason="http_status_503"),
            "doc-1",
        ]
    )
    transport, sleep = _transport(indexer)

    await transport.send_observability_payload(_vendor_payload())

    assert indexer.calls == 3
    assert sleep.delays == [0.25, 0.5]


@pytest.mark.asyncio
async def test_http_429_retriable_error_retries_and_then_succeeds() -> None:
    indexer = FlakyIndexer(
        [
            _delivery_error(status_code=429, retriable=True, reason="http_status_429"),
            "doc-1",
        ]
    )
    transport, sleep = _transport(indexer)

    await transport.send_observability_payload(_vendor_payload())

    assert indexer.calls == 2
    assert sleep.delays == [0.25]


@pytest.mark.asyncio
async def test_timeout_and_connection_retriable_errors_retry() -> None:
    for side_effect in (
        httpx.ReadTimeout("read timeout"),
        httpx.ConnectError("connection refused"),
    ):
        indexer = FlakyIndexer([side_effect, "doc-1"])
        transport, sleep = _transport(indexer)

        await transport.send_observability_payload(_vendor_payload())

        assert indexer.calls == 2
        assert sleep.delays == [0.25]


@pytest.mark.asyncio
async def test_http_400_non_retriable_error_normalizes_operation_and_does_not_retry() -> None:
    indexer = FlakyIndexer(
        [_delivery_error(status_code=400, retriable=False, reason="http_status_400")]
    )
    transport, sleep = _transport(indexer)

    with pytest.raises(ElasticsearchDeliveryError) as exc_info:
        await transport.send_observability_payload(_vendor_payload())

    assert indexer.calls == 1
    assert sleep.delays == []
    assert exc_info.value.detail.operation == "send_observability_payload"
    assert exc_info.value.detail.retriable is False


@pytest.mark.asyncio
async def test_retry_enabled_false_does_not_retry() -> None:
    indexer = FlakyIndexer(
        [_delivery_error(status_code=503, retriable=True, reason="http_status_503")]
    )
    transport, sleep = _transport(
        indexer,
        retry_policy=ElasticsearchRetryPolicy(enabled=False),
    )

    with pytest.raises(ElasticsearchDeliveryError):
        await transport.send_observability_payload(_vendor_payload())

    assert indexer.calls == 1
    assert sleep.delays == []


@pytest.mark.asyncio
async def test_max_attempts_one_does_not_retry() -> None:
    indexer = FlakyIndexer(
        [_delivery_error(status_code=503, retriable=True, reason="http_status_503")]
    )
    transport, sleep = _transport(
        indexer,
        retry_policy=ElasticsearchRetryPolicy(max_attempts=1),
    )

    with pytest.raises(ElasticsearchDeliveryError):
        await transport.send_observability_payload(_vendor_payload())

    assert indexer.calls == 1
    assert sleep.delays == []


@pytest.mark.asyncio
async def test_exhausted_retries_raise_safe_classified_error() -> None:
    indexer = FlakyIndexer(
        [
            _delivery_error(status_code=503, retriable=True, reason="http_status_503"),
            _delivery_error(status_code=503, retriable=True, reason="http_status_503"),
            _delivery_error(status_code=503, retriable=True, reason="http_status_503"),
        ]
    )
    transport, sleep = _transport(indexer)

    with pytest.raises(ElasticsearchDeliveryError) as exc_info:
        await transport.send_observability_payload(_vendor_payload())

    detail = exc_info.value.detail
    assert detail.operation == "send_observability_payload"
    assert detail.status_code == 503
    assert detail.retriable is True
    assert indexer.calls == 3
    assert sleep.delays == [0.25, 0.5]


@pytest.mark.asyncio
async def test_exhausted_retry_message_does_not_include_raw_document_body() -> None:
    class UnsafeDocumentIndexer:
        def __init__(self) -> None:
            self.calls = 0

        def index_document(
            self,
            *,
            index: str,
            document: dict[str, Any],
            doc_id: str | None = None,
        ) -> str:
            self.calls += 1
            merged = dict(document)
            merged.update(_unsafe_document())
            assert merged["intergrax.prompt"] == _UNSAFE_PROMPT
            raise classify_elasticsearch_delivery_error(
                _http_status_error(503),
                operation="index_document",
                index=index,
            )

    indexer = UnsafeDocumentIndexer()
    transport, _sleep = _transport(
        indexer,
        retry_policy=ElasticsearchRetryPolicy(max_attempts=3),
    )

    with pytest.raises(ElasticsearchDeliveryError) as exc_info:
        await transport.send_observability_payload(_vendor_payload())

    message = str(exc_info.value)
    for unsafe in (_UNSAFE_PROMPT, _UNSAFE_SECRET, _UNSAFE_PATH):
        assert unsafe not in message
    assert indexer.calls == 3


def test_backoff_delays_are_bounded_by_max_backoff_seconds() -> None:
    policy = ElasticsearchRetryPolicy(
        max_attempts=6,
        initial_backoff_seconds=0.25,
        max_backoff_seconds=2.0,
    )
    previous = 0.0
    delays: list[float] = []
    for retry_number in range(1, 5):
        delay = compute_elasticsearch_retry_backoff_seconds(
            retry_after_failure_number=retry_number,
            previous_backoff_seconds=previous,
            policy=policy,
        )
        delays.append(delay)
        previous = delay

    assert delays == [0.25, 0.5, 1.0, 2.0]


@pytest.mark.asyncio
async def test_transport_backoff_sequence_respects_max_backoff_seconds() -> None:
    indexer = FlakyIndexer(
        [
            _delivery_error(status_code=503, retriable=True, reason="http_status_503"),
            _delivery_error(status_code=503, retriable=True, reason="http_status_503"),
            _delivery_error(status_code=503, retriable=True, reason="http_status_503"),
            _delivery_error(status_code=503, retriable=True, reason="http_status_503"),
            "doc-1",
        ]
    )
    transport, sleep = _transport(
        indexer,
        retry_policy=ElasticsearchRetryPolicy(
            max_attempts=5,
            initial_backoff_seconds=0.25,
            max_backoff_seconds=2.0,
        ),
    )

    await transport.send_observability_payload(_vendor_payload())

    assert sleep.delays == [0.25, 0.5, 1.0, 2.0]
