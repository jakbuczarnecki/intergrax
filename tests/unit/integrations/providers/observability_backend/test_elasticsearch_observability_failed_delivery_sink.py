# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Elasticsearch failed-delivery sink contract (OBS-VENDOR-6C-A)."""

from __future__ import annotations

from dataclasses import asdict, fields
from datetime import datetime, timezone
from typing import Any

import pytest

from intergrax.integrations.providers.observability_backend.elasticsearch.client import (
    ElasticsearchDeliveryError,
    ElasticsearchDeliveryErrorDetail,
)
from intergrax.integrations.providers.observability_backend.elasticsearch.config import (
    ElasticsearchRetryPolicy,
)
from intergrax.integrations.providers.observability_backend.elasticsearch.transport import (
    ElasticsearchFailedDeliveryRecord,
    ElasticsearchHttpObservabilityTransport,
)
from intergrax.runtime.integrations.observability import ObservabilityVendorPayload

pytestmark = pytest.mark.unit

_UNSAFE_PROMPT = "RAW_PROMPT_DO_NOT_LEAK_abc123"
_UNSAFE_SECRET = "secret-api-key-should-not-appear"
_UNSAFE_PATH = "C:\\Users\\secret\\document.pdf"
_UNSAFE_TOOL_ARGS = '{"tool":"grep","args":{"pattern":"secret"}}'
_UNSAFE_CHUNK = "chunk-content-must-not-leak-xyz"
_SAFE_RECORD_FIELD_NAMES = frozenset(field.name for field in fields(ElasticsearchFailedDeliveryRecord))


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


class RecordingSleep:
    async def __call__(self, delay: float) -> None:
        return None


class RecordingFailedDeliverySink:
    def __init__(self) -> None:
        self.records: list[ElasticsearchFailedDeliveryRecord] = []

    def record_failed_delivery(self, record: ElasticsearchFailedDeliveryRecord) -> None:
        self.records.append(record)


class FlakyIndexer:
    def __init__(self, outcomes: list[BaseException | str]) -> None:
        self._outcomes = list(outcomes)
        self.calls = 0

    def index_document(
        self,
        *,
        index: str,
        document: dict[str, Any],
        doc_id: str | None = None,
    ) -> str:
        self.calls += 1
        if not self._outcomes:
            return "generated-id"
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class UnsafeDocumentIndexer:
    def __init__(self, *, error: ElasticsearchDeliveryError) -> None:
        self._error = error
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
        merged.update(
            {
                "intergrax.prompt": _UNSAFE_PROMPT,
                "intergrax.secret": _UNSAFE_SECRET,
                "intergrax.path": _UNSAFE_PATH,
                "intergrax.tool_args": _UNSAFE_TOOL_ARGS,
                "intergrax.chunk": _UNSAFE_CHUNK,
            }
        )
        raise self._error


def _transport(
    indexer: FlakyIndexer | UnsafeDocumentIndexer,
    *,
    retry_policy: ElasticsearchRetryPolicy | None = None,
    sink: RecordingFailedDeliverySink | None = None,
) -> tuple[ElasticsearchHttpObservabilityTransport, RecordingFailedDeliverySink]:
    active_sink = sink or RecordingFailedDeliverySink()
    transport = ElasticsearchHttpObservabilityTransport(
        indexer,
        index="observability-events",
        retry_policy=retry_policy,
        sleep=RecordingSleep(),
        failed_delivery_sink=active_sink,
    )
    return transport, active_sink


def _record_text(record: ElasticsearchFailedDeliveryRecord) -> str:
    return str(asdict(record))


@pytest.mark.asyncio
async def test_sink_not_called_on_first_attempt_success() -> None:
    indexer = FlakyIndexer(["doc-1"])
    transport, sink = _transport(indexer)

    await transport.send_observability_payload(_vendor_payload())

    assert indexer.calls == 1
    assert sink.records == []


@pytest.mark.asyncio
async def test_sink_not_called_when_retriable_failures_eventually_succeed() -> None:
    indexer = FlakyIndexer(
        [
            _delivery_error(status_code=503, retriable=True, reason="http_status_503"),
            _delivery_error(status_code=503, retriable=True, reason="http_status_503"),
            "doc-1",
        ]
    )
    transport, sink = _transport(indexer)

    await transport.send_observability_payload(_vendor_payload())

    assert indexer.calls == 3
    assert sink.records == []


@pytest.mark.asyncio
async def test_sink_called_once_for_non_retriable_http_400() -> None:
    indexer = FlakyIndexer(
        [_delivery_error(status_code=400, retriable=False, reason="http_status_400")]
    )
    transport, sink = _transport(indexer)

    with pytest.raises(ElasticsearchDeliveryError) as exc_info:
        await transport.send_observability_payload(_vendor_payload())

    assert indexer.calls == 1
    assert len(sink.records) == 1
    record = sink.records[0]
    assert record.operation == "send_observability_payload"
    assert record.status_code == 400
    assert record.retriable is False
    assert record.attempts == 1
    assert record.exhausted is True
    assert exc_info.value.detail.operation == "send_observability_payload"


@pytest.mark.asyncio
async def test_sink_called_once_after_retries_exhausted_for_http_503() -> None:
    indexer = FlakyIndexer(
        [
            _delivery_error(status_code=503, retriable=True, reason="http_status_503"),
            _delivery_error(status_code=503, retriable=True, reason="http_status_503"),
            _delivery_error(status_code=503, retriable=True, reason="http_status_503"),
        ]
    )
    transport, sink = _transport(indexer)

    with pytest.raises(ElasticsearchDeliveryError):
        await transport.send_observability_payload(_vendor_payload())

    assert indexer.calls == 3
    assert len(sink.records) == 1
    record = sink.records[0]
    assert record.status_code == 503
    assert record.retriable is True
    assert record.attempts == 3
    assert record.exhausted is True


@pytest.mark.asyncio
async def test_sink_record_contains_safe_fields_only() -> None:
    indexer = FlakyIndexer(
        [_delivery_error(status_code=400, retriable=False, reason="http_status_400")]
    )
    transport, sink = _transport(indexer)

    with pytest.raises(ElasticsearchDeliveryError):
        await transport.send_observability_payload(_vendor_payload())

    record = sink.records[0]
    assert frozenset(asdict(record)) == _SAFE_RECORD_FIELD_NAMES
    assert record.provider_id == "elasticsearch"
    assert record.index == "observability-events"
    assert record.reason == "http_status_400"


@pytest.mark.asyncio
async def test_sink_record_does_not_contain_unsafe_payload_content() -> None:
    indexer = UnsafeDocumentIndexer(
        error=_delivery_error(status_code=400, retriable=False, reason="http_status_400"),
    )
    transport, sink = _transport(indexer)

    with pytest.raises(ElasticsearchDeliveryError):
        await transport.send_observability_payload(_vendor_payload())

    record_text = _record_text(sink.records[0])
    for unsafe in (
        _UNSAFE_PROMPT,
        _UNSAFE_SECRET,
        _UNSAFE_PATH,
        _UNSAFE_TOOL_ARGS,
        _UNSAFE_CHUNK,
        "run-1",
        "event-1",
    ):
        assert unsafe not in record_text


@pytest.mark.asyncio
async def test_sink_failure_does_not_mask_original_elasticsearch_delivery_error() -> None:
    class ExplodingSink:
        def record_failed_delivery(self, record: ElasticsearchFailedDeliveryRecord) -> None:
            raise RuntimeError("sink exploded")

    indexer = FlakyIndexer(
        [_delivery_error(status_code=400, retriable=False, reason="http_status_400")]
    )
    transport = ElasticsearchHttpObservabilityTransport(
        indexer,
        index="observability-events",
        sleep=RecordingSleep(),
        failed_delivery_sink=ExplodingSink(),
    )

    with pytest.raises(ElasticsearchDeliveryError) as exc_info:
        await transport.send_observability_payload(_vendor_payload())

    assert exc_info.value.detail.status_code == 400
    assert exc_info.value.detail.operation == "send_observability_payload"
