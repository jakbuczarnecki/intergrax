# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Elasticsearch file failed-delivery sink (OBS-VENDOR-6C-B1)."""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from intergrax.integrations.providers.observability_backend.elasticsearch.client import (
    ElasticsearchDeliveryError,
    ElasticsearchDeliveryErrorDetail,
)
from intergrax.integrations.providers.observability_backend.elasticsearch.failed_delivery import (
    FileElasticsearchFailedDeliverySink,
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


def _sample_record(**overrides: object) -> ElasticsearchFailedDeliveryRecord:
    defaults: dict[str, object] = {
        "provider_id": "elasticsearch",
        "operation": "send_observability_payload",
        "index": "observability-events",
        "status_code": 503,
        "reason": "http_status_503",
        "retriable": True,
        "attempts": 3,
        "exhausted": True,
    }
    defaults.update(overrides)
    return ElasticsearchFailedDeliveryRecord(**defaults)  # type: ignore[arg-type]


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

    def index_document(
        self,
        *,
        index: str,
        document: dict[str, Any],
        doc_id: str | None = None,
    ) -> str:
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


def test_file_sink_appends_one_jsonl_record(tmp_path: Path) -> None:
    output_path = tmp_path / "failed-deliveries.jsonl"
    sink = FileElasticsearchFailedDeliverySink(output_path)
    record = _sample_record()

    sink.record_failed_delivery(record)

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == asdict(record)


def test_file_sink_appends_multiple_records_without_overwriting(tmp_path: Path) -> None:
    output_path = tmp_path / "failed-deliveries.jsonl"
    sink = FileElasticsearchFailedDeliverySink(output_path)
    first = _sample_record(attempts=1, reason="http_status_400", status_code=400, retriable=False)
    second = _sample_record(attempts=3, reason="http_status_503", status_code=503, retriable=True)

    sink.record_failed_delivery(first)
    sink.record_failed_delivery(second)

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == asdict(first)
    assert json.loads(lines[1]) == asdict(second)


def test_file_sink_creates_parent_directories(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "dir" / "failed-deliveries.jsonl"
    sink = FileElasticsearchFailedDeliverySink(output_path)

    sink.record_failed_delivery(_sample_record())

    assert output_path.is_file()


def test_jsonl_contains_exactly_safe_record_fields(tmp_path: Path) -> None:
    output_path = tmp_path / "failed-deliveries.jsonl"
    sink = FileElasticsearchFailedDeliverySink(output_path)
    record = _sample_record(status_code=None)

    sink.record_failed_delivery(record)

    parsed = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert frozenset(parsed) == _SAFE_RECORD_FIELD_NAMES
    assert parsed == asdict(record)


@pytest.mark.asyncio
async def test_jsonl_does_not_contain_unsafe_payload_content(tmp_path: Path) -> None:
    output_path = tmp_path / "failed-deliveries.jsonl"
    sink = FileElasticsearchFailedDeliverySink(output_path)
    indexer = UnsafeDocumentIndexer(
        error=_delivery_error(status_code=400, retriable=False, reason="http_status_400"),
    )
    transport = ElasticsearchHttpObservabilityTransport(
        indexer,
        index="observability-events",
        sleep=RecordingSleep(),
        failed_delivery_sink=sink,
    )

    with pytest.raises(ElasticsearchDeliveryError):
        await transport.send_observability_payload(_vendor_payload())

    written = output_path.read_text(encoding="utf-8")
    for unsafe in (
        _UNSAFE_PROMPT,
        _UNSAFE_SECRET,
        _UNSAFE_PATH,
        _UNSAFE_TOOL_ARGS,
        _UNSAFE_CHUNK,
        "run-1",
        "event-1",
    ):
        assert unsafe not in written


@pytest.mark.asyncio
async def test_transport_uses_file_failed_delivery_sink(tmp_path: Path) -> None:
    output_path = tmp_path / "failed-deliveries.jsonl"
    sink = FileElasticsearchFailedDeliverySink(output_path)
    indexer = FlakyIndexer(
        [_delivery_error(status_code=400, retriable=False, reason="http_status_400")]
    )
    transport = ElasticsearchHttpObservabilityTransport(
        indexer,
        index="observability-events",
        sleep=RecordingSleep(),
        failed_delivery_sink=sink,
    )

    with pytest.raises(ElasticsearchDeliveryError):
        await transport.send_observability_payload(_vendor_payload())

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["operation"] == "send_observability_payload"
    assert parsed["status_code"] == 400
    assert parsed["attempts"] == 1
    assert parsed["exhausted"] is True


@pytest.mark.asyncio
async def test_transport_without_failed_delivery_sink_uses_no_op_default(tmp_path: Path) -> None:
    output_path = tmp_path / "should-not-exist.jsonl"
    indexer = FlakyIndexer(
        [_delivery_error(status_code=400, retriable=False, reason="http_status_400")]
    )
    transport = ElasticsearchHttpObservabilityTransport(
        indexer,
        index="observability-events",
        sleep=RecordingSleep(),
    )

    with pytest.raises(ElasticsearchDeliveryError):
        await transport.send_observability_payload(_vendor_payload())

    assert not output_path.exists()
