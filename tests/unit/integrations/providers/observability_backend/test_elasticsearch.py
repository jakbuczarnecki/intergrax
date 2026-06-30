# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Elasticsearch integration provider (Phase M.6 P2)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from intergrax.integrations._shared.conformance import assert_observability_backend
from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationConfigurationError
from intergrax.integrations.providers.observability_backend.elasticsearch.integration import (
    ElasticsearchObservabilityIntegration,
)
from intergrax.integrations.providers.observability_backend.elasticsearch.bundle import (
    ElasticsearchIntegrationBundle,
    create_elasticsearch_integration,
    create_elasticsearch_observability_backend,
    create_elasticsearch_observability_integration,
    create_elasticsearch_observability_transport,
)
from intergrax.integrations.providers.observability_backend.elasticsearch.config import (
    ENV_ELASTICSEARCH_INDEX,
    ENV_ELASTICSEARCH_URL,
    ElasticsearchIntegrationConfig,
)
from intergrax.integrations.providers.observability_backend.elasticsearch.register import register_elasticsearch_integration
from intergrax.integrations.providers.observability_backend.elasticsearch.transport import (
    ElasticsearchHttpObservabilityTransport,
    map_vendor_payload_to_elasticsearch_document,
)
from intergrax.integrations.registry.bootstrap import register_default_integrations, reset_default_integrations_state
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.factory import resolve
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.integrations.observability import (
    ObservabilityVendorPayload,
    ObservabilityVendorSignal,
)
from intergrax.runtime.observability.export_attributes import (
    ApplicationObservabilityAttributes,
    observability_attribute_key,
    sanitize_application_observability_attributes,
)
from intergrax.runtime.observability.export_boundary import (
    ExportRecordKind,
    ExportStatus,
    FORBIDDEN_EXPORT_CONTENT_FIELDS,
    ObservabilityExportEnvelope,
)
from intergrax.runtime.observability.export_policy import (
    ObservabilityExportPolicy,
    apply_observability_export_policy,
)

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_ES_PKG = _PROJECT_ROOT / "intergrax" / "integrations" / "providers" / "elasticsearch"
_THIS_TEST = Path(__file__).resolve()
_SCAN_ROOTS = ("intergrax", "applications", "agents", "tests")
_SKIP_DIR_NAMES = {".venv", "build", "__pycache__", "node_modules"}
_FORBIDDEN_OUTSIDE_PROVIDER = (
    "ElasticsearchRestClient(",
    "integrations.providers.elasticsearch.client",
    "integrations.providers.elasticsearch.opens",
    "httpx.Client(",
)


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_catalog()
    reset_default_integrations_state()
    yield
    clear_catalog()
    reset_default_integrations_state()


def _elasticsearch_config() -> ElasticsearchIntegrationConfig:
    return ElasticsearchIntegrationConfig(
        base_url="http://elasticsearch.local:9200",
        index="logs-*",
    )


def _mock_http_client(*, post_payload: dict | None = None) -> MagicMock:
    client = MagicMock()
    response = MagicMock()
    response.json.return_value = post_payload or {}
    response.raise_for_status.return_value = None
    client.post.return_value = response
    return client


def _iter_python_files(*roots: str):
    for root_name in roots:
        root = _PROJECT_ROOT / root_name
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if any(part in _SKIP_DIR_NAMES for part in path.parts):
                continue
            yield path


def test_httpx_client_only_created_in_opens_module() -> None:
    violations: list[str] = []
    for path in _ES_PKG.glob("*.py"):
        if path.name == "opens.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "httpx" in text:
            violations.append(path.name)
    assert violations == []


def test_elasticsearch_not_constructed_outside_provider_package() -> None:
    violations: list[str] = []
    for path in _iter_python_files(*_SCAN_ROOTS):
        if path.resolve() == _THIS_TEST.resolve():
            continue
        if _ES_PKG in path.parents:
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in _FORBIDDEN_OUTSIDE_PROVIDER:
            if pattern in text:
                violations.append(f"{path.relative_to(_PROJECT_ROOT).as_posix()}: {pattern}")
    assert violations == []


def test_elasticsearch_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_ELASTICSEARCH_URL, "http://es:9200")
    monkeypatch.setenv(ENV_ELASTICSEARCH_INDEX, "app-logs-*")
    config = ElasticsearchIntegrationConfig.from_env()
    assert config.base_url == "http://es:9200"
    assert config.index == "app-logs-*"


def test_query_instant_parses_value_count() -> None:
    http = _mock_http_client(post_payload={"aggregations": {"count": {"value": 42}}})
    backend = create_elasticsearch_observability_backend(
        **_elasticsearch_config().model_dump(),
        http_client=http,
    )

    result = backend.query_instant("level:error", eval_time=1_700_000_000.0)

    assert result.result_type == "vector"
    assert len(result.series) == 1
    assert result.series[0].points[0].value == 42.0
    assert result.series[0].points[0].timestamp == 1_700_000_000.0
    http.post.assert_called_once()
    assert http.post.call_args.args[0] == "/logs-*/_search"
    assert_observability_backend(backend)


def test_query_range_parses_date_histogram() -> None:
    http = _mock_http_client(
        post_payload={
            "aggregations": {
                "timeline": {
                    "buckets": [
                        {"key": 1_700_000_000_000, "doc_count": 5},
                        {"key": 1_700_000_015_000, "doc_count": 7},
                    ]
                }
            }
        }
    )
    backend = create_elasticsearch_observability_backend(
        **_elasticsearch_config().model_dump(),
        http_client=http,
    )

    result = backend.query_range(
        "service:api",
        start=1_700_000_000.0,
        end=1_700_000_030.0,
        step="15s",
    )

    assert result.result_type == "matrix"
    assert len(result.series[0].points) == 2
    assert result.series[0].points[0].value == 5.0
    body = http.post.call_args.kwargs["json"]
    assert body["aggs"]["timeline"]["date_histogram"]["fixed_interval"] == "15s"


def test_create_elasticsearch_integration_bundle() -> None:
    http = _mock_http_client(post_payload={"aggregations": {"count": {"value": 0}}})
    bundle = create_elasticsearch_integration(**_elasticsearch_config().model_dump(), http_client=http)

    assert isinstance(bundle, ElasticsearchIntegrationBundle)
    assert isinstance(bundle.observability_backend, ElasticsearchObservabilityIntegration)


def test_register_and_resolve_via_profile() -> None:
    register_elasticsearch_integration()
    profile = IntegrationProfile(observability_backend="elasticsearch")
    http = _mock_http_client(post_payload={"aggregations": {"count": {"value": 1}}})

    backend = resolve(
        IntegrationCategory.OBSERVABILITY_BACKEND,
        profile=profile,
        config={**_elasticsearch_config().model_dump(), "http_client": http},
    )

    assert_observability_backend(backend)
    assert isinstance(backend, ElasticsearchObservabilityIntegration)


def test_register_default_integrations_includes_elasticsearch() -> None:
    register_default_integrations()
    profile = IntegrationProfile(observability_backend="elasticsearch")
    http = _mock_http_client(post_payload={"aggregations": {"count": {"value": 1}}})

    backend = resolve(
        IntegrationCategory.OBSERVABILITY_BACKEND,
        profile=profile,
        config={**_elasticsearch_config().model_dump(), "http_client": http},
    )

    assert isinstance(backend, ElasticsearchObservabilityIntegration)


def test_opens_creates_httpx_client_when_not_injected() -> None:
    config = _elasticsearch_config()
    mock_client = MagicMock()

    with patch(
        "intergrax.integrations.providers.observability_backend.elasticsearch.opens._create_http_client",
        return_value=mock_client,
    ) as create_mock:
        from intergrax.integrations.providers.observability_backend.elasticsearch.opens import open_elasticsearch_rest_client

        client = open_elasticsearch_rest_client(config)

    create_mock.assert_called_once_with(config)
    assert client.config is config


class ExampleWorkspaceObservabilityAttributes(ApplicationObservabilityAttributes):
    namespace: str = "local_workspace"
    operation: str | None = "index_job"
    file_count: int | None = None


class FakeElasticsearchTransport:
    def __init__(self) -> None:
        self.payloads: list[ObservabilityVendorPayload] = []
        self.send_count = 0

    async def send_observability_payload(self, payload: ObservabilityVendorPayload) -> None:
        self.send_count += 1
        self.payloads.append(payload)


class FakeElasticsearchIndexer:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def index_document(
        self,
        *,
        index: str,
        document: dict[str, Any],
        doc_id: str | None = None,
    ) -> str:
        self.calls.append({"index": index, "document": dict(document), "doc_id": doc_id})
        return doc_id or "generated-id"


def _sanitized_envelope_with_attributes() -> ObservabilityExportEnvelope:
    attributes = ExampleWorkspaceObservabilityAttributes(file_count=3)
    sanitize_result = sanitize_application_observability_attributes(attributes)
    envelope = ObservabilityExportEnvelope(
        record_kind=ExportRecordKind.RUNTIME_EVENT,
        run_id="run-1",
        task_id="task-1",
        agent_id="agent-1",
        capability="search",
        event_type="tool.completed",
        status=ExportStatus.SUCCEEDED,
        latency_ms=42,
        counts={"hit_count": 2},
        tool_id="grep",
        event_id="event-1",
        application_attributes=attributes,
        sanitized_application_attributes=sanitize_result.sanitized,
    )
    policy_result = apply_observability_export_policy(
        envelope,
        ObservabilityExportPolicy(enabled=True),
    )
    assert policy_result.exported and policy_result.envelope is not None
    return policy_result.envelope


def _vendor_payload_from_envelope(envelope: ObservabilityExportEnvelope) -> ObservabilityVendorPayload:
    integration = ElasticsearchObservabilityIntegration.from_transport(FakeElasticsearchTransport())
    return integration.map_envelope(envelope).payload


@pytest.mark.asyncio
async def test_elasticsearch_integration_disabled_does_not_send() -> None:
    transport = FakeElasticsearchTransport()
    integration = ElasticsearchObservabilityIntegration.from_transport(transport, enabled=False)
    envelope = _sanitized_envelope_with_attributes()

    await integration.export(envelope)

    assert transport.send_count == 0
    assert transport.payloads == []


@pytest.mark.asyncio
async def test_elasticsearch_integration_enabled_sends_one_payload() -> None:
    transport = FakeElasticsearchTransport()
    integration = ElasticsearchObservabilityIntegration.from_transport(transport, enabled=True)
    envelope = _sanitized_envelope_with_attributes()

    await integration.export(envelope)

    assert transport.send_count == 1
    assert len(transport.payloads) == 1
    assert transport.payloads[0].provider_id == "elasticsearch"
    assert transport.payloads[0].run_id == "run-1"


def test_elasticsearch_map_envelope_rejects_raw_application_attributes() -> None:
    integration = ElasticsearchObservabilityIntegration.from_transport(FakeElasticsearchTransport())
    attributes = ExampleWorkspaceObservabilityAttributes(file_count=1)
    envelope = ObservabilityExportEnvelope(
        record_kind=ExportRecordKind.RUNTIME_EVENT,
        application_attributes=attributes,
    )

    with pytest.raises(ValueError, match="raw application_attributes"):
        integration.map_envelope(envelope)


def test_create_elasticsearch_observability_integration_enabled_without_transport_fails() -> None:
    with pytest.raises(IntegrationConfigurationError, match="transport"):
        create_elasticsearch_observability_integration(enabled=True, transport=None)


def test_elasticsearch_document_mapping_excludes_raw_content_fields() -> None:
    envelope = _sanitized_envelope_with_attributes()
    payload = _vendor_payload_from_envelope(envelope)
    document = map_vendor_payload_to_elasticsearch_document(payload)

    serialized = json.dumps(document).lower()
    forbidden_samples = (
        "raw prompt text",
        "secret-api-key",
        "c:\\users\\secret\\document.pdf",
        "/home/user/secret/document.pdf",
    )
    for sample in forbidden_samples:
        assert sample not in serialized

    for field_name in FORBIDDEN_EXPORT_CONTENT_FIELDS:
        assert field_name not in document

    assert "application_attributes" not in document
    assert document["intergrax.run_id"] == "run-1"
    assert document["intergrax.event_id"] == "event-1"
    assert observability_attribute_key("local_workspace", "file_count") in document


@pytest.mark.asyncio
async def test_elasticsearch_http_transport_indexes_through_fake_indexer() -> None:
    indexer = FakeElasticsearchIndexer()
    transport = ElasticsearchHttpObservabilityTransport(
        indexer,
        index="observability-events",
        timestamp_field="@timestamp",
    )
    envelope = _sanitized_envelope_with_attributes()
    payload = _vendor_payload_from_envelope(envelope)

    with patch(
        "intergrax.integrations.providers.observability_backend.elasticsearch.transport.asyncio.to_thread",
        wraps=asyncio.to_thread,
    ) as to_thread_mock:
        await transport.send_observability_payload(payload)

    to_thread_mock.assert_called_once()
    assert to_thread_mock.call_args.args[0].__name__ == "index_document"
    assert to_thread_mock.call_args.kwargs["index"] == "observability-events"
    assert len(indexer.calls) == 1
    call = indexer.calls[0]
    assert to_thread_mock.call_args.kwargs["document"] == call["document"]
    assert call["index"] == "observability-events"
    assert call["doc_id"] is None
    assert call["document"]["intergrax.provider_id"] == "elasticsearch"
    assert call["document"]["intergrax.event_id"] == "event-1"
    assert call["document"]["@timestamp"] == payload.recorded_at.isoformat()


@pytest.mark.asyncio
async def test_elasticsearch_http_transport_preserves_correlation_id_in_document() -> None:
    indexer = FakeElasticsearchIndexer()
    transport = ElasticsearchHttpObservabilityTransport(
        indexer,
        index="observability-events",
    )
    envelope = _sanitized_envelope_with_attributes()
    payload = _vendor_payload_from_envelope(envelope)
    payload = payload.model_copy(update={"event_id": "", "correlation_id": "corr-42"})

    await transport.send_observability_payload(payload)

    call = indexer.calls[0]
    assert call["doc_id"] is None
    assert call["document"]["intergrax.correlation_id"] == "corr-42"
    assert "intergrax.event_id" not in call["document"]


@pytest.mark.asyncio
async def test_elasticsearch_http_transport_uses_rest_client_index() -> None:
    http = _mock_http_client()
    backend = create_elasticsearch_observability_backend(
        **_elasticsearch_config().model_dump(),
        http_client=http,
    )
    transport = create_elasticsearch_observability_transport(
        client=backend.rest_client,
        index="observability-events",
    )
    envelope = _sanitized_envelope_with_attributes()
    payload = _vendor_payload_from_envelope(envelope)

    with patch(
        "intergrax.integrations.providers.observability_backend.elasticsearch.transport.asyncio.to_thread",
        wraps=asyncio.to_thread,
    ):
        await transport.send_observability_payload(payload)

    http.post.assert_called_once()
    assert http.post.call_args.args[0] == "/observability-events/_doc"
    body = http.post.call_args.kwargs["json"]
    assert body["intergrax.run_id"] == "run-1"
    assert body["intergrax.event_id"] == "event-1"
    assert "prompt" not in body


def test_create_elasticsearch_observability_integration_factory() -> None:
    transport = FakeElasticsearchTransport()
    integration = create_elasticsearch_observability_integration(transport=transport, enabled=True)

    assert isinstance(integration, ElasticsearchObservabilityIntegration)
    assert integration.transport is transport
    assert integration.supported_signals == (
        ObservabilityVendorSignal.EVENTS,
        ObservabilityVendorSignal.LOGS,
        ObservabilityVendorSignal.TRACES,
        ObservabilityVendorSignal.METRICS,
    )


def test_no_real_network_in_elasticsearch_vendor_transport_tests() -> None:
    transport = FakeElasticsearchTransport()
    integration = create_elasticsearch_observability_integration(transport=transport, enabled=True)
    envelope = _sanitized_envelope_with_attributes()

    asyncio.run(integration.export(envelope))

    assert transport.send_count == 1
