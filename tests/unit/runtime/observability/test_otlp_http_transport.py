# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.observability.export_attributes import (
    ApplicationObservabilityAttributes,
    observability_attribute_key,
    sanitize_application_observability_attributes,
)
from intergrax.runtime.observability.export_boundary import (
    FORBIDDEN_EXPORT_CONTENT_FIELDS,
    ExportRecordKind,
    ObservabilityExportEnvelope,
    envelope_from_runtime_event,
    envelope_is_content_safe,
)
from intergrax.runtime.observability.export_policy import (
    ObservabilityExportPolicy,
    try_export_observability_envelope,
)
from intergrax.runtime.observability.otlp_exporter import (
    OtlpObservabilityExporter,
    OtlpObservabilityExporterConfig,
    OtlpTransport,
)
from intergrax.runtime.observability.otlp_http_transport import OtlpHttpTransport

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_OTLP_HTTP_TRANSPORT_PATH = (
    _PROJECT_ROOT / "intergrax" / "runtime" / "observability" / "otlp_http_transport.py"
)

_FORBIDDEN_VENDOR_TOKENS = (
    "langfuse",
    "arize",
    "phoenix",
    "elasticsearch",
    "opentelemetry",
    "integrations.providers.observability_backend",
)


class ExampleApplicationObservabilityAttributes(ApplicationObservabilityAttributes):
    namespace: str = "example"
    operation: str = "example.run"
    result_count: int = 0
    strategy: str | None = None


def _default_config() -> OtlpObservabilityExporterConfig:
    return OtlpObservabilityExporterConfig(
        endpoint="https://collector.example/v1/logs",
        service_name="intergrax.test",
        service_version="1.0.0",
        environment="test",
        timeout_seconds=5.0,
        headers={
            "Authorization": "Bearer test-token",
            "X-Custom-Safe": "safe-value",
        },
    )


def _sample_payload() -> dict[str, Any]:
    return {
        "resourceLogs": [
            {
                "resource": {"attributes": [{"key": "service.name", "value": {"stringValue": "intergrax.test"}}]},
                "scopeLogs": [
                    {
                        "scope": {"name": "intergrax.observability.export"},
                        "logRecords": [
                            {
                                "timeUnixNano": "1234567890000000000",
                                "severityText": "SUCCEEDED",
                                "body": {"stringValue": "runtime_event"},
                                "attributes": [
                                    {
                                        "key": "intergrax.run_id",
                                        "value": {"stringValue": "run-1"},
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
    }


def _mock_client(*, status_code: int = 200) -> httpx.AsyncClient:
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.raise_for_status = MagicMock()
    if status_code >= 400:
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error",
            request=MagicMock(),
            response=response,
        )

    client = MagicMock(spec=httpx.AsyncClient)
    client.post = AsyncMock(return_value=response)
    return client


def _post_kwargs(client: httpx.AsyncClient) -> dict[str, Any]:
    assert client.post.await_count == 1
    return client.post.await_args.kwargs


@pytest.mark.asyncio
async def test_otlp_http_transport_implements_otlp_transport_protocol() -> None:
    transport = OtlpHttpTransport(client=_mock_client())
    assert isinstance(transport, OtlpTransport)


@pytest.mark.asyncio
async def test_sends_json_post_to_configured_endpoint() -> None:
    client = _mock_client()
    transport = OtlpHttpTransport(client=client)
    config = _default_config()
    payload = _sample_payload()

    await transport.send(payload, config=config)

    kwargs = _post_kwargs(client)
    assert kwargs["content"] == json.dumps(payload, ensure_ascii=False).encode("utf-8")
    assert client.post.await_args.args[0] == config.endpoint


@pytest.mark.asyncio
async def test_sets_content_type_application_json() -> None:
    client = _mock_client()
    transport = OtlpHttpTransport(client=client)
    config = OtlpObservabilityExporterConfig(
        endpoint="https://collector.example/v1/logs",
        service_name="intergrax.test",
    )

    await transport.send(_sample_payload(), config=config)

    headers = _post_kwargs(client)["headers"]
    assert headers["Content-Type"] == "application/json"


@pytest.mark.asyncio
async def test_includes_configured_safe_headers_in_request() -> None:
    client = _mock_client()
    transport = OtlpHttpTransport(client=client)
    config = _default_config()

    await transport.send(_sample_payload(), config=config)

    headers = _post_kwargs(client)["headers"]
    assert headers["Authorization"] == "Bearer test-token"
    assert headers["X-Custom-Safe"] == "safe-value"
    assert headers["Content-Type"] == "application/json"


@pytest.mark.asyncio
async def test_does_not_put_endpoint_headers_or_tokens_into_telemetry_payload() -> None:
    client = _mock_client()
    transport = OtlpHttpTransport(client=client)
    config = _default_config()
    payload = _sample_payload()

    await transport.send(payload, config=config)

    body = _post_kwargs(client)["content"].decode("utf-8")
    assert config.endpoint not in body
    assert "Authorization" not in body
    assert "Bearer test-token" not in body
    assert "X-Custom-Safe" not in body
    assert json.loads(body) == payload


@pytest.mark.asyncio
async def test_raises_on_non_2xx_response() -> None:
    client = _mock_client(status_code=503)
    transport = OtlpHttpTransport(client=client)

    with pytest.raises(httpx.HTTPStatusError):
        await transport.send(_sample_payload(), config=_default_config())


@pytest.mark.asyncio
async def test_transport_failure_is_isolated_through_try_export() -> None:
    client = _mock_client(status_code=500)
    transport = OtlpHttpTransport(client=client)
    exporter = OtlpObservabilityExporter(_default_config(), transport)
    envelope = ObservabilityExportEnvelope(record_kind=ExportRecordKind.RUNTIME_EVENT, run_id="run-1")

    result = await try_export_observability_envelope(
        envelope,
        exporter=exporter,
        policy=ObservabilityExportPolicy(enabled=True),
    )

    assert result.exported is False
    assert result.reason == "exporter_failed"
    assert client.post.await_count == 1


@pytest.mark.asyncio
async def test_payload_contains_sanitized_application_attributes_when_provided() -> None:
    client = _mock_client()
    transport = OtlpHttpTransport(client=client)
    exporter = OtlpObservabilityExporter(_default_config(), transport)
    sanitized_attrs = ExampleApplicationObservabilityAttributes(result_count=5, strategy="safe")
    sanitized = sanitize_application_observability_attributes(sanitized_attrs).sanitized
    envelope = ObservabilityExportEnvelope(
        record_kind=ExportRecordKind.RUNTIME_EVENT,
        run_id="run-1",
        sanitized_application_attributes=sanitized,
    )

    await exporter.export(envelope)

    body = json.loads(_post_kwargs(client)["content"].decode("utf-8"))
    attrs = body["resourceLogs"][0]["scopeLogs"][0]["logRecords"][0]["attributes"]
    attr_map = {item["key"]: item["value"] for item in attrs}
    assert attr_map[observability_attribute_key("example", "result_count")]["intValue"] == "5"
    assert attr_map[observability_attribute_key("example", "strategy")]["stringValue"] == "safe"
    assert attr_map["intergrax.application.namespace"]["stringValue"] == "example"


@pytest.mark.asyncio
async def test_payload_does_not_contain_raw_application_attributes() -> None:
    client = _mock_client()
    transport = OtlpHttpTransport(client=client)
    exporter = OtlpObservabilityExporter(_default_config(), transport)
    attrs = ExampleApplicationObservabilityAttributes(result_count=3)
    envelope = ObservabilityExportEnvelope(
        record_kind=ExportRecordKind.RUNTIME_EVENT,
        run_id="run-1",
        application_attributes=attrs,
    )

    await exporter.export(envelope)

    serialized = _post_kwargs(client)["content"].decode("utf-8")
    assert "application_attributes" not in serialized


@pytest.mark.asyncio
async def test_end_to_end_through_try_export_strips_raw_runtime_payload_content() -> None:
    client = _mock_client()
    transport = OtlpHttpTransport(client=client)
    exporter = OtlpObservabilityExporter(_default_config(), transport)
    attrs = ExampleApplicationObservabilityAttributes(result_count=5, strategy="safe")
    event = RuntimeEvent(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        tenant_id="tenant-a",
        agent_id="agent-1",
        event_type=RuntimeEventType.TOOL_COMPLETED,
        phase=ExecutionPhase.STEP_EXECUTION,
        payload={
            "tool_id": "workspace.read_file",
            "latency_ms": 9,
            "prompt": "secret prompt",
            "content": "raw body",
            "source_path": "C:\\Users\\secret\\project\\file.txt",
        },
    )
    envelope = envelope_from_runtime_event(event)
    envelope = envelope.model_copy(update={"application_attributes": attrs})

    result = await try_export_observability_envelope(
        envelope,
        exporter=exporter,
        policy=ObservabilityExportPolicy(enabled=True, export_content=False),
    )

    assert result.exported is True
    body = json.loads(_post_kwargs(client)["content"].decode("utf-8"))
    serialized = json.dumps(body)
    assert envelope_is_content_safe(result.envelope)  # type: ignore[arg-type]
    assert "secret prompt" not in serialized
    assert "raw body" not in serialized
    assert "C:\\Users\\secret" not in serialized
    attrs = body["resourceLogs"][0]["scopeLogs"][0]["logRecords"][0]["attributes"]
    attr_keys = {item["key"] for item in attrs}
    for key in FORBIDDEN_EXPORT_CONTENT_FIELDS:
        assert key not in attr_keys


def test_otlp_http_transport_has_no_vendor_sdk_coupling() -> None:
    source = _OTLP_HTTP_TRANSPORT_PATH.read_text(encoding="utf-8")
    for token in _FORBIDDEN_VENDOR_TOKENS:
        assert token not in source, (
            f"otlp_http_transport.py contains forbidden vendor coupling token: {token}"
        )


@pytest.mark.asyncio
async def test_does_not_perform_real_network_calls() -> None:
    client = _mock_client()
    transport = OtlpHttpTransport(client=client)

    await transport.send(_sample_payload(), config=_default_config())

    client.post.assert_awaited_once()
