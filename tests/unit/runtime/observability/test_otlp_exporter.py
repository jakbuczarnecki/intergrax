# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.observability.export_attributes import (
    ApplicationObservabilityAttributes,
    observability_attribute_key,
)
from intergrax.runtime.observability.export_boundary import (
    FORBIDDEN_EXPORT_CONTENT_FIELDS,
    ExportRecordKind,
    ExportStatus,
    ObservabilityExportEnvelope,
    ObservabilityExporter,
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

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_OTLP_EXPORTER_PATH = _PROJECT_ROOT / "intergrax" / "runtime" / "observability" / "otlp_exporter.py"

_FORBIDDEN_VENDOR_TOKENS = (
    "langfuse",
    "arize",
    "phoenix",
    "elasticsearch",
    "opentelemetry",
    "integrations.providers.observability_backend",
    "httpx",
    "requests",
)


class ExampleApplicationObservabilityAttributes(ApplicationObservabilityAttributes):
    namespace: str = "example"
    operation: str = "example.run"
    result_count: int = 0
    strategy: str | None = None
    tags: list[str] | None = None


class FakeOtlpTransport:
    def __init__(self, *, fail: bool = False) -> None:
        self.payloads: list[dict[str, Any]] = []
        self.configs: list[OtlpObservabilityExporterConfig] = []
        self._fail = fail
        self.send_count = 0

    async def send(
        self,
        payload: dict[str, Any],
        *,
        config: OtlpObservabilityExporterConfig,
    ) -> None:
        self.send_count += 1
        if self._fail:
            raise RuntimeError("transport failure")
        self.payloads.append(payload)
        self.configs.append(config)


def _default_config() -> OtlpObservabilityExporterConfig:
    return OtlpObservabilityExporterConfig(
        endpoint="https://collector.example/v1/logs",
        service_name="intergrax.test",
        service_version="1.0.0",
        environment="test",
        timeout_seconds=5.0,
        headers={"Authorization": "Bearer test-token"},
    )


def _exporter(transport: FakeOtlpTransport | None = None) -> tuple[OtlpObservabilityExporter, FakeOtlpTransport]:
    active_transport = transport or FakeOtlpTransport()
    return OtlpObservabilityExporter(_default_config(), active_transport), active_transport


def _log_record_attributes(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return payload["resourceLogs"][0]["scopeLogs"][0]["logRecords"][0]["attributes"]


def _attribute_map(payload: dict[str, Any]) -> dict[str, Any]:
    attrs = _log_record_attributes(payload)
    mapped: dict[str, Any] = {}
    for item in attrs:
        key = item["key"]
        value = item["value"]
        if "stringValue" in value:
            mapped[key] = value["stringValue"]
        elif "intValue" in value:
            mapped[key] = int(value["intValue"])
        elif "boolValue" in value:
            mapped[key] = value["boolValue"]
        elif "doubleValue" in value:
            mapped[key] = value["doubleValue"]
        elif "arrayValue" in value:
            mapped[key] = [entry["stringValue"] for entry in value["arrayValue"]["values"]]
        else:
            mapped[key] = value
    return mapped


def _payload_serialized(payload: dict[str, Any]) -> str:
    return json.dumps(payload)


@pytest.mark.asyncio
async def test_otlp_exporter_implements_observability_exporter_protocol() -> None:
    exporter, _ = _exporter()
    assert isinstance(exporter, ObservabilityExporter)


@pytest.mark.asyncio
async def test_exports_one_sanitized_envelope_through_fake_transport() -> None:
    transport = FakeOtlpTransport()
    exporter, _ = _exporter(transport)
    envelope = ObservabilityExportEnvelope(
        record_kind=ExportRecordKind.RUNTIME_EVENT,
        run_id="run-1",
        event_id="evt-1",
    )

    await exporter.export(envelope)

    assert transport.send_count == 1
    assert len(transport.payloads) == 1
    assert _attribute_map(transport.payloads[0])["intergrax.run_id"] == "run-1"


@pytest.mark.asyncio
async def test_exports_multiple_envelopes_in_order() -> None:
    transport = FakeOtlpTransport()
    exporter, _ = _exporter(transport)
    first = ObservabilityExportEnvelope(
        record_kind=ExportRecordKind.TOOL_CALL,
        run_id="run-1",
        event_id="evt-1",
        tool_id="tool-a",
    )
    second = ObservabilityExportEnvelope(
        record_kind=ExportRecordKind.RAG_CALL,
        run_id="run-1",
        event_id="evt-2",
        tool_id="rag.retrieve",
    )

    await exporter.export(first)
    await exporter.export(second)

    assert transport.send_count == 2
    assert _attribute_map(transport.payloads[0])["intergrax.event_id"] == "evt-1"
    assert _attribute_map(transport.payloads[0])["intergrax.tool_id"] == "tool-a"
    assert _attribute_map(transport.payloads[1])["intergrax.event_id"] == "evt-2"
    assert _attribute_map(transport.payloads[1])["intergrax.tool_id"] == "rag.retrieve"


@pytest.mark.asyncio
async def test_payload_contains_stable_safe_platform_metadata_fields() -> None:
    transport = FakeOtlpTransport()
    exporter, _ = _exporter(transport)
    envelope = ObservabilityExportEnvelope(
        record_kind=ExportRecordKind.TOOL_CALL,
        run_id="run-1",
        task_id="task-1",
        agent_id="agent-1",
        capability="workspace.write",
        tool_id="workspace.write_file",
        event_type="tool.completed",
        status=ExportStatus.SUCCEEDED,
        latency_ms=12,
        counts={"hit_count": 2},
        schema_id="agent_run_trace.v1",
        sha256="abc123",
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        source_schema_id="agent_run_trace.v1",
        correlation_id="corr-1",
        event_id="evt-1",
        artifact_ref="artifacts/run-1/output.json",
        safe_relative_path="docs/README.md",
    )

    await exporter.export(envelope)

    attrs = _attribute_map(transport.payloads[0])
    assert attrs["intergrax.schema_version"] == "observability_export_envelope.v1"
    assert attrs["intergrax.record_kind"] == "tool_call"
    assert attrs["intergrax.run_id"] == "run-1"
    assert attrs["intergrax.task_id"] == "task-1"
    assert attrs["intergrax.agent_id"] == "agent-1"
    assert attrs["intergrax.capability"] == "workspace.write"
    assert attrs["intergrax.tool_id"] == "workspace.write_file"
    assert attrs["intergrax.event_type"] == "tool.completed"
    assert attrs["intergrax.status"] == "succeeded"
    assert attrs["intergrax.latency_ms"] == 12
    assert attrs["intergrax.counts.hit_count"] == 2
    assert attrs["intergrax.schema_id"] == "agent_run_trace.v1"
    assert attrs["intergrax.sha256"] == "abc123"
    assert attrs["intergrax.tenant_id"] == "tenant-a"
    assert attrs["intergrax.workspace_id"] == "workspace-a"
    assert attrs["intergrax.event_id"] == "evt-1"
    assert attrs["intergrax.artifact_ref"] == "artifacts/run-1/output.json"
    assert attrs["intergrax.safe_relative_path"] == "docs/README.md"
    log_record = transport.payloads[0]["resourceLogs"][0]["scopeLogs"][0]["logRecords"][0]
    assert "timeUnixNano" in log_record


@pytest.mark.asyncio
async def test_payload_contains_sanitized_application_attributes_with_namespaced_keys() -> None:
    transport = FakeOtlpTransport()
    exporter, _ = _exporter(transport)
    sanitized_attrs = ExampleApplicationObservabilityAttributes(
        result_count=5,
        strategy="safe",
        tags=["alpha", "beta"],
    )
    from intergrax.runtime.observability.export_attributes import sanitize_application_observability_attributes

    sanitized = sanitize_application_observability_attributes(sanitized_attrs).sanitized
    envelope = ObservabilityExportEnvelope(
        record_kind=ExportRecordKind.RUNTIME_EVENT,
        run_id="run-1",
        sanitized_application_attributes=sanitized,
    )

    await exporter.export(envelope)

    attrs = _attribute_map(transport.payloads[0])
    assert attrs[observability_attribute_key("example", "result_count")] == 5
    assert attrs[observability_attribute_key("example", "strategy")] == "safe"
    assert attrs[observability_attribute_key("example", "tags")] == ["alpha", "beta"]
    assert attrs["intergrax.application.namespace"] == "example"


@pytest.mark.asyncio
async def test_payload_does_not_contain_raw_application_attributes() -> None:
    transport = FakeOtlpTransport()
    exporter, _ = _exporter(transport)
    attrs = ExampleApplicationObservabilityAttributes(result_count=3)
    envelope = ObservabilityExportEnvelope(
        record_kind=ExportRecordKind.RUNTIME_EVENT,
        run_id="run-1",
        application_attributes=attrs,
    )

    await exporter.export(envelope)

    serialized = _payload_serialized(transport.payloads[0])
    assert "application_attributes" not in serialized


@pytest.mark.asyncio
async def test_payload_does_not_contain_raw_sensitive_content() -> None:
    transport = FakeOtlpTransport()
    exporter, _ = _exporter(transport)

    class SensitiveExampleAttributes(ApplicationObservabilityAttributes):
        namespace: str = "example"
        operation: str = "example.run"
        prompt: str = "secret prompt"
        result_count: int = 1

    attrs = SensitiveExampleAttributes()
    from intergrax.runtime.observability.export_attributes import sanitize_application_observability_attributes

    sanitized = sanitize_application_observability_attributes(attrs).sanitized
    envelope = ObservabilityExportEnvelope(
        record_kind=ExportRecordKind.RUNTIME_EVENT,
        run_id="run-1",
        tool_id="workspace.read_file",
        latency_ms=9,
        sanitized_application_attributes=sanitized,
    )

    await exporter.export(envelope)

    serialized = _payload_serialized(transport.payloads[0])
    assert envelope_is_content_safe(envelope)
    attrs = _attribute_map(transport.payloads[0])
    for key in FORBIDDEN_EXPORT_CONTENT_FIELDS:
        assert key not in attrs
    assert "secret prompt" not in serialized
    assert "raw body" not in serialized
    assert "C:\\Users\\secret" not in serialized


@pytest.mark.asyncio
async def test_works_through_try_export_with_enabled_policy_and_export_content_false() -> None:
    transport = FakeOtlpTransport()
    exporter, _ = _exporter(transport)
    attrs = ExampleApplicationObservabilityAttributes(result_count=5, strategy="safe")
    event = RuntimeEvent(
        task_id="task-1",
        run_id="run-1",
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
    policy = ObservabilityExportPolicy(enabled=True, export_content=False)

    result = await try_export_observability_envelope(
        envelope,
        exporter=exporter,
        policy=policy,
    )

    assert result.exported is True
    assert transport.send_count == 1
    attrs_map = _attribute_map(transport.payloads[0])
    assert attrs_map["intergrax.tool_id"] == "workspace.read_file"
    assert attrs_map["intergrax.latency_ms"] == 9
    assert attrs_map[observability_attribute_key("example", "result_count")] == 5
    serialized = _payload_serialized(transport.payloads[0])
    for key in FORBIDDEN_EXPORT_CONTENT_FIELDS:
        assert key not in attrs_map
    assert "secret prompt" not in serialized
    assert "raw body" not in serialized
    assert "C:\\Users\\secret" not in serialized


@pytest.mark.asyncio
async def test_disabled_policy_does_not_send_payloads_through_try_export() -> None:
    transport = FakeOtlpTransport()
    exporter, _ = _exporter(transport)
    envelope = ObservabilityExportEnvelope(record_kind=ExportRecordKind.RUNTIME_EVENT, run_id="run-1")

    result = await try_export_observability_envelope(
        envelope,
        exporter=exporter,
        policy=ObservabilityExportPolicy(enabled=False),
    )

    assert result.exported is False
    assert transport.send_count == 0
    assert transport.payloads == []


@pytest.mark.asyncio
async def test_transport_failure_is_isolated_through_try_export() -> None:
    transport = FakeOtlpTransport(fail=True)
    exporter, _ = _exporter(transport)
    envelope = ObservabilityExportEnvelope(record_kind=ExportRecordKind.RUNTIME_EVENT, run_id="run-1")
    policy = ObservabilityExportPolicy(enabled=True)

    result = await try_export_observability_envelope(
        envelope,
        exporter=exporter,
        policy=policy,
    )

    assert result.exported is False
    assert result.reason == "exporter_failed"
    assert transport.send_count == 1


def test_otlp_exporter_has_no_vendor_sdk_coupling() -> None:
    source = _OTLP_EXPORTER_PATH.read_text(encoding="utf-8")
    for token in _FORBIDDEN_VENDOR_TOKENS:
        assert token not in source, f"otlp_exporter.py contains forbidden vendor coupling token: {token}"


def test_fake_transport_does_not_perform_network_calls() -> None:
    transport = FakeOtlpTransport()
    assert isinstance(transport, OtlpTransport)
