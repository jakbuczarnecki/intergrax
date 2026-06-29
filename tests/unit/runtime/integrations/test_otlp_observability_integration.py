# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationContract,
    PlatformIntegrationKind,
    derive_platform_integration_id,
)
from intergrax.runtime.integrations.observability import (
    OBSERVABILITY_VENDOR_INTEGRATION_CONTRACT_SCHEMA,
    ObservabilityVendorIntegrationContract,
    ObservabilityVendorSignal,
)
from intergrax.runtime.integrations.observability_otlp import (
    OTLP_OBSERVABILITY_PROVIDER_ID,
    OtlpObservabilityIntegration,
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
    ObservabilityExporter,
    envelope_from_runtime_event,
    envelope_is_content_safe,
)
from intergrax.runtime.observability.export_policy import (
    ObservabilityExportPolicy,
    apply_observability_export_policy,
    try_export_observability_envelope,
)
from intergrax.runtime.observability.jsonl_exporter import JsonlObservabilityExporter
from intergrax.runtime.observability.otlp_exporter import (
    OtlpObservabilityExporter,
    OtlpObservabilityExporterConfig,
    OtlpTransport,
)

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_OTLP_INTEGRATION_PATH = _PROJECT_ROOT / "intergrax" / "runtime" / "integrations" / "observability_otlp.py"
_JSONL_EXPORTER_PATH = _PROJECT_ROOT / "intergrax" / "runtime" / "observability" / "jsonl_exporter.py"

_FORBIDDEN_VENDOR_IMPORT_PREFIXES = (
    "langfuse",
    "arize",
    "phoenix",
    "opentelemetry",
    "elasticsearch",
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


def _integration(
    transport: FakeOtlpTransport | None = None,
) -> tuple[OtlpObservabilityIntegration, FakeOtlpTransport]:
    active_transport = transport or FakeOtlpTransport()
    exporter = OtlpObservabilityExporter(_default_config(), active_transport)
    return OtlpObservabilityIntegration.from_exporter(exporter, enabled=True), active_transport


def _sanitized_envelope_with_attributes() -> ObservabilityExportEnvelope:
    attributes = ExampleApplicationObservabilityAttributes(result_count=3, strategy="safe")
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
        application_attributes=attributes,
        sanitized_application_attributes=sanitize_result.sanitized,
    )
    policy_result = apply_observability_export_policy(
        envelope,
        ObservabilityExportPolicy(enabled=True),
    )
    assert policy_result.exported and policy_result.envelope is not None
    return policy_result.envelope


def _attribute_map(payload: dict[str, Any]) -> dict[str, Any]:
    attrs = payload["resourceLogs"][0]["scopeLogs"][0]["logRecords"][0]["attributes"]
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


def test_otlp_integration_derives_from_observability_vendor_integration_contract() -> None:
    integration, _ = _integration()

    assert isinstance(integration, PlatformIntegrationContract)
    assert isinstance(integration, ObservabilityVendorIntegrationContract)
    assert isinstance(integration, OtlpObservabilityIntegration)


def test_otlp_integration_exposes_core_fields() -> None:
    integration, _ = _integration()

    assert integration.schema_id == OBSERVABILITY_VENDOR_INTEGRATION_CONTRACT_SCHEMA
    assert integration.provider_id == OTLP_OBSERVABILITY_PROVIDER_ID
    assert integration.integration_id == derive_platform_integration_id(
        OTLP_OBSERVABILITY_PROVIDER_ID,
        PlatformIntegrationKind.OBSERVABILITY_VENDOR.value,
    )
    assert integration.integration_kind == PlatformIntegrationKind.OBSERVABILITY_VENDOR.value
    assert PlatformIntegrationCapability.EXPORT in integration.capabilities
    assert PlatformIntegrationCapability.HEALTH_CHECK in integration.capabilities


def test_otlp_integration_exposes_supported_signals() -> None:
    integration, _ = _integration()

    assert ObservabilityVendorSignal.EVENTS in integration.supported_signals
    assert ObservabilityVendorSignal.LOGS in integration.supported_signals
    assert ObservabilityVendorSignal.LLM_EVENTS in integration.supported_signals
    public_view = integration.public_view()
    assert "events" in public_view["supported_signals"]
    assert "logs" in public_view["supported_signals"]


@pytest.mark.asyncio
async def test_otlp_integration_maps_sanitized_envelope_to_otlp_payload() -> None:
    transport = FakeOtlpTransport()
    integration, _ = _integration(transport)
    envelope = _sanitized_envelope_with_attributes()

    await integration.export(envelope)

    assert transport.send_count == 1
    attrs = _attribute_map(transport.payloads[0])
    assert attrs["intergrax.run_id"] == "run-1"
    assert attrs["intergrax.task_id"] == "task-1"
    assert attrs["intergrax.agent_id"] == "agent-1"
    assert attrs["intergrax.capability"] == "search"
    assert attrs["intergrax.event_type"] == "tool.completed"
    assert attrs["intergrax.status"] == "succeeded"
    assert attrs["intergrax.latency_ms"] == 42
    assert attrs["intergrax.counts.hit_count"] == 2
    assert attrs["intergrax.tool_id"] == "grep"


@pytest.mark.asyncio
async def test_otlp_integration_includes_sanitized_application_attributes_with_namespaced_keys() -> None:
    transport = FakeOtlpTransport()
    integration, _ = _integration(transport)
    envelope = _sanitized_envelope_with_attributes()

    await integration.export(envelope)

    attrs = _attribute_map(transport.payloads[0])
    assert attrs[observability_attribute_key("example", "result_count")] == 3
    assert attrs[observability_attribute_key("example", "strategy")] == "safe"
    assert attrs["intergrax.application.namespace"] == "example"


@pytest.mark.asyncio
async def test_otlp_integration_rejects_raw_application_attributes() -> None:
    integration, _ = _integration()
    attributes = ExampleApplicationObservabilityAttributes(result_count=1)
    envelope = ObservabilityExportEnvelope(
        record_kind=ExportRecordKind.RUNTIME_EVENT,
        application_attributes=attributes,
    )

    with pytest.raises(ValueError, match="raw application_attributes"):
        integration.map_envelope(envelope)


@pytest.mark.asyncio
async def test_otlp_integration_does_not_export_raw_application_attributes() -> None:
    transport = FakeOtlpTransport()
    integration, _ = _integration(transport)
    attrs = ExampleApplicationObservabilityAttributes(result_count=3)
    envelope = ObservabilityExportEnvelope(
        record_kind=ExportRecordKind.RUNTIME_EVENT,
        run_id="run-1",
        application_attributes=attrs,
    )

    with pytest.raises(ValueError, match="raw application_attributes"):
        await integration.export(envelope)

    assert transport.send_count == 0


@pytest.mark.asyncio
async def test_otlp_integration_does_not_export_forbidden_sensitive_content() -> None:
    transport = FakeOtlpTransport()
    integration, _ = _integration(transport)
    envelope = _sanitized_envelope_with_attributes()

    await integration.export(envelope)

    serialized = json.dumps(transport.payloads[0]).lower()
    forbidden_samples = (
        "raw prompt text",
        "secret-api-key",
        "c:\\users\\secret\\document.pdf",
        "/home/user/secret/document.pdf",
    )
    for sample in forbidden_samples:
        assert sample not in serialized

    attrs = _attribute_map(transport.payloads[0])
    for field_name in FORBIDDEN_EXPORT_CONTENT_FIELDS:
        assert field_name not in attrs

    assert "endpoint" not in serialized
    assert "headers" not in serialized
    assert "bearer test-token" not in serialized


@pytest.mark.asyncio
async def test_otlp_integration_delivers_through_injected_transport() -> None:
    transport = FakeOtlpTransport()
    integration, _ = _integration(transport)
    envelope = ObservabilityExportEnvelope(
        record_kind=ExportRecordKind.RUNTIME_EVENT,
        run_id="run-1",
        event_id="evt-1",
    )

    await integration.export(envelope)

    assert transport.send_count == 1
    assert isinstance(transport, OtlpTransport)
    assert _attribute_map(transport.payloads[0])["intergrax.run_id"] == "run-1"


@pytest.mark.asyncio
async def test_transport_failure_is_isolated_through_try_export() -> None:
    transport = FakeOtlpTransport(fail=True)
    integration, _ = _integration(transport)
    envelope = ObservabilityExportEnvelope(record_kind=ExportRecordKind.RUNTIME_EVENT, run_id="run-1")
    policy = ObservabilityExportPolicy(enabled=True)

    result = await try_export_observability_envelope(
        envelope,
        exporter=integration,
        policy=policy,
    )

    assert result.exported is False
    assert result.reason == "exporter_failed"
    assert transport.send_count == 1


@pytest.mark.asyncio
async def test_otlp_integration_implements_observability_exporter_protocol() -> None:
    integration, _ = _integration()
    assert isinstance(integration, ObservabilityExporter)


@pytest.mark.asyncio
async def test_works_through_try_export_with_runtime_event_and_policy() -> None:
    transport = FakeOtlpTransport()
    integration, _ = _integration(transport)
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
    envelope = envelope_from_runtime_event(event).model_copy(update={"application_attributes": attrs})
    policy = ObservabilityExportPolicy(enabled=True, export_content=False)

    result = await try_export_observability_envelope(
        envelope,
        exporter=integration,
        policy=policy,
    )

    assert result.exported is True
    assert transport.send_count == 1
    assert envelope_is_content_safe(result.envelope)  # type: ignore[arg-type]
    attrs_map = _attribute_map(transport.payloads[0])
    assert attrs_map["intergrax.tool_id"] == "workspace.read_file"
    assert attrs_map[observability_attribute_key("example", "result_count")] == 5
    serialized = json.dumps(transport.payloads[0])
    assert "secret prompt" not in serialized
    assert "raw body" not in serialized
    assert "C:\\Users\\secret" not in serialized


def test_jsonl_exporter_module_unchanged() -> None:
    assert JsonlObservabilityExporter is not None
    source = _JSONL_EXPORTER_PATH.read_text(encoding="utf-8")
    assert "ObservabilityVendorIntegrationContract" not in source


def test_no_vendor_sdk_imports_in_otlp_integration_module() -> None:
    source = _OTLP_INTEGRATION_PATH.read_text(encoding="utf-8").lower()
    for token in _FORBIDDEN_VENDOR_IMPORT_PREFIXES:
        assert f"import {token}" not in source
        assert f"from {token}" not in source
