# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.hooks.hook_registry import HookRegistry
from intergrax.runtime.observability.export_attributes import (
    ApplicationObservabilityAttributes,
    observability_attribute_key,
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
from intergrax.runtime.observability.operator_wiring import (
    ObservabilityExportBackend,
    ObservabilityExportOperatorConfig,
    ObservabilityExportOperatorConfigError,
    OtlpExportOperatorConfig,
    build_otlp_observability_export_runtime_plugin,
    build_otlp_observability_exporter,
)
from intergrax.runtime.observability.otlp_exporter import (
    OtlpObservabilityExporter,
    OtlpObservabilityExporterConfig,
    OtlpTransport,
)
from intergrax.runtime.observability.otlp_http_transport import OtlpHttpTransport

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_OPERATOR_WIRING_PATH = _PROJECT_ROOT / "intergrax" / "runtime" / "observability" / "operator_wiring.py"

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


class FakeOtlpTransport:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []
        self.configs: list[OtlpObservabilityExporterConfig] = []
        self.send_count = 0

    async def send(
        self,
        payload: dict[str, Any],
        *,
        config: OtlpObservabilityExporterConfig,
    ) -> None:
        self.send_count += 1
        self.payloads.append(payload)
        self.configs.append(config)


def _enabled_config(
    *,
    export_content: bool = False,
    otlp: OtlpExportOperatorConfig | None = None,
) -> ObservabilityExportOperatorConfig:
    return ObservabilityExportOperatorConfig(
        enabled=True,
        export_content=export_content,
        backend=ObservabilityExportBackend.OTLP,
        otlp=otlp
        or OtlpExportOperatorConfig(
            endpoint="https://collector.example/v1/logs",
            service_name="intergrax.test",
            service_version="1.0.0",
            environment="test",
            timeout_seconds=5.0,
            headers={"Authorization": "Bearer test-token"},
        ),
    )


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


def test_default_operator_config_is_disabled() -> None:
    config = ObservabilityExportOperatorConfig()

    assert config.enabled is False
    assert config.export_content is False
    assert config.backend is ObservabilityExportBackend.OTLP
    assert config.otlp is None


def test_disabled_config_does_not_create_active_export_wiring() -> None:
    config = ObservabilityExportOperatorConfig(enabled=False)

    assert build_otlp_observability_export_runtime_plugin(config) is None

    with pytest.raises(ObservabilityExportOperatorConfigError, match="disabled"):
        build_otlp_observability_exporter(config)


def test_enabled_otlp_config_creates_exporter_with_otlp_http_transport() -> None:
    config = _enabled_config()
    transport = FakeOtlpTransport()

    exporter = build_otlp_observability_exporter(config, transport=transport)

    assert isinstance(exporter, OtlpObservabilityExporter)
    assert exporter.config.endpoint == "https://collector.example/v1/logs"
    assert exporter.config.service_name == "intergrax.test"
    assert exporter.config.headers == {"Authorization": "Bearer test-token"}
    assert isinstance(exporter._transport, FakeOtlpTransport)  # noqa: SLF001

    default_exporter = build_otlp_observability_exporter(config)
    assert isinstance(default_exporter._transport, OtlpHttpTransport)  # noqa: SLF001


def test_created_runtime_plugin_uses_metadata_only_policy() -> None:
    source = _OPERATOR_WIRING_PATH.read_text(encoding="utf-8")
    assert "ObservabilityExportPolicy(enabled=True, export_content=False)" in source


@pytest.mark.asyncio
async def test_created_runtime_plugin_blocks_export_content_even_when_operator_config_requests_it() -> None:
    config = _enabled_config(export_content=True)
    transport = FakeOtlpTransport()
    plugin = build_otlp_observability_export_runtime_plugin(config, transport=transport)
    assert plugin is not None

    bus = RuntimeEventBus(record_history=False)
    plugin.register(bus, HookRegistry(), MagicMock())
    event = RuntimeEvent(
        task_id="task-1",
        run_id="run-1",
        event_type=RuntimeEventType.TOOL_COMPLETED,
        phase=ExecutionPhase.STEP_EXECUTION,
        payload={
            "tool_id": "workspace.read_file",
            "latency_ms": 9,
            "prompt": "secret prompt",
            "content": "raw body",
        },
    )

    await bus.publish(event)

    assert transport.send_count == 1
    serialized = json.dumps(transport.payloads[0])
    assert "secret prompt" not in serialized
    assert "raw body" not in serialized


def test_factory_refuses_when_otlp_backend_config_is_missing() -> None:
    config = ObservabilityExportOperatorConfig(enabled=True, otlp=None)

    with pytest.raises(ObservabilityExportOperatorConfigError, match="otlp export configuration"):
        build_otlp_observability_exporter(config)

    with pytest.raises(ObservabilityExportOperatorConfigError, match="otlp export configuration"):
        build_otlp_observability_export_runtime_plugin(config)


def test_factory_does_not_register_plugin_globally() -> None:
    source = _OPERATOR_WIRING_PATH.read_text(encoding="utf-8")
    assert "register_journal_export_plugin" not in source

    config = _enabled_config()
    plugin = build_otlp_observability_export_runtime_plugin(config, transport=FakeOtlpTransport())
    assert plugin is not None
    assert plugin.plugin_id == "runtime.observability_export"


@pytest.mark.asyncio
async def test_factory_does_not_perform_network_calls() -> None:
    transport = FakeOtlpTransport()
    exporter = build_otlp_observability_exporter(_enabled_config(), transport=transport)

    await exporter.export(ObservabilityExportEnvelope(record_kind=ExportRecordKind.RUNTIME_EVENT, run_id="run-1"))

    assert transport.send_count == 1
    assert isinstance(transport, OtlpTransport)


@pytest.mark.asyncio
async def test_headers_are_transport_config_only_not_in_payload() -> None:
    transport = FakeOtlpTransport()
    exporter = build_otlp_observability_exporter(_enabled_config(), transport=transport)

    await exporter.export(ObservabilityExportEnvelope(record_kind=ExportRecordKind.RUNTIME_EVENT, run_id="run-1"))

    assert transport.configs[0].headers == {"Authorization": "Bearer test-token"}
    serialized = json.dumps(transport.payloads[0])
    assert "Authorization" not in serialized
    assert "Bearer test-token" not in serialized
    assert "https://collector.example/v1/logs" not in serialized


@pytest.mark.asyncio
async def test_runtime_event_through_plugin_reaches_injected_transport() -> None:
    transport = FakeOtlpTransport()
    plugin = build_otlp_observability_export_runtime_plugin(_enabled_config(), transport=transport)
    assert plugin is not None

    bus = RuntimeEventBus(record_history=False)
    plugin.register(bus, HookRegistry(), MagicMock())
    event = RuntimeEvent(
        task_id="task-1",
        run_id="run-1",
        event_type=RuntimeEventType.TASK_COMPLETED,
        phase=ExecutionPhase.COMPLETION,
        payload={"journal_ref": {"event_count": 1}},
    )

    await bus.publish(event)

    assert transport.send_count == 1
    assert _attribute_map(transport.payloads[0])["intergrax.run_id"] == "run-1"


@pytest.mark.asyncio
async def test_raw_application_attributes_are_not_exported_only_sanitized_are_used() -> None:
    transport = FakeOtlpTransport()
    exporter = build_otlp_observability_exporter(_enabled_config(), transport=transport)
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
        },
    )
    envelope = envelope_from_runtime_event(event).model_copy(update={"application_attributes": attrs})
    result = await try_export_observability_envelope(
        envelope,
        exporter=exporter,
        policy=ObservabilityExportPolicy(enabled=True, export_content=False),
    )

    assert result.exported is True
    assert transport.send_count == 1
    serialized = json.dumps(transport.payloads[0])
    assert "application_attributes" not in serialized
    assert "secret prompt" not in serialized
    assert "raw body" not in serialized
    assert envelope_is_content_safe(result.envelope)  # type: ignore[arg-type]
    attr_map = _attribute_map(transport.payloads[0])
    assert attr_map[observability_attribute_key("example", "result_count")] == 5
    assert attr_map[observability_attribute_key("example", "strategy")] == "safe"
    for key in FORBIDDEN_EXPORT_CONTENT_FIELDS:
        assert key not in attr_map


def test_operator_wiring_has_no_vendor_sdk_coupling() -> None:
    source = _OPERATOR_WIRING_PATH.read_text(encoding="utf-8")
    for token in _FORBIDDEN_VENDOR_TOKENS:
        assert token not in source, f"operator_wiring.py contains forbidden vendor coupling token: {token}"
