# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.hooks.hook_registry import HookRegistry
from intergrax.runtime.observability.operator_wiring import (
    ObservabilityExportOperatorConfig,
    OtlpExportOperatorConfig,
)
from intergrax.runtime.observability.otlp_exporter import OtlpObservabilityExporterConfig, OtlpTransport
from intergrax.runtime.plugins.contract import RuntimePlugin
from local_workspace_application.host.observability_wiring import build_local_workspace_observability_plugins

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_LKW_OBSERVABILITY_WIRING_PATH = (
    _PROJECT_ROOT
    / "applications"
    / "local_workspace_application"
    / "host"
    / "observability_wiring.py"
)

_FORBIDDEN_VENDOR_TOKENS = (
    "langfuse",
    "arize",
    "phoenix",
    "elasticsearch",
    "opentelemetry",
    "integrations.providers.observability_backend",
)


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
) -> ObservabilityExportOperatorConfig:
    return ObservabilityExportOperatorConfig(
        enabled=True,
        export_content=export_content,
        backend_id="otlp",
        otlp=OtlpExportOperatorConfig(
            endpoint="https://collector.example/v1/logs",
            service_name="intergrax.lkw.test",
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


def test_default_none_config_registers_no_observability_export_plugin() -> None:
    assert build_local_workspace_observability_plugins(None) == ()


def test_disabled_observability_export_operator_config_registers_no_plugin() -> None:
    config = ObservabilityExportOperatorConfig(enabled=False)

    assert build_local_workspace_observability_plugins(config) == ()


def test_enabled_otlp_config_returns_exactly_one_runtime_plugin() -> None:
    plugins = build_local_workspace_observability_plugins(
        _enabled_config(),
        transport=FakeOtlpTransport(),
    )

    assert len(plugins) == 1
    assert isinstance(plugins[0], RuntimePlugin)
    assert plugins[0].plugin_id == "runtime.observability_export"


def test_helper_uses_platform_build_otlp_observability_export_runtime_plugin_path() -> None:
    config = _enabled_config()
    transport = FakeOtlpTransport()
    sentinel = RuntimePlugin(plugin_id="runtime.observability_export", version="1.0.0")

    with patch(
        "local_workspace_application.host.observability_wiring.build_otlp_observability_export_runtime_plugin",
        return_value=sentinel,
    ) as build_plugin:
        plugins = build_local_workspace_observability_plugins(config, transport=transport)

    build_plugin.assert_called_once_with(config, transport=transport)
    assert plugins == (sentinel,)


@pytest.mark.asyncio
async def test_runtime_event_through_plugin_reaches_injected_fake_transport() -> None:
    transport = FakeOtlpTransport()
    plugins = build_local_workspace_observability_plugins(_enabled_config(), transport=transport)
    assert len(plugins) == 1

    bus = RuntimeEventBus(record_history=False)
    plugins[0].register(bus, HookRegistry(), MagicMock())
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
async def test_export_content_true_in_app_config_is_still_forced_to_metadata_only_policy() -> None:
    transport = FakeOtlpTransport()
    plugins = build_local_workspace_observability_plugins(
        _enabled_config(export_content=True),
        transport=transport,
    )
    assert len(plugins) == 1

    bus = RuntimeEventBus(record_history=False)
    plugins[0].register(bus, HookRegistry(), MagicMock())
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


@pytest.mark.asyncio
async def test_raw_prompt_content_and_local_path_do_not_appear_in_exported_payload() -> None:
    transport = FakeOtlpTransport()
    plugins = build_local_workspace_observability_plugins(_enabled_config(), transport=transport)
    assert len(plugins) == 1

    bus = RuntimeEventBus(record_history=False)
    plugins[0].register(bus, HookRegistry(), MagicMock())
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
            "local_path": "/home/user/secret/doc.pdf",
            "safe_relative_path": "docs/public-note.md",
        },
    )

    await bus.publish(event)

    assert transport.send_count == 1
    serialized = json.dumps(transport.payloads[0])
    assert "secret prompt" not in serialized
    assert "raw body" not in serialized
    assert "/home/user/secret/doc.pdf" not in serialized
    assert "local_path" not in serialized


def test_no_lkw_specific_exporter_class_is_introduced() -> None:
    source = _LKW_OBSERVABILITY_WIRING_PATH.read_text(encoding="utf-8")
    assert "class " not in source or "ObservabilityExporter" not in source
    assert "LocalWorkspaceObservabilityExporter" not in source
    assert "LkwObservabilityExporter" not in source


def test_no_vendor_sdk_imports_are_introduced() -> None:
    source = _LKW_OBSERVABILITY_WIRING_PATH.read_text(encoding="utf-8")
    for token in _FORBIDDEN_VENDOR_TOKENS:
        assert token not in source, f"observability_wiring.py contains forbidden vendor coupling token: {token}"


def test_helper_does_not_call_otlp_http_transport_directly() -> None:
    source = _LKW_OBSERVABILITY_WIRING_PATH.read_text(encoding="utf-8")
    assert "OtlpHttpTransport" not in source
