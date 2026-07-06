# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import inspect
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.hooks.hook_registry import HookRegistry
from intergrax.runtime.observability.operator_wiring import (
    ElasticsearchExportOperatorConfig,
    ObservabilityExportBackendRegistry,
    ObservabilityExportBackendRegistryError,
    ObservabilityExportOperatorConfig,
    OtlpExportOperatorConfig,
    SentryExportOperatorConfig,
    build_otlp_observability_integration,
)
from intergrax.runtime.observability.elasticsearch_export_wiring import (
    build_elasticsearch_observability_integration,
)
from intergrax.runtime.observability.export_policy import ObservabilityExportPolicy
from intergrax.runtime.observability.problem_reporter import ProblemReportContext, report_problem
from intergrax.runtime.observability.sentry_export_wiring import (
    build_sentry_observability_integration,
)
from intergrax.runtime.observability.otlp_exporter import OtlpObservabilityExporterConfig
from intergrax.runtime.plugins.contract import RuntimePlugin
from local_workspace_application.host.observability_wiring import build_local_workspace_observability_plugins
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_LKW_OBSERVABILITY_WIRING_PATH = (
    _PROJECT_ROOT
    / "applications"
    / "local_workspace_application"
    / "host"
    / "observability_wiring.py"
)
_LKW_SETTINGS_PATH = (
    _PROJECT_ROOT
    / "applications"
    / "local_workspace_application"
    / "host"
    / "settings.py"
)

_FORBIDDEN_VENDOR_TOKENS = (
    "langfuse",
    "arize",
    "phoenix",
    "elasticsearch",
    "opentelemetry",
    "integrations.providers.observability_backend",
    "sentry_sdk",
    "integrations.providers.observability_backend.sentry",
    "SentrySdkObservabilityTransport",
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


def _enabled_elasticsearch_config() -> ObservabilityExportOperatorConfig:
    return ObservabilityExportOperatorConfig(
        enabled=True,
        export_content=False,
        backend_id="elasticsearch",
        elasticsearch=ElasticsearchExportOperatorConfig(
            base_url="http://elasticsearch.local:9200",
            index="logs-*",
            timeout_seconds=5.0,
        ),
    )


def _enabled_sentry_config() -> ObservabilityExportOperatorConfig:
    return ObservabilityExportOperatorConfig(
        enabled=True,
        export_content=False,
        backend_id="sentry",
        sentry=SentryExportOperatorConfig(
            dsn="https://example@sentry.io/1",
            environment="test",
            release="lkw-test",
            server_name="local-lkw",
            shutdown_timeout_seconds=2.0,
            flush_after_capture=True,
        ),
    )


class FakeElasticsearchTransport:
    async def send_observability_payload(self, payload: object) -> None:
        return None


def _elasticsearch_registry_with_transport(
    transport: FakeElasticsearchTransport,
) -> ObservabilityExportBackendRegistry:
    registry = ObservabilityExportBackendRegistry()
    registry.register(
        "elasticsearch",
        lambda config: build_elasticsearch_observability_integration(config, transport=transport),
    )
    return registry


class FakeSentryTransport:
    def __init__(self) -> None:
        self.payloads: list[Any] = []

    async def send_observability_payload(self, payload: object) -> None:
        self.payloads.append(payload)


def _sentry_registry_with_transport(
    transport: FakeSentryTransport,
) -> ObservabilityExportBackendRegistry:
    registry = ObservabilityExportBackendRegistry()
    registry.register(
        "sentry",
        lambda config: build_sentry_observability_integration(config, transport=transport),
    )
    return registry


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


def _otlp_registry_with_transport(transport: FakeOtlpTransport) -> ObservabilityExportBackendRegistry:
    registry = ObservabilityExportBackendRegistry()
    registry.register(
        "otlp",
        lambda config: build_otlp_observability_integration(config, transport=transport),
    )
    return registry


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


def test_lkw_observability_wiring_does_not_expose_otlp_transport() -> None:
    source = _LKW_OBSERVABILITY_WIRING_PATH.read_text(encoding="utf-8")
    assert "OtlpTransport" not in source

    sig = inspect.signature(build_local_workspace_observability_plugins)
    assert "transport" not in sig.parameters


def test_enabled_elasticsearch_config_returns_exactly_one_runtime_plugin() -> None:
    plugins = build_local_workspace_observability_plugins(
        _enabled_elasticsearch_config(),
        registry=_elasticsearch_registry_with_transport(FakeElasticsearchTransport()),
    )

    assert len(plugins) == 1
    assert isinstance(plugins[0], RuntimePlugin)
    assert plugins[0].plugin_id == "runtime.observability_export"


def test_enabled_sentry_config_returns_exactly_one_runtime_plugin() -> None:
    plugins = build_local_workspace_observability_plugins(
        _enabled_sentry_config(),
        registry=_sentry_registry_with_transport(FakeSentryTransport()),
    )

    assert len(plugins) == 1
    assert isinstance(plugins[0], RuntimePlugin)
    assert plugins[0].plugin_id == "runtime.observability_export"


def test_enabled_otlp_config_returns_exactly_one_runtime_plugin() -> None:
    plugins = build_local_workspace_observability_plugins(_enabled_config())

    assert len(plugins) == 1
    assert isinstance(plugins[0], RuntimePlugin)
    assert plugins[0].plugin_id == "runtime.observability_export"


def test_helper_uses_platform_build_observability_export_runtime_plugin_path() -> None:
    config = _enabled_config()
    sentinel = RuntimePlugin(plugin_id="runtime.observability_export", version="1.0.0")

    with patch(
        "local_workspace_application.host.observability_wiring.build_observability_export_runtime_plugin",
        return_value=sentinel,
    ) as build_plugin:
        plugins = build_local_workspace_observability_plugins(config)

    build_plugin.assert_called_once_with(config, registry=None)
    assert plugins == (sentinel,)


def test_custom_registry_closure_injects_fake_transport() -> None:
    transport = FakeOtlpTransport()
    plugins = build_local_workspace_observability_plugins(
        _enabled_config(),
        registry=_otlp_registry_with_transport(transport),
    )

    assert len(plugins) == 1
    assert plugins[0].plugin_id == "runtime.observability_export"


def test_unregistered_backend_id_fails_at_plugin_build() -> None:
    config = ObservabilityExportOperatorConfig(enabled=True, backend_id="acme_observability")

    with pytest.raises(
        ObservabilityExportBackendRegistryError,
        match="no observability export backend builder registered for 'acme_observability'",
    ):
        build_local_workspace_observability_plugins(config)


@pytest.mark.asyncio
async def test_controlled_problem_envelope_reaches_fake_sentry_transport() -> None:
    transport = FakeSentryTransport()
    integration = build_sentry_observability_integration(
        _enabled_sentry_config(),
        transport=transport,
    )
    context = ProblemReportContext(
        run_id="run-proof-1",
        task_id="task-proof-1",
        agent_id="local_search",
        capability="local.workspace.search",
        correlation_id="corr-proof-1",
    )

    result = await report_problem(
        context=context,
        problem_kind="lkw.retrieve_failed",
        error_code="LKW_RETRIEVE_FAILED",
        source_layer="lkw",
        source_component="retrieve",
        tool_id="rag.retrieve",
        exporter=integration,
        policy=ObservabilityExportPolicy(enabled=True, export_content=False),
    )

    assert result.exported is True
    assert len(transport.payloads) == 1
    payload = transport.payloads[0]
    assert payload.record_type == "problem_signal"
    assert payload.problem_kind == "lkw.retrieve_failed"
    assert payload.problem_error_code == "LKW_RETRIEVE_FAILED"
    assert payload.run_id == "run-proof-1"
    assert payload.correlation_id == "corr-proof-1"


@pytest.mark.asyncio
async def test_export_content_true_with_sentry_is_still_forced_to_metadata_only_policy() -> None:
    transport = FakeSentryTransport()
    plugins = build_local_workspace_observability_plugins(
        replace(_enabled_sentry_config(), export_content=True),
        registry=_sentry_registry_with_transport(transport),
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

    assert len(transport.payloads) == 1
    serialized = json.dumps(transport.payloads[0].model_dump(mode="json"))
    assert "secret prompt" not in serialized
    assert "raw body" not in serialized


@pytest.mark.asyncio
async def test_runtime_event_through_plugin_reaches_injected_fake_transport() -> None:
    transport = FakeOtlpTransport()
    plugins = build_local_workspace_observability_plugins(
        _enabled_config(),
        registry=_otlp_registry_with_transport(transport),
    )
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
        registry=_otlp_registry_with_transport(transport),
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
    plugins = build_local_workspace_observability_plugins(
        _enabled_config(),
        registry=_otlp_registry_with_transport(transport),
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


@pytest.fixture
def _clear_observability_env() -> Generator[None, None, None]:
    keys = [k for k in os.environ if k.startswith("LOCAL_WORKSPACE_OBSERVABILITY_")]
    saved = {k: os.environ[k] for k in keys}
    for k in keys:
        del os.environ[k]
    yield
    for k in saved:
        os.environ[k] = saved[k]
    for k in [k for k in os.environ if k.startswith("LOCAL_WORKSPACE_OBSERVABILITY_")]:
        if k not in saved:
            del os.environ[k]


def test_lkw_default_env_produces_provider_retry_defaults(_clear_observability_env: None) -> None:
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_ENABLED"] = "true"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_BACKEND"] = "elasticsearch"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_ELASTICSEARCH_URL"] = "http://elasticsearch.local:9200"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_ELASTICSEARCH_INDEX"] = "logs-*"

    settings = LocalWorkspaceBackendSettings.from_env()
    config = settings.build_observability_export_config()

    assert config is not None
    assert config.elasticsearch is not None
    elasticsearch = config.elasticsearch
    assert elasticsearch.retry_enabled is True
    assert elasticsearch.retry_max_attempts == 3
    assert elasticsearch.retry_initial_backoff_seconds == 0.25
    assert elasticsearch.retry_max_backoff_seconds == 2.0
    assert elasticsearch.failed_delivery_file_path is None


def test_lkw_env_retry_overrides_are_passed_into_elasticsearch_export_operator_config(
    _clear_observability_env: None,
) -> None:
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_ENABLED"] = "true"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_BACKEND"] = "elasticsearch"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_ELASTICSEARCH_URL"] = "http://elasticsearch.local:9200"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_ELASTICSEARCH_INDEX"] = "logs-*"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_ELASTICSEARCH_RETRY_ENABLED"] = "false"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_ELASTICSEARCH_RETRY_MAX_ATTEMPTS"] = "5"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_ELASTICSEARCH_RETRY_INITIAL_BACKOFF_SECONDS"] = "0.5"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_ELASTICSEARCH_RETRY_MAX_BACKOFF_SECONDS"] = "4.0"

    settings = LocalWorkspaceBackendSettings.from_env()
    config = settings.build_observability_export_config()

    assert config is not None
    assert config.elasticsearch is not None
    elasticsearch = config.elasticsearch
    assert elasticsearch.retry_enabled is False
    assert elasticsearch.retry_max_attempts == 5
    assert elasticsearch.retry_initial_backoff_seconds == 0.5
    assert elasticsearch.retry_max_backoff_seconds == 4.0


def test_lkw_env_failed_delivery_file_path_override_passed_to_operator_config(
    _clear_observability_env: None,
) -> None:
    configured_path = "applications/local_workspace_application/.observability/elasticsearch/failed-deliveries.jsonl"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_ENABLED"] = "true"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_BACKEND"] = "elasticsearch"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_ELASTICSEARCH_URL"] = "http://elasticsearch.local:9200"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_ELASTICSEARCH_INDEX"] = "logs-*"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_ELASTICSEARCH_FAILED_DELIVERY_FILE_PATH"] = configured_path

    settings = LocalWorkspaceBackendSettings.from_env()
    config = settings.build_observability_export_config()

    assert config is not None
    assert config.elasticsearch is not None
    assert config.elasticsearch.failed_delivery_file_path == configured_path


@pytest.mark.parametrize("raw_value", ["", "   "])
def test_lkw_empty_whitespace_failed_delivery_file_path_disables_sink(
    _clear_observability_env: None,
    raw_value: str,
) -> None:
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_ENABLED"] = "true"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_BACKEND"] = "elasticsearch"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_ELASTICSEARCH_URL"] = "http://elasticsearch.local:9200"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_ELASTICSEARCH_INDEX"] = "logs-*"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_ELASTICSEARCH_FAILED_DELIVERY_FILE_PATH"] = raw_value

    settings = LocalWorkspaceBackendSettings.from_env()
    config = settings.build_observability_export_config()

    assert config is not None
    assert config.elasticsearch is not None
    assert config.elasticsearch.failed_delivery_file_path is None


def test_lkw_settings_do_not_import_elasticsearch_failed_delivery_sink() -> None:
    source = _LKW_SETTINGS_PATH.read_text(encoding="utf-8")
    assert "FileElasticsearchFailedDeliverySink" not in source
    assert "integrations.providers.observability_backend.elasticsearch.failed_delivery" not in source
    assert "json.dump" not in source
    assert "open(" not in source


def test_lkw_settings_do_not_import_sentry_provider_sdk() -> None:
    source = _LKW_SETTINGS_PATH.read_text(encoding="utf-8")
    forbidden = (
        "sentry_sdk",
        "integrations.providers.observability_backend.sentry",
        "SentrySdkObservabilityTransport",
        "create_sentry_observability_transport",
        "build_sentry_observability_integration",
    )
    for token in forbidden:
        assert token not in source, f"settings.py contains forbidden Sentry coupling token: {token}"
    assert "observability_sentry" in source
    assert "LOCAL_WORKSPACE_OBSERVABILITY_SENTRY_DSN" in source or "OBSERVABILITY_SENTRY_DSN" in source
