# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
import inspect

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
    NoOpObservabilityExporter,
    ObservabilityExportEnvelope,
    envelope_from_runtime_event,
    envelope_is_content_safe,
)
from intergrax.runtime.observability.export_policy import (
    ObservabilityExportPolicy,
    try_export_observability_envelope,
)
from intergrax.integrations.providers.observability_backend.elasticsearch.integration import (
    ElasticsearchObservabilityIntegration,
)
from intergrax.integrations.providers.observability_backend.elasticsearch.transport import (
    ElasticsearchHttpObservabilityTransport,
)
from intergrax.runtime.observability.operator_wiring import (
    DEFAULT_OBSERVABILITY_EXPORT_BACKEND_REGISTRY,
    ElasticsearchExportOperatorConfig,
    ObservabilityExportBackendRegistry,
    ObservabilityExportBackendRegistryError,
    ObservabilityExportOperatorConfig,
    ObservabilityExportOperatorConfigError,
    OtlpExportOperatorConfig,
    build_observability_export_integration,
    build_observability_export_runtime_plugin,
    build_otlp_observability_export_runtime_plugin,
    build_otlp_observability_exporter,
    build_otlp_observability_integration,
    parse_observability_export_backend_id,
)
from intergrax.runtime.observability.elasticsearch_export_wiring import (
    build_elasticsearch_observability_integration,
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

_FORBIDDEN_VENDOR_SDK_TOKENS = (
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


def _enabled_elasticsearch_config(
    *,
    export_content: bool = False,
    elasticsearch: ElasticsearchExportOperatorConfig | None = None,
) -> ObservabilityExportOperatorConfig:
    return ObservabilityExportOperatorConfig(
        enabled=True,
        export_content=export_content,
        backend_id="elasticsearch",
        elasticsearch=elasticsearch
        or ElasticsearchExportOperatorConfig(
            base_url="http://elasticsearch.local:9200",
            index="logs-*",
            timeout_seconds=5.0,
        ),
    )


class FakeElasticsearchTransport:
    async def send_observability_payload(self, payload: object) -> None:
        return None


def _enabled_config(
    *,
    export_content: bool = False,
    otlp: OtlpExportOperatorConfig | None = None,
) -> ObservabilityExportOperatorConfig:
    return ObservabilityExportOperatorConfig(
        enabled=True,
        export_content=export_content,
        backend_id="otlp",
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


def test_parse_observability_export_backend_id_accepts_otlp() -> None:
    assert parse_observability_export_backend_id("otlp") == "otlp"


def test_parse_observability_export_backend_id_trims_and_normalizes() -> None:
    assert parse_observability_export_backend_id(" elasticsearch ") == "elasticsearch"


def test_parse_observability_export_backend_id_normalizes_uppercase() -> None:
    assert parse_observability_export_backend_id("ACME_OBSERVABILITY") == "acme_observability"


def test_parse_observability_export_backend_id_rejects_empty() -> None:
    with pytest.raises(
        ObservabilityExportOperatorConfigError,
        match="invalid observability export backend id: ''",
    ):
        parse_observability_export_backend_id("")


def test_parse_observability_export_backend_id_rejects_invalid_format() -> None:
    with pytest.raises(
        ObservabilityExportOperatorConfigError,
        match="invalid observability export backend id: 'foo/bar'",
    ):
        parse_observability_export_backend_id("foo/bar")


def test_default_registry_contains_otlp() -> None:
    builder = DEFAULT_OBSERVABILITY_EXPORT_BACKEND_REGISTRY.get("otlp")
    assert callable(builder)


def test_default_registry_contains_elasticsearch() -> None:
    builder = DEFAULT_OBSERVABILITY_EXPORT_BACKEND_REGISTRY.get("elasticsearch")
    assert callable(builder)


def test_registry_registers_builder_by_backend_id() -> None:
    registry = ObservabilityExportBackendRegistry()
    called: list[str] = []

    def _builder(config: ObservabilityExportOperatorConfig) -> object:
        called.append(config.backend_id)
        return object()

    registry.register("acme_observability", _builder)
    config = ObservabilityExportOperatorConfig(enabled=True, backend_id="acme_observability")
    registry.get("acme_observability")(config)
    assert called == ["acme_observability"]


def test_duplicate_backend_id_registration_fails() -> None:
    registry = ObservabilityExportBackendRegistry()

    registry.register("otlp", lambda _config: object())
    with pytest.raises(
        ObservabilityExportBackendRegistryError,
        match="already registered for 'otlp'",
    ):
        registry.register("otlp", lambda _config: object())


def test_missing_builder_raises_clear_error() -> None:
    registry = ObservabilityExportBackendRegistry()
    with pytest.raises(
        ObservabilityExportBackendRegistryError,
        match="no observability export backend builder registered for 'custom_or_unregistered'",
    ):
        registry.get("custom_or_unregistered")


def test_valid_non_registered_backend_id_fails_as_missing_builder() -> None:
    config = ObservabilityExportOperatorConfig(
        enabled=True,
        backend_id="custom_or_unregistered",
    )

    with pytest.raises(
        ObservabilityExportBackendRegistryError,
        match="no observability export backend builder registered for 'custom_or_unregistered'",
    ):
        build_observability_export_runtime_plugin(config)


def test_generic_build_observability_export_runtime_plugin_has_no_transport_argument() -> None:
    sig = inspect.signature(build_observability_export_runtime_plugin)
    assert "transport" not in sig.parameters


def test_generic_build_observability_export_integration_has_no_transport_argument() -> None:
    sig = inspect.signature(build_observability_export_integration)
    assert "transport" not in sig.parameters


def test_generic_builder_registry_does_not_pass_transport_kwarg_to_custom_builders() -> None:
    registry = ObservabilityExportBackendRegistry()
    called_with: list[ObservabilityExportOperatorConfig] = []

    def _builder(config: ObservabilityExportOperatorConfig) -> object:
        called_with.append(config)
        return NoOpObservabilityExporter()

    registry.register("acme_observability", _builder)
    config = ObservabilityExportOperatorConfig(enabled=True, backend_id="acme_observability")
    build_observability_export_integration(config, registry=registry)
    assert called_with == [config]


def test_custom_registry_builder_can_build_non_otlp_backend_plugin() -> None:
    registry = ObservabilityExportBackendRegistry()
    registry.register("acme_observability", lambda _config: NoOpObservabilityExporter())
    config = ObservabilityExportOperatorConfig(enabled=True, backend_id="acme_observability")

    plugin = build_observability_export_runtime_plugin(config, registry=registry)

    assert plugin is not None
    assert plugin.plugin_id == "runtime.observability_export"


def test_default_otlp_registry_still_builds_otlp_runtime_plugin() -> None:
    plugin = build_observability_export_runtime_plugin(_enabled_config())

    assert plugin is not None
    assert plugin.plugin_id == "runtime.observability_export"


def test_otlp_registry_closure_can_inject_fake_transport() -> None:
    transport = FakeOtlpTransport()
    registry = ObservabilityExportBackendRegistry()
    registry.register(
        "otlp",
        lambda config: build_otlp_observability_integration(config, transport=transport),
    )

    plugin = build_observability_export_runtime_plugin(_enabled_config(), registry=registry)

    assert plugin is not None
    assert plugin.plugin_id == "runtime.observability_export"


def test_default_operator_config_is_disabled() -> None:
    config = ObservabilityExportOperatorConfig()

    assert config.enabled is False
    assert config.export_content is False
    assert config.backend_id == "otlp"
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
    for token in _FORBIDDEN_VENDOR_SDK_TOKENS:
        assert token not in source, f"operator_wiring.py contains forbidden vendor coupling token: {token}"


def test_elasticsearch_registry_builds_elasticsearch_observability_integration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = FakeElasticsearchTransport()
    monkeypatch.setattr(
        "intergrax.integrations.providers.observability_backend.elasticsearch.bundle.create_elasticsearch_observability_transport",
        lambda **_kwargs: sentinel,
    )

    integration = build_observability_export_integration(_enabled_elasticsearch_config())

    assert isinstance(integration, ElasticsearchObservabilityIntegration)
    assert integration._transport is sentinel  # noqa: SLF001


def test_elasticsearch_builder_uses_provider_transport_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = ElasticsearchHttpObservabilityTransport(MagicMock(), index="logs-*")
    create_transport = MagicMock(return_value=sentinel)
    monkeypatch.setattr(
        "intergrax.integrations.providers.observability_backend.elasticsearch.bundle.create_elasticsearch_observability_transport",
        create_transport,
    )

    integration = build_elasticsearch_observability_integration(_enabled_elasticsearch_config())

    create_transport.assert_called_once()
    assert isinstance(integration, ElasticsearchObservabilityIntegration)
    assert integration._transport is sentinel  # noqa: SLF001


def test_elasticsearch_builder_forwards_retry_policy_to_transport_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from intergrax.integrations.providers.observability_backend.elasticsearch.config import (
        ElasticsearchRetryPolicy,
    )

    sentinel = ElasticsearchHttpObservabilityTransport(MagicMock(), index="logs-*")
    create_transport = MagicMock(return_value=sentinel)
    monkeypatch.setattr(
        "intergrax.integrations.providers.observability_backend.elasticsearch.bundle.create_elasticsearch_observability_transport",
        create_transport,
    )
    retry = ElasticsearchRetryPolicy(
        enabled=False,
        max_attempts=2,
        initial_backoff_seconds=0.5,
        max_backoff_seconds=1.0,
    )
    config = _enabled_elasticsearch_config(
        elasticsearch=ElasticsearchExportOperatorConfig(
            base_url="http://elasticsearch.local:9200",
            index="logs-*",
            timeout_seconds=5.0,
            retry_enabled=retry.enabled,
            retry_max_attempts=retry.max_attempts,
            retry_initial_backoff_seconds=retry.initial_backoff_seconds,
            retry_max_backoff_seconds=retry.max_backoff_seconds,
        ),
    )

    build_elasticsearch_observability_integration(config)

    create_transport.assert_called_once()
    assert create_transport.call_args.kwargs["retry_policy"] == retry


def test_elasticsearch_registry_closure_can_inject_fake_transport() -> None:
    transport = FakeElasticsearchTransport()
    registry = ObservabilityExportBackendRegistry()
    registry.register(
        "elasticsearch",
        lambda config: build_elasticsearch_observability_integration(config, transport=transport),
    )

    plugin = build_observability_export_runtime_plugin(
        _enabled_elasticsearch_config(),
        registry=registry,
    )

    assert plugin is not None
    assert plugin.plugin_id == "runtime.observability_export"


def test_disabled_elasticsearch_config_does_not_create_runtime_plugin() -> None:
    config = ObservabilityExportOperatorConfig(
        enabled=False,
        backend_id="elasticsearch",
        elasticsearch=ElasticsearchExportOperatorConfig(
            base_url="http://elasticsearch.local:9200",
            index="logs-*",
        ),
    )

    assert build_observability_export_runtime_plugin(config) is None


def test_elasticsearch_builder_fails_fast_when_base_url_missing() -> None:
    config = ObservabilityExportOperatorConfig(
        enabled=True,
        backend_id="elasticsearch",
        elasticsearch=ElasticsearchExportOperatorConfig(base_url="", index="logs-*"),
    )

    with pytest.raises(ObservabilityExportOperatorConfigError, match="base_url is required"):
        build_elasticsearch_observability_integration(config)


def test_elasticsearch_builder_fails_fast_when_index_missing() -> None:
    config = ObservabilityExportOperatorConfig(
        enabled=True,
        backend_id="elasticsearch",
        elasticsearch=ElasticsearchExportOperatorConfig(
            base_url="http://elasticsearch.local:9200",
            index="",
        ),
    )

    with pytest.raises(ObservabilityExportOperatorConfigError, match="index is required"):
        build_elasticsearch_observability_integration(config)
