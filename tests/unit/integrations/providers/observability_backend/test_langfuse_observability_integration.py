# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any

import pytest

from intergrax.integrations._shared.conformance import assert_observability_backend
from intergrax.integrations.providers.observability_backend.langfuse.bundle import (
    create_langfuse_observability_backend,
    create_langfuse_observability_integration,
)
from intergrax.integrations.providers.observability_backend.langfuse.integration import (
    LANGFUSE_OBSERVABILITY_PROVIDER_ID,
    LANGFUSE_SUPPORTED_SIGNALS,
    LangfuseObservabilityIntegration,
)
from intergrax.integrations.providers.observability_backend.langfuse.manifest import MANIFEST
from intergrax.integrations.providers.observability_backend.langfuse.register import (
    register_langfuse_integration,
)
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationContract,
    PlatformIntegrationKind,
    derive_platform_integration_id,
)
from intergrax.runtime.integrations.observability import (
    OBSERVABILITY_VENDOR_INTEGRATION_CONTRACT_SCHEMA,
    ObservabilityVendorIntegrationContract,
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

_PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[5]
_LANGFUSE_INTEGRATION_PATH = (
    _PROJECT_ROOT
    / "intergrax"
    / "integrations"
    / "providers"
    / "observability_backend"
    / "langfuse"
    / "integration.py"
)

_FORBIDDEN_VENDOR_IMPORT_PREFIXES = ("langfuse",)


class ExampleWorkspaceObservabilityAttributes(ApplicationObservabilityAttributes):
    namespace: str = "local_workspace"
    operation: str | None = "index_job"
    file_count: int | None = None


class FakeLangfuseTransport:
    def __init__(self) -> None:
        self.payloads: list[ObservabilityVendorPayload] = []
        self.send_count = 0

    async def send_observability_payload(self, payload: ObservabilityVendorPayload) -> None:
        self.send_count += 1
        self.payloads.append(payload)


class _FakeObsClient:
    def query_instant(self, promql: str, *, eval_time: float | None = None) -> float:
        return 3.0

    def query_range(
        self,
        promql: str,
        *,
        start: float,
        end: float,
        step: str,
    ) -> list[dict[str, float]]:
        return [{"timestamp": start, "value": 3.0}]

    def query_traces(self, *, limit: int = 20, name: str | None = None) -> Any:
        from intergrax.integrations.contracts.observability_backend import TraceQueryResult

        return TraceQueryResult(traces=[])


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
        application_attributes=attributes,
        sanitized_application_attributes=sanitize_result.sanitized,
    )
    policy_result = apply_observability_export_policy(
        envelope,
        ObservabilityExportPolicy(enabled=True),
    )
    assert policy_result.exported and policy_result.envelope is not None
    return policy_result.envelope


def test_langfuse_integration_derives_from_observability_vendor_contract() -> None:
    integration = LangfuseObservabilityIntegration.from_transport(FakeLangfuseTransport())

    assert isinstance(integration, PlatformIntegrationContract)
    assert isinstance(integration, ObservabilityVendorIntegrationContract)
    assert isinstance(integration, LangfuseObservabilityIntegration)


def test_langfuse_integration_exposes_provider_id() -> None:
    integration = LangfuseObservabilityIntegration.from_transport(FakeLangfuseTransport())

    assert integration.provider_id == LANGFUSE_OBSERVABILITY_PROVIDER_ID == "langfuse"


def test_langfuse_integration_exposes_integration_kind() -> None:
    integration = LangfuseObservabilityIntegration.from_transport(FakeLangfuseTransport())

    assert integration.integration_kind == PlatformIntegrationKind.OBSERVABILITY_VENDOR.value
    assert integration.integration_id == derive_platform_integration_id("langfuse", "observability_vendor")
    assert integration.schema_id == OBSERVABILITY_VENDOR_INTEGRATION_CONTRACT_SCHEMA


def test_langfuse_integration_exposes_supported_signals() -> None:
    integration = LangfuseObservabilityIntegration.from_transport(FakeLangfuseTransport())

    assert integration.supported_signals == LANGFUSE_SUPPORTED_SIGNALS
    assert integration.supported_signals == (
        ObservabilityVendorSignal.EVENTS,
        ObservabilityVendorSignal.TRACES,
        ObservabilityVendorSignal.LLM_EVENTS,
    )
    public_view = integration.public_view()
    assert public_view["supported_signals"] == ["events", "traces", "llm_events"]


def test_langfuse_integration_maps_sanitized_envelope_to_payload() -> None:
    integration = LangfuseObservabilityIntegration.from_transport(FakeLangfuseTransport())
    envelope = _sanitized_envelope_with_attributes()

    mapping = integration.map_envelope(envelope)
    payload = mapping.payload

    assert payload.provider_id == "langfuse"
    assert payload.integration_id == "langfuse:observability_vendor"
    assert payload.integration_kind == PlatformIntegrationKind.OBSERVABILITY_VENDOR.value
    assert payload.record_type == ExportRecordKind.RUNTIME_EVENT.value
    assert payload.run_id == "run-1"
    assert payload.task_id == "task-1"
    assert payload.agent_id == "agent-1"
    assert mapping.signal == ObservabilityVendorSignal.EVENTS


def test_langfuse_payload_includes_sanitized_application_attributes_with_namespaced_keys() -> None:
    integration = LangfuseObservabilityIntegration.from_transport(FakeLangfuseTransport())
    envelope = _sanitized_envelope_with_attributes()

    payload = integration.map_envelope(envelope).payload

    assert payload.sanitized_application_attributes is not None
    attributes = payload.sanitized_application_attributes.attributes
    assert observability_attribute_key("local_workspace", "file_count") in attributes
    assert attributes[observability_attribute_key("local_workspace", "file_count")] == 3


def test_langfuse_map_envelope_rejects_raw_application_attributes() -> None:
    integration = LangfuseObservabilityIntegration.from_transport(FakeLangfuseTransport())
    attributes = ExampleWorkspaceObservabilityAttributes(file_count=1)
    envelope = ObservabilityExportEnvelope(
        record_kind=ExportRecordKind.RUNTIME_EVENT,
        application_attributes=attributes,
    )

    with pytest.raises(ValueError, match="raw application_attributes"):
        integration.map_envelope(envelope)


def test_langfuse_payload_does_not_include_sensitive_content() -> None:
    integration = LangfuseObservabilityIntegration.from_transport(FakeLangfuseTransport())
    envelope = _sanitized_envelope_with_attributes()

    payload = integration.map_envelope(envelope).payload
    serialized = json.dumps(payload.model_dump(mode="json")).lower()

    forbidden_samples = (
        "raw prompt text",
        "secret-api-key",
        "c:\\users\\secret\\document.pdf",
        "/home/user/secret/document.pdf",
    )
    for sample in forbidden_samples:
        assert sample not in serialized

    for field_name in FORBIDDEN_EXPORT_CONTENT_FIELDS:
        assert f'"{field_name}"' not in serialized

    dumped = payload.model_dump()
    assert "application_attributes" not in dumped


@pytest.mark.asyncio
async def test_langfuse_integration_delivers_through_injected_fake_transport() -> None:
    transport = FakeLangfuseTransport()
    integration = LangfuseObservabilityIntegration.from_transport(transport, enabled=True)
    envelope = _sanitized_envelope_with_attributes()

    await integration.export(envelope)

    assert transport.send_count == 1
    assert len(transport.payloads) == 1
    assert transport.payloads[0].provider_id == "langfuse"


def test_create_langfuse_observability_backend_still_returns_query_facade() -> None:
    backend = create_langfuse_observability_backend(client=_FakeObsClient())

    assert_observability_backend(backend)
    assert backend.query_instant("up").series[0].points[0].value == 3.0


def test_register_langfuse_integration_still_uses_legacy_factory() -> None:
    assert MANIFEST.slug == "langfuse"
    assert register_langfuse_integration.__module__.endswith(".register")
    source = inspect.getsource(register_langfuse_integration)
    assert "create_langfuse_observability_backend" in source
    assert "create_langfuse_observability_integration" not in source


def test_create_langfuse_observability_integration_factory() -> None:
    transport = FakeLangfuseTransport()
    integration = create_langfuse_observability_integration(transport=transport, enabled=True)

    assert isinstance(integration, LangfuseObservabilityIntegration)
    assert integration.transport is transport
    assert PlatformIntegrationCapability.EXPORT in integration.capabilities


def test_no_vendor_sdk_imports_in_langfuse_integration_module() -> None:
    source = _LANGFUSE_INTEGRATION_PATH.read_text(encoding="utf-8").lower()

    for token in _FORBIDDEN_VENDOR_IMPORT_PREFIXES:
        assert f"import {token}" not in source
        assert f"from {token}" not in source


def test_no_real_network_in_tests() -> None:
    transport = FakeLangfuseTransport()
    integration = create_langfuse_observability_integration(transport=transport, enabled=True)
    envelope = _sanitized_envelope_with_attributes()

    asyncio.run(integration.export(envelope))

    assert transport.send_count == 1
