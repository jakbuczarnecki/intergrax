# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json

import pytest
from pydantic import PrivateAttr

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

_FORBIDDEN_VENDOR_IMPORT_PREFIXES = (
    "langfuse",
    "arize",
    "phoenix",
    "opentelemetry",
    "elasticsearch",
)


class ExampleWorkspaceObservabilityAttributes(ApplicationObservabilityAttributes):
    namespace: str = "local_workspace"
    operation: str | None = "index_job"
    file_count: int | None = None


class ExampleObservabilityVendorIntegration(ObservabilityVendorIntegrationContract):
    _delivered: list[ObservabilityVendorPayload] = PrivateAttr(default_factory=list)

    async def deliver_payload(self, payload: ObservabilityVendorPayload) -> None:
        self._delivered.append(payload)


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


def test_observability_vendor_contract_derives_from_platform_integration_contract() -> None:
    contract = ObservabilityVendorIntegrationContract.for_provider(
        provider_id="example",
        supported_signals=(ObservabilityVendorSignal.EVENTS,),
    )

    assert isinstance(contract, PlatformIntegrationContract)
    assert isinstance(contract, ObservabilityVendorIntegrationContract)


def test_observability_vendor_contract_exposes_core_fields() -> None:
    contract = ObservabilityVendorIntegrationContract.for_provider(
        provider_id="langfuse",
        supported_signals=(ObservabilityVendorSignal.EVENTS, ObservabilityVendorSignal.LLM_EVENTS),
        display_name="Langfuse Observability",
        version="1.0.0",
    )

    assert contract.schema_id == OBSERVABILITY_VENDOR_INTEGRATION_CONTRACT_SCHEMA
    assert contract.integration_id == derive_platform_integration_id("langfuse", "observability_vendor")
    assert contract.provider_id == "langfuse"
    assert contract.integration_kind == PlatformIntegrationKind.OBSERVABILITY_VENDOR.value
    assert PlatformIntegrationCapability.EXPORT in contract.capabilities
    assert PlatformIntegrationCapability.HEALTH_CHECK in contract.capabilities


def test_observability_vendor_contract_exposes_supported_signals() -> None:
    contract = ObservabilityVendorIntegrationContract.for_provider(
        provider_id="arize",
        supported_signals=(
            ObservabilityVendorSignal.TRACES,
            ObservabilityVendorSignal.METRICS,
        ),
    )

    assert contract.supported_signals == (
        ObservabilityVendorSignal.TRACES,
        ObservabilityVendorSignal.METRICS,
    )
    public_view = contract.public_view()
    assert public_view["supported_signals"] == ["traces", "metrics"]


def test_example_vendor_subclass_maps_sanitized_envelope_to_payload() -> None:
    integration = ExampleObservabilityVendorIntegration.for_provider(
        provider_id="example",
        supported_signals=(ObservabilityVendorSignal.EVENTS,),
    )
    envelope = _sanitized_envelope_with_attributes()

    mapping = integration.map_envelope(envelope)
    payload = mapping.payload

    assert payload.provider_id == "example"
    assert payload.integration_id == "example:observability_vendor"
    assert payload.integration_kind == PlatformIntegrationKind.OBSERVABILITY_VENDOR.value
    assert payload.record_type == ExportRecordKind.RUNTIME_EVENT.value
    assert payload.run_id == "run-1"
    assert payload.task_id == "task-1"
    assert payload.agent_id == "agent-1"
    assert payload.capability == "search"
    assert payload.event_type == "tool.completed"
    assert payload.status == ExportStatus.SUCCEEDED.value
    assert payload.latency_ms == 42
    assert payload.counts == {"hit_count": 2}
    assert payload.tool_id == "grep"


def test_payload_includes_sanitized_application_attributes_with_namespaced_keys() -> None:
    integration = ExampleObservabilityVendorIntegration.for_provider(provider_id="example")
    envelope = _sanitized_envelope_with_attributes()

    payload = integration.map_envelope(envelope).payload

    assert payload.sanitized_application_attributes is not None
    attributes = payload.sanitized_application_attributes.attributes
    assert observability_attribute_key("local_workspace", "file_count") in attributes
    assert attributes[observability_attribute_key("local_workspace", "file_count")] == 3
    assert observability_attribute_key("local_workspace", "namespace") in attributes


def test_payload_does_not_include_raw_application_attributes() -> None:
    integration = ExampleObservabilityVendorIntegration.for_provider(provider_id="example")
    envelope = _sanitized_envelope_with_attributes()

    payload = integration.map_envelope(envelope).payload
    dumped = payload.model_dump()

    assert "application_attributes" not in dumped
    assert "sanitized_application_attributes" in dumped


def test_payload_does_not_include_forbidden_sensitive_content() -> None:
    integration = ExampleObservabilityVendorIntegration.for_provider(provider_id="example")
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

    assert "endpoint" not in serialized
    assert "headers" not in serialized
    assert "token" not in serialized


def test_same_provider_id_allows_other_future_category_integrations() -> None:
    observability = ObservabilityVendorIntegrationContract.for_provider(
        provider_id="elasticsearch",
        supported_signals=(ObservabilityVendorSignal.LOGS,),
    )
    vector_store = PlatformIntegrationContract.for_provider(
        provider_id="elasticsearch",
        integration_kind=PlatformIntegrationKind.VECTOR_STORE,
        capabilities=(PlatformIntegrationCapability.READ, PlatformIntegrationCapability.WRITE),
    )

    assert observability.provider_id == vector_store.provider_id == "elasticsearch"
    assert observability.integration_kind == PlatformIntegrationKind.OBSERVABILITY_VENDOR.value
    assert vector_store.integration_kind == PlatformIntegrationKind.VECTOR_STORE.value
    assert observability.integration_id != vector_store.integration_id


def test_map_envelope_rejects_raw_application_attributes() -> None:
    integration = ExampleObservabilityVendorIntegration.for_provider(provider_id="example")
    attributes = ExampleWorkspaceObservabilityAttributes(file_count=1)
    envelope = ObservabilityExportEnvelope(
        record_kind=ExportRecordKind.RUNTIME_EVENT,
        application_attributes=attributes,
    )

    with pytest.raises(ValueError, match="raw application_attributes"):
        integration.map_envelope(envelope)


def test_no_concrete_vendor_adapter_is_implemented() -> None:
    import asyncio
    from datetime import datetime, timezone

    contract = ObservabilityVendorIntegrationContract.for_provider(provider_id="custom")
    payload = ObservabilityVendorPayload(
        provider_id="custom",
        integration_id="custom:observability_vendor",
        integration_kind=PlatformIntegrationKind.OBSERVABILITY_VENDOR.value,
        record_type=ExportRecordKind.DIAGNOSTIC.value,
        recorded_at=datetime.now(timezone.utc),
    )

    with pytest.raises(NotImplementedError):
        asyncio.run(contract.deliver_payload(payload))


def test_no_vendor_sdk_imports_in_observability_integration_module() -> None:
    import intergrax.runtime.integrations.observability as observability_module

    module_path = observability_module.__file__
    assert module_path is not None
    source = open(module_path, encoding="utf-8").read().lower()

    for token in _FORBIDDEN_VENDOR_IMPORT_PREFIXES:
        assert f"import {token}" not in source
        assert f"from {token}" not in source
