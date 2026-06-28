# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from intergrax.runtime.observability.export_attributes import (
    APPLICATION_OBSERVABILITY_ATTRIBUTES_SCHEMA,
    ApplicationObservabilityAttributes,
    observability_attribute_key,
    sanitize_application_observability_attributes,
)
from intergrax.runtime.observability.export_boundary import (
    FORBIDDEN_EXPORT_CONTENT_FIELDS,
    ExportRecordKind,
    ObservabilityExportEnvelope,
    envelope_is_content_safe,
)
from intergrax.runtime.observability.export_policy import (
    ObservabilityExportPolicy,
    apply_observability_export_policy,
    try_export_observability_envelope,
)
from intergrax.runtime.observability.jsonl_exporter import JsonlObservabilityExporter

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_EXPORT_ATTRIBUTES_PATH = (
    _PROJECT_ROOT / "intergrax" / "runtime" / "observability" / "export_attributes.py"
)

_FORBIDDEN_VENDOR_TOKENS = (
    "langfuse",
    "arize",
    "phoenix",
    "elasticsearch",
    "opentelemetry",
    "otlp",
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


def test_base_application_attributes_expose_schema_and_namespace() -> None:
    attrs = ApplicationObservabilityAttributes(namespace="billing", operation="billing.invoice")

    assert attrs.schema_version == APPLICATION_OBSERVABILITY_ATTRIBUTES_SCHEMA
    assert attrs.namespace == "billing"
    safe = attrs.to_safe_attributes()
    assert safe[observability_attribute_key("billing", "namespace")] == "billing"
    assert safe[observability_attribute_key("billing", "operation")] == "billing.invoice"


def test_custom_subclass_defines_application_specific_safe_fields() -> None:
    attrs = ExampleApplicationObservabilityAttributes(
        result_count=7,
        strategy="fast",
        tags=["alpha", "beta"],
    )

    safe = attrs.to_safe_attributes()
    assert safe[observability_attribute_key("example", "operation")] == "example.run"
    assert safe[observability_attribute_key("example", "result_count")] == 7
    assert safe[observability_attribute_key("example", "strategy")] == "fast"
    assert safe[observability_attribute_key("example", "tags")] == ["alpha", "beta"]


def test_enabled_policy_preserves_safe_scalar_application_attributes() -> None:
    attrs = ExampleApplicationObservabilityAttributes(result_count=3, strategy="balanced")
    envelope = ObservabilityExportEnvelope(
        record_kind=ExportRecordKind.RUNTIME_EVENT,
        run_id="run-1",
        application_attributes=attrs,
    )
    policy = ObservabilityExportPolicy(enabled=True)

    result = apply_observability_export_policy(envelope, policy)

    assert result.exported is True
    assert result.envelope is not None
    assert result.envelope.application_attributes is None
    sanitized = result.envelope.sanitized_application_attributes
    assert sanitized is not None
    assert sanitized.namespace == "example"
    assert sanitized.attributes[observability_attribute_key("example", "result_count")] == 3
    assert sanitized.attributes[observability_attribute_key("example", "strategy")] == "balanced"


def test_enabled_policy_preserves_safe_list_string_attributes() -> None:
    attrs = ExampleApplicationObservabilityAttributes(tags=["one", "two"])
    envelope = ObservabilityExportEnvelope(
        record_kind=ExportRecordKind.TOOL_CALL,
        run_id="run-1",
        application_attributes=attrs,
    )
    policy = ObservabilityExportPolicy(enabled=True)

    result = apply_observability_export_policy(envelope, policy)

    assert result.exported is True
    assert result.envelope is not None
    sanitized = result.envelope.sanitized_application_attributes
    assert sanitized is not None
    assert sanitized.attributes[observability_attribute_key("example", "tags")] == ["one", "two"]


def test_unsafe_nested_values_are_rejected_or_dropped() -> None:
    with pytest.raises(ValidationError):
        ExampleApplicationObservabilityAttributes.model_validate(
            {
                "namespace": "example",
                "operation": "example.run",
                "result_count": {"nested": "dict"},
            }
        )

    nested_attrs = ExampleApplicationObservabilityAttributes.model_construct(
        namespace="example",
        operation="example.run",
        result_count=1,
        strategy={"nested": "dict"},
    )
    safe = nested_attrs.to_safe_attributes()
    assert observability_attribute_key("example", "strategy") not in safe

    bytes_attrs = ExampleApplicationObservabilityAttributes.model_construct(
        namespace="example",
        operation="example.run",
        result_count=1,
        strategy=b"raw-bytes",
    )
    assert observability_attribute_key("example", "strategy") not in bytes_attrs.to_safe_attributes()

    result = sanitize_application_observability_attributes(nested_attrs)
    assert result.sanitized is not None
    assert observability_attribute_key("example", "result_count") in result.sanitized.attributes


def test_path_like_values_are_hashed_by_policy() -> None:
    unsafe_path = "C:\\Users\\secret\\project\\file.txt"
    attrs = ExampleApplicationObservabilityAttributes(strategy=unsafe_path)
    envelope = ObservabilityExportEnvelope(
        record_kind=ExportRecordKind.RUNTIME_EVENT,
        run_id="run-1",
        application_attributes=attrs,
    )
    policy = ObservabilityExportPolicy(enabled=True)

    result = apply_observability_export_policy(envelope, policy)

    assert result.exported is True
    assert result.envelope is not None
    sanitized = result.envelope.sanitized_application_attributes
    assert sanitized is not None
    key = observability_attribute_key("example", "strategy")
    exported_value = sanitized.attributes[key]
    assert isinstance(exported_value, str)
    assert exported_value != unsafe_path
    assert len(exported_value) == 64


def test_raw_sensitive_fields_are_not_exported_by_default() -> None:
    class SensitiveExampleAttributes(ApplicationObservabilityAttributes):
        namespace: str = "example"
        operation: str = "example.run"
        prompt: str = "secret prompt"
        result_count: int = 1

    attrs = SensitiveExampleAttributes()
    envelope = ObservabilityExportEnvelope(
        record_kind=ExportRecordKind.RUNTIME_EVENT,
        run_id="run-1",
        application_attributes=attrs,
    )
    policy = ObservabilityExportPolicy(enabled=True)

    result = apply_observability_export_policy(envelope, policy)

    assert result.exported is True
    assert result.envelope is not None
    sanitized = result.envelope.sanitized_application_attributes
    assert sanitized is not None
    assert observability_attribute_key("example", "prompt") not in sanitized.attributes
    assert sanitized.attributes[observability_attribute_key("example", "result_count")] == 1
    serialized = result.envelope.model_dump_json()
    for key in FORBIDDEN_EXPORT_CONTENT_FIELDS:
        assert f'"{key}"' not in serialized


@pytest.mark.asyncio
async def test_jsonl_exporter_serializes_sanitized_application_attributes(tmp_path: Path) -> None:
    attrs = ExampleApplicationObservabilityAttributes(result_count=5, strategy="safe")
    envelope = ObservabilityExportEnvelope(
        record_kind=ExportRecordKind.RUNTIME_EVENT,
        run_id="run-1",
        application_attributes=attrs,
    )
    policy = ObservabilityExportPolicy(enabled=True)
    output_path = tmp_path / "export.jsonl"
    exporter = JsonlObservabilityExporter(output_path, create_parent_dirs=True)

    result = await try_export_observability_envelope(
        envelope,
        exporter=exporter,
        policy=policy,
    )

    assert result.exported is True
    record = json.loads(output_path.read_text(encoding="utf-8").strip())
    sanitized = record["sanitized_application_attributes"]
    assert sanitized["schema_version"] == "sanitized_application_observability_attributes.v1"
    assert sanitized["namespace"] == "example"
    assert sanitized["attributes"][observability_attribute_key("example", "result_count")] == 5
    assert envelope_is_content_safe(result.envelope)  # type: ignore[arg-type]


def test_disabled_policy_does_not_export_application_attributes() -> None:
    attrs = ExampleApplicationObservabilityAttributes(result_count=2)
    envelope = ObservabilityExportEnvelope(
        record_kind=ExportRecordKind.RUNTIME_EVENT,
        run_id="run-1",
        application_attributes=attrs,
    )

    result = apply_observability_export_policy(envelope)

    assert result.exported is False
    assert result.envelope is None


def test_export_attributes_has_no_vendor_sdk_coupling() -> None:
    source = _EXPORT_ATTRIBUTES_PATH.read_text(encoding="utf-8")
    for token in _FORBIDDEN_VENDOR_TOKENS:
        assert token not in source, (
            f"export_attributes.py contains forbidden vendor coupling token: {token}"
        )
