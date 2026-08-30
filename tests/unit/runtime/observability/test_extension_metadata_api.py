# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from testing_support.runtime_events import runtime_event_test_identity
from intergrax.runtime.observability.export_attributes import (
    OBSERVABILITY_ARTIFACT_REFERENCE_SCHEMA,
    ApplicationObservabilityAttributes,
    ObservabilityArtifactReference,
    observability_attribute_key,
)
from intergrax.runtime.observability.export_boundary import (
    FORBIDDEN_EXPORT_CONTENT_FIELDS,
    ExportRecordKind,
    ObservabilityExportEnvelope,
    envelope_from_runtime_event,
    envelope_is_content_safe,
    envelope_with_observability_extensions,
)
from intergrax.runtime.observability.export_policy import (
    ObservabilityExportPolicy,
    apply_observability_export_policy,
)

pytestmark = pytest.mark.unit


class ExampleApplicationObservabilityAttributes(ApplicationObservabilityAttributes):
    namespace: str = "example"
    operation: str = "example.run"
    result_count: int = 0


def test_artifact_reference_accepts_reference_fields() -> None:
    ref = ObservabilityArtifactReference(
        artifact_ref="artifact-123",
        sha256="a" * 64,
        safe_relative_path="reports/summary.json",
        schema_id="workspace_report.v1",
    )

    assert ref.schema_version == OBSERVABILITY_ARTIFACT_REFERENCE_SCHEMA
    assert ref.artifact_ref == "artifact-123"
    assert ref.sha256 == "a" * 64
    assert ref.safe_relative_path == "reports/summary.json"
    assert ref.schema_id == "workspace_report.v1"


@pytest.mark.parametrize(
    "artifact_ref",
    ["/etc/passwd", "C:\\Users\\secret\\file.txt", "reports/../secrets.txt"],
)
def test_artifact_reference_rejects_unsafe_paths(artifact_ref: str) -> None:
    with pytest.raises(ValidationError):
        ObservabilityArtifactReference(artifact_ref=artifact_ref)

    with pytest.raises(ValidationError):
        ObservabilityArtifactReference(safe_relative_path=artifact_ref, schema_id="x.v1")


def test_artifact_reference_rejects_empty_reference() -> None:
    with pytest.raises(ValidationError):
        ObservabilityArtifactReference()


@pytest.mark.parametrize("forbidden_field", ["content", "prompt", "query", "chunks"])
def test_artifact_reference_forbids_extra_content_fields(forbidden_field: str) -> None:
    with pytest.raises(ValidationError):
        ObservabilityArtifactReference.model_validate(
            {
                "schema_version": OBSERVABILITY_ARTIFACT_REFERENCE_SCHEMA,
                "artifact_ref": "artifact-1",
                forbidden_field: "raw secret",
            }
        )


def test_envelope_helper_attaches_application_attributes() -> None:
    base = ObservabilityExportEnvelope(record_kind=ExportRecordKind.RUNTIME_EVENT, run_id="run-1")
    attrs = ExampleApplicationObservabilityAttributes(result_count=4)

    extended = envelope_with_observability_extensions(base, application_attributes=attrs)

    assert extended.application_attributes is attrs
    assert base.application_attributes is None


def test_envelope_helper_attaches_artifact_reference_fields() -> None:
    base = ObservabilityExportEnvelope(record_kind=ExportRecordKind.TOOL_CALL, run_id="run-1")
    ref = ObservabilityArtifactReference(
        artifact_ref="artifact-42",
        sha256="digest",
        safe_relative_path="out/result.json",
        schema_id="tool_output.v1",
    )

    extended = envelope_with_observability_extensions(base, artifact_ref=ref)

    assert extended.artifact_ref == "artifact-42"
    assert extended.sha256 == "digest"
    assert extended.safe_relative_path == "out/result.json"
    assert extended.schema_id == "tool_output.v1"
    assert base.artifact_ref == ""
    assert base.sha256 == ""


def test_envelope_helper_returns_new_envelope_without_mutating_original() -> None:
    base = ObservabilityExportEnvelope(
        record_kind=ExportRecordKind.RUNTIME_EVENT,
        run_id="run-1",
        event_id="event-1",
    )
    attrs = ExampleApplicationObservabilityAttributes(result_count=1)
    ref = ObservabilityArtifactReference(artifact_ref="artifact-9", schema_id="x.v1")

    extended = envelope_with_observability_extensions(
        base,
        application_attributes=attrs,
        artifact_ref=ref,
    )

    assert extended is not base
    assert extended.run_id == "run-1"
    assert extended.event_id == "event-1"
    assert base.application_attributes is None
    assert base.artifact_ref == ""


def test_policy_clears_raw_application_attributes_and_emits_sanitized() -> None:
    attrs = ExampleApplicationObservabilityAttributes(result_count=2)
    envelope = envelope_with_observability_extensions(
        ObservabilityExportEnvelope(record_kind=ExportRecordKind.RUNTIME_EVENT, run_id="run-1"),
        application_attributes=attrs,
    )
    policy = ObservabilityExportPolicy(enabled=True)

    result = apply_observability_export_policy(envelope, policy)

    assert result.exported is True
    assert result.envelope is not None
    assert result.envelope.application_attributes is None
    sanitized = result.envelope.sanitized_application_attributes
    assert sanitized is not None
    assert sanitized.attributes[observability_attribute_key("example", "result_count")] == 2
    assert envelope_is_content_safe(result.envelope)


def test_policy_drops_forbidden_application_fields() -> None:
    class SensitiveAttributes(ApplicationObservabilityAttributes):
        namespace: str = "example"
        operation: str = "example.run"
        prompt: str = "secret prompt"
        query: str = "secret query"
        content: str = "secret content"
        result_count: int = 1

    envelope = envelope_with_observability_extensions(
        ObservabilityExportEnvelope(record_kind=ExportRecordKind.RUNTIME_EVENT, run_id="run-1"),
        application_attributes=SensitiveAttributes(),
    )
    policy = ObservabilityExportPolicy(enabled=True)

    result = apply_observability_export_policy(envelope, policy)

    assert result.exported is True
    assert result.envelope is not None
    sanitized = result.envelope.sanitized_application_attributes
    assert sanitized is not None
    for forbidden in ("prompt", "query", "content"):
        assert observability_attribute_key("example", forbidden) not in sanitized.attributes
    serialized = result.envelope.model_dump_json()
    for key in FORBIDDEN_EXPORT_CONTENT_FIELDS:
        assert f'"{key}"' not in serialized


def test_runtime_event_payload_arbitrary_fields_are_not_auto_exported() -> None:
    event = RuntimeEvent(
        event_type=RuntimeEventType.TOOL_COMPLETED,
        phase=ExecutionPhase.STEP_EXECUTION,
        payload={
            "tool_id": "workspace.read_file",
            "latency_ms": 9,
            "prompt": "secret prompt",
            "query": "secret query",
            "content": "secret content",
            "chunks": ["chunk-1"],
        },
        **runtime_event_test_identity(),
    )

    envelope = envelope_from_runtime_event(event)

    assert envelope.tool_id == "workspace.read_file"
    assert envelope.latency_ms == 9
    serialized = envelope.model_dump_json()
    for key in ("prompt", "query", "content", "chunks"):
        assert f'"{key}"' not in serialized
    assert envelope_is_content_safe(envelope)
