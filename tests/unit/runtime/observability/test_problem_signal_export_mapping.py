# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.runtime.observability.export_attributes import (
    ApplicationObservabilityAttributes,
    ObservabilityArtifactReference,
    observability_attribute_key,
)
from intergrax.runtime.observability.export_boundary import (
    FORBIDDEN_EXPORT_CONTENT_FIELDS,
    ExportRecordKind,
    ExportStatus,
    envelope_is_content_safe,
)
from intergrax.runtime.observability.export_policy import (
    ObservabilityExportPolicy,
    apply_observability_export_policy,
)
from intergrax.runtime.observability.problem_export import (
    envelope_from_problem_signal,
    problem_signal_export_status,
)
from intergrax.runtime.observability.problem_signal import PlatformProblemSignal

pytestmark = pytest.mark.unit


class LkwProblemAttributes(ApplicationObservabilityAttributes):
    namespace: str = "lkw"
    operation: str = "local.workspace.pipeline"
    pipeline_stage: str = "search"
    source_count: int = 2


class ForbiddenFieldProblemAttributes(ApplicationObservabilityAttributes):
    namespace: str = "lkw"
    operation: str = "local.workspace.pipeline"
    prompt: str = "secret prompt"
    query: str = "secret query"
    content: str = "secret content"
    chunks: list[str] = ["chunk-1"]
    tool_args: str = "secret args"
    source_count: int = 1
    safe_field: str = "ok"


class AgentProblemAttributes(ApplicationObservabilityAttributes):
    namespace: str = "agent"
    operation: str = "agent.run"
    retry_count: int = 2


def test_envelope_from_problem_signal_maps_core_fields() -> None:
    signal = PlatformProblemSignal(
        problem_id="problem-1",
        problem_kind="lkw.retrieve_failed",
        severity="error",
        error_code="LKW_RETRIEVE_FAILED",
        run_id="run-1",
        task_id="task-1",
        agent_id="local_search",
        tool_id="local_search",
        capability="local.workspace.pipeline",
        correlation_id="corr-1",
    )

    envelope = envelope_from_problem_signal(signal)

    assert envelope.record_kind == ExportRecordKind.PROBLEM_SIGNAL
    assert envelope.event_type == "lkw.retrieve_failed"
    assert envelope.status == ExportStatus.FAILED
    assert envelope.source_schema_id == "platform_problem_signal.v1"
    assert envelope.event_id == "problem-1"
    assert envelope.problem_kind == "lkw.retrieve_failed"
    assert envelope.problem_severity == "error"
    assert envelope.problem_error_code == "LKW_RETRIEVE_FAILED"


@pytest.mark.parametrize(
    ("severity", "expected"),
    [
        ("error", ExportStatus.FAILED),
        ("critical", ExportStatus.FAILED),
        ("warning", ExportStatus.UNKNOWN),
        ("degraded", ExportStatus.UNKNOWN),
        ("custom-risk", ExportStatus.UNKNOWN),
    ],
)
def test_problem_signal_status_mapping_is_minimal_and_plugin_safe(
    severity: str,
    expected: ExportStatus,
) -> None:
    signal = PlatformProblemSignal(problem_kind="platform.exception", severity=severity)
    assert problem_signal_export_status(signal) == expected


def test_envelope_from_problem_signal_preserves_plugin_taxonomy_strings() -> None:
    signal = PlatformProblemSignal(
        problem_kind="custom_app.invoice_validation_failed",
        source_layer="custom_app.billing",
        severity="degraded",
        status="retrying",
    )

    envelope = envelope_from_problem_signal(signal)

    assert envelope.event_type == "custom_app.invoice_validation_failed"
    assert envelope.problem_kind == "custom_app.invoice_validation_failed"
    assert envelope.status == ExportStatus.UNKNOWN


def test_envelope_from_problem_signal_maps_primary_artifact_ref_only() -> None:
    first = ObservabilityArtifactReference(
        artifact_ref="artifact-first",
        sha256="a" * 64,
        safe_relative_path="reports/first.json",
        schema_id="workspace_report.v1",
    )
    second = ObservabilityArtifactReference(
        artifact_ref="artifact-second",
        sha256="b" * 64,
        safe_relative_path="reports/second.json",
        schema_id="workspace_report.v2",
    )
    signal = PlatformProblemSignal(
        problem_kind="platform.artifact_failure",
        artifact_refs=(first, second),
    )

    envelope = envelope_from_problem_signal(signal)

    assert envelope.artifact_ref == "artifact-first"
    assert envelope.sha256 == "a" * 64
    assert envelope.safe_relative_path == "reports/first.json"
    assert envelope.schema_id == "workspace_report.v1"
    assert envelope.counts["artifact_ref_count"] == 2

    serialized = envelope.model_dump_json()
    assert "artifact-second" not in serialized
    assert "reports/second.json" not in serialized
    assert '"artifact_refs"' not in serialized


def test_envelope_from_problem_signal_applies_existing_policy_to_application_attributes() -> None:
    signal = PlatformProblemSignal(
        problem_kind="lkw.retrieve_failed",
        application_attributes=LkwProblemAttributes(),
    )

    envelope = envelope_from_problem_signal(signal)
    result = apply_observability_export_policy(
        envelope,
        ObservabilityExportPolicy(enabled=True),
    )

    assert result.exported is True
    assert result.envelope is not None
    assert result.envelope.application_attributes is None
    sanitized = result.envelope.sanitized_application_attributes
    assert sanitized is not None
    assert sanitized.attributes[observability_attribute_key("lkw", "pipeline_stage")] == "search"
    assert sanitized.attributes[observability_attribute_key("lkw", "source_count")] == 2


def test_forbidden_custom_application_fields_are_dropped_after_policy() -> None:
    signal = PlatformProblemSignal(
        problem_kind="lkw.retrieve_failed",
        application_attributes=ForbiddenFieldProblemAttributes(),
    )

    envelope = envelope_from_problem_signal(signal)
    result = apply_observability_export_policy(
        envelope,
        ObservabilityExportPolicy(enabled=True),
    )

    assert result.exported is True
    assert result.envelope is not None
    sanitized = result.envelope.sanitized_application_attributes
    assert sanitized is not None
    for forbidden in ("prompt", "query", "content", "chunks", "tool_args"):
        assert observability_attribute_key("lkw", forbidden) not in sanitized.attributes
    assert sanitized.attributes[observability_attribute_key("lkw", "safe_field")] == "ok"

    serialized = result.envelope.model_dump_json()
    for key in FORBIDDEN_EXPORT_CONTENT_FIELDS:
        assert f'"{key}"' not in serialized


def test_agent_attributes_are_not_silently_exported_in_obs_problem_2() -> None:
    """Agent attribute export mapping is deferred in OBS-PROBLEM-2."""
    signal = PlatformProblemSignal(
        problem_kind="agent.run_failed",
        agent_attributes=AgentProblemAttributes(),
    )

    envelope = envelope_from_problem_signal(signal)
    result = apply_observability_export_policy(
        envelope,
        ObservabilityExportPolicy(enabled=True),
    )

    assert result.exported is True
    assert result.envelope is not None
    serialized = result.envelope.model_dump_json()
    assert '"agent_attributes"' not in serialized
    sanitized = result.envelope.sanitized_application_attributes
    if sanitized is not None:
        assert observability_attribute_key("agent", "retry_count") not in sanitized.attributes


def test_safe_message_is_not_exported_in_obs_problem_2() -> None:
    signal = PlatformProblemSignal(
        problem_kind="platform.exception",
        safe_message="safe summary",
    )

    envelope = envelope_from_problem_signal(signal)
    serialized = envelope.model_dump_json()

    assert '"safe_message"' not in serialized
    assert "safe summary" not in serialized


def test_problem_signal_export_envelope_is_content_safe() -> None:
    signal = PlatformProblemSignal(
        problem_kind="lkw.retrieve_failed",
        severity="error",
        error_code="LKW_RETRIEVE_FAILED",
        application_attributes=LkwProblemAttributes(),
        artifact_refs=(
            ObservabilityArtifactReference(
                artifact_ref="artifact-1",
                sha256="c" * 64,
                safe_relative_path="reports/summary.json",
                schema_id="workspace_report.v1",
            ),
        ),
    )

    envelope = envelope_from_problem_signal(signal)
    result = apply_observability_export_policy(
        envelope,
        ObservabilityExportPolicy(enabled=True),
    )

    assert result.exported is True
    assert result.envelope is not None
    assert envelope_is_content_safe(result.envelope)

    serialized = result.envelope.model_dump_json()
    for key in ("prompt", "query", "content", "chunks", "tool_args", "secret", "absolute_path"):
        assert f'"{key}"' not in serialized
