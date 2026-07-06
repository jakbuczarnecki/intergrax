# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intergrax.runtime.observability.export_attributes import (
    OBSERVABILITY_ARTIFACT_REFERENCE_SCHEMA,
    ApplicationObservabilityAttributes,
    ObservabilityArtifactReference,
    observability_attribute_key,
    sanitize_application_observability_attributes,
)
from intergrax.runtime.observability.problem_signal import (
    PLATFORM_PROBLEM_SIGNAL_SCHEMA,
    PROBLEM_KIND_PLATFORM_EXCEPTION,
    PROBLEM_SEVERITY_ERROR,
    PROBLEM_STATUS_DETECTED,
    PlatformProblemSignal,
    _problem_signal_json_is_content_safe,
    problem_signal_is_content_safe,
)

pytestmark = pytest.mark.unit


class LkwProblemAttributes(ApplicationObservabilityAttributes):
    namespace: str = "lkw"
    operation: str = "local.workspace.pipeline"
    pipeline_stage: str = "search"
    source_count: int = 2


class AgentProblemAttributes(ApplicationObservabilityAttributes):
    namespace: str = "agent"
    operation: str = "agent.run"
    retry_count: int = 1
    fallback_used: bool = False


class SensitiveProblemAttributes(ApplicationObservabilityAttributes):
    namespace: str = "lkw"
    operation: str = "problem"
    prompt: str = "secret prompt"
    query: str = "secret query"
    content: str = "secret content"
    source_count: int = 1


def test_problem_signal_accepts_minimal_core_fields() -> None:
    signal = PlatformProblemSignal(problem_kind=PROBLEM_KIND_PLATFORM_EXCEPTION)

    assert signal.schema_version == PLATFORM_PROBLEM_SIGNAL_SCHEMA
    assert signal.problem_kind == "platform.exception"
    assert signal.severity == PROBLEM_SEVERITY_ERROR
    assert signal.status == PROBLEM_STATUS_DETECTED
    assert signal.occurred_at is not None

    explicit = PlatformProblemSignal(
        problem_kind="platform.tool_failure",
        severity="warning",
        status="detected",
    )
    assert explicit.severity == "warning"
    assert explicit.status == "detected"

    with pytest.raises(ValidationError):
        signal.problem_kind = "other"


def test_problem_signal_accepts_plugin_defined_taxonomy_values() -> None:
    signal = PlatformProblemSignal(
        problem_kind="lkw.retrieve_failed",
        source_layer="lkw.pipeline",
        source_component="local_search",
        severity="degraded",
        status="retrying",
    )

    assert signal.problem_kind == "lkw.retrieve_failed"
    assert signal.source_layer == "lkw.pipeline"
    assert signal.source_component == "local_search"
    assert signal.severity == "degraded"
    assert signal.status == "retrying"


@pytest.mark.parametrize(
    "forbidden_field",
    ["prompt", "query", "content", "chunks", "tool_args", "secret", "absolute_path"],
)
def test_problem_signal_forbids_extra_top_level_raw_fields(forbidden_field: str) -> None:
    with pytest.raises(ValidationError):
        PlatformProblemSignal.model_validate(
            {
                "schema_version": PLATFORM_PROBLEM_SIGNAL_SCHEMA,
                "problem_kind": "platform.exception",
                forbidden_field: "raw secret",
            }
        )


def test_problem_signal_accepts_typed_application_attributes() -> None:
    attrs = LkwProblemAttributes()
    signal = PlatformProblemSignal(
        problem_kind="lkw.retrieve_failed",
        application_attributes=attrs,
    )

    assert signal.application_attributes is attrs
    assert signal.application_attributes.pipeline_stage == "search"
    assert signal.application_attributes.source_count == 2


def test_problem_signal_accepts_typed_agent_attributes() -> None:
    attrs = AgentProblemAttributes(retry_count=2, fallback_used=True)
    signal = PlatformProblemSignal(
        problem_kind="agent.run_failed",
        agent_attributes=attrs,
    )

    assert signal.agent_attributes is attrs
    assert signal.agent_attributes.retry_count == 2
    assert signal.agent_attributes.fallback_used is True


def test_problem_signal_accepts_reference_only_artifact_refs() -> None:
    ref = ObservabilityArtifactReference(
        artifact_ref="artifact-123",
        sha256="a" * 64,
        safe_relative_path="reports/summary.json",
        schema_id="workspace_report.v1",
    )
    signal = PlatformProblemSignal(
        problem_kind="platform.artifact_failure",
        artifact_refs=(ref,),
    )

    assert len(signal.artifact_refs) == 1
    stored = signal.artifact_refs[0]
    assert stored.artifact_ref == "artifact-123"
    assert stored.sha256 == "a" * 64
    assert stored.safe_relative_path == "reports/summary.json"
    assert stored.schema_id == "workspace_report.v1"
    assert stored.schema_version == OBSERVABILITY_ARTIFACT_REFERENCE_SCHEMA


@pytest.mark.parametrize(
    "unsafe_path",
    ["/etc/passwd", "C:\\Users\\secret\\file.txt", "reports/../secrets.txt"],
)
def test_problem_signal_rejects_unsafe_artifact_ref_paths(unsafe_path: str) -> None:
    with pytest.raises(ValidationError):
        PlatformProblemSignal(
            problem_kind="platform.artifact_failure",
            artifact_refs=(ObservabilityArtifactReference(artifact_ref=unsafe_path),),
        )

    with pytest.raises(ValidationError):
        PlatformProblemSignal(
            problem_kind="platform.artifact_failure",
            artifact_refs=(
                ObservabilityArtifactReference(
                    safe_relative_path=unsafe_path,
                    schema_id="x.v1",
                ),
            ),
        )


def test_problem_signal_is_content_safe_for_allowed_fields() -> None:
    signal = PlatformProblemSignal(
        problem_kind="platform.exception",
        safe_message="A safe summary that mentions message without using a forbidden key",
        error_code="E001",
    )

    assert problem_signal_is_content_safe(signal) is True


def test_problem_signal_is_not_content_safe_if_forbidden_key_is_present_in_serialized_shape() -> None:
    signal = PlatformProblemSignal(
        problem_kind="platform.exception",
        safe_message="summary with message word in value",
    )
    assert problem_signal_is_content_safe(signal) is True

    safe_json = signal.model_dump_json()
    polluted = safe_json.replace("}", ', "message": "secret"}')
    assert _problem_signal_json_is_content_safe(polluted) is False
    assert _problem_signal_json_is_content_safe('{"safe_message":"ok","problem_kind":"x"}') is True


def test_custom_attributes_with_forbidden_field_names_are_sanitized_by_existing_sanitizer() -> None:
    result = sanitize_application_observability_attributes(SensitiveProblemAttributes())

    assert result.sanitized is not None
    for forbidden in ("prompt", "query", "content"):
        assert observability_attribute_key("lkw", forbidden) not in result.sanitized.attributes
    assert result.sanitized.attributes[observability_attribute_key("lkw", "source_count")] == 1
    assert observability_attribute_key("lkw", "operation") in result.sanitized.attributes
