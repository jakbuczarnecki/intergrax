# © Artur Czarnecki. All rights reserved.

"""Controlled LKW-shaped platform problem signal proof (LKW-PF-ERR-1).

Not endpoint runtime integration — proves developer-facing problem reporting
through the platform helper without manual signal/envelope/policy construction.
"""

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
    envelope_is_content_safe,
)
from intergrax.runtime.observability.export_policy import ObservabilityExportPolicy
from intergrax.runtime.observability.problem_reporter import (
    ProblemReportContext,
    report_problem,
)

pytestmark = pytest.mark.unit


class LkwFailureProblemAttributes(ApplicationObservabilityAttributes):
    namespace: str = "lkw"
    operation: str = "local.workspace.search"
    pipeline_stage: str = "search"
    collection_id: str = "lkw-evidence-smoke-ws"
    failure_mode: str = "retrieve_failed"
    source_count: int = 1


class UnsafeLkwFailureProblemAttributes(ApplicationObservabilityAttributes):
    namespace: str = "lkw"
    operation: str = "local.workspace.search"
    pipeline_stage: str = "search"
    query: str = "raw secret query"
    content: str = "raw secret content"
    chunks: list[str] = ["raw secret chunk"]
    source_path: str = "/secret/full/path/document.txt"
    source_count: int = 1


_LKW_CONTEXT = ProblemReportContext(
    run_id="run-lkw-failure-proof",
    task_id="task-lkw-failure-proof",
    tenant_id="tenant-lkw-proof",
    workspace_id="workspace-lkw-proof",
    agent_id="local_search",
    capability="local.workspace.search",
    correlation_id="corr-lkw-failure-proof",
)


@pytest.mark.asyncio
async def test_lkw_controlled_retrieve_failure_reports_platform_problem_signal() -> None:
    result = await report_problem(
        context=_LKW_CONTEXT,
        problem_kind="lkw.retrieve_failed",
        severity="error",
        error_code="LKW_RETRIEVE_FAILED",
        source_layer="lkw.pipeline",
        source_component="local_search",
        tool_id="rag.retrieve",
        application_attributes=LkwFailureProblemAttributes(),
        policy=ObservabilityExportPolicy(enabled=True),
    )

    assert result.exported is True
    envelope = result.envelope
    assert envelope is not None
    assert envelope.record_kind == ExportRecordKind.PROBLEM_SIGNAL
    assert envelope.problem_kind == "lkw.retrieve_failed"
    assert envelope.problem_severity == "error"
    assert envelope.problem_error_code == "LKW_RETRIEVE_FAILED"
    assert envelope.event_type == "lkw.retrieve_failed"
    assert envelope.run_id == _LKW_CONTEXT.run_id
    assert envelope.task_id == _LKW_CONTEXT.task_id
    assert envelope.agent_id == "local_search"
    assert envelope.tool_id == "rag.retrieve"
    assert envelope.capability == "local.workspace.search"
    assert envelope.tenant_id == "tenant-lkw-proof"
    assert envelope.workspace_id == "workspace-lkw-proof"

    sanitized = envelope.sanitized_application_attributes
    assert sanitized is not None
    assert sanitized.attributes[observability_attribute_key("lkw", "pipeline_stage")] == "search"
    assert sanitized.attributes[observability_attribute_key("lkw", "collection_id")] == (
        "lkw-evidence-smoke-ws"
    )
    assert sanitized.attributes[observability_attribute_key("lkw", "failure_mode")] == (
        "retrieve_failed"
    )
    assert sanitized.attributes[observability_attribute_key("lkw", "source_count")] == 1
    assert envelope.application_attributes is None


@pytest.mark.asyncio
async def test_lkw_controlled_failure_does_not_leak_raw_query_content_chunks_or_paths() -> None:
    result = await report_problem(
        context=_LKW_CONTEXT,
        problem_kind="lkw.retrieve_failed",
        error_code="LKW_RETRIEVE_FAILED",
        source_layer="lkw.pipeline",
        source_component="local_search",
        application_attributes=UnsafeLkwFailureProblemAttributes(),
        policy=ObservabilityExportPolicy(enabled=True),
    )

    assert result.exported is True
    assert result.envelope is not None
    sanitized = result.envelope.sanitized_application_attributes
    assert sanitized is not None

    for forbidden in ("query", "content", "chunks", "source_path", "absolute_path", "file_path", "tool_args", "prompt"):
        assert observability_attribute_key("lkw", forbidden) not in sanitized.attributes

    serialized = result.envelope.model_dump_json()
    for key in FORBIDDEN_EXPORT_CONTENT_FIELDS:
        assert f'"{key}"' not in serialized
    for raw in (
        "raw secret query",
        "raw secret content",
        "raw secret chunk",
        "/secret/full/path/document.txt",
    ):
        assert raw not in serialized

    assert sanitized.attributes[observability_attribute_key("lkw", "pipeline_stage")] == "search"
    assert sanitized.attributes[observability_attribute_key("lkw", "source_count")] == 1


@pytest.mark.asyncio
async def test_lkw_controlled_failure_can_attach_reference_only_artifact() -> None:
    artifact = ObservabilityArtifactReference(
        artifact_ref="artifact-lkw-failure-proof",
        sha256="d" * 64,
        safe_relative_path="reports/lkw-failure-proof.json",
        schema_id="lkw_failure_proof.v1",
    )

    result = await report_problem(
        context=_LKW_CONTEXT,
        problem_kind="lkw.retrieve_failed",
        error_code="LKW_RETRIEVE_FAILED",
        source_layer="lkw.pipeline",
        source_component="local_search",
        artifact_refs=(artifact,),
        policy=ObservabilityExportPolicy(enabled=True),
    )

    assert result.exported is True
    envelope = result.envelope
    assert envelope is not None
    assert envelope.artifact_ref == "artifact-lkw-failure-proof"
    assert envelope.sha256 == "d" * 64
    assert envelope.safe_relative_path == "reports/lkw-failure-proof.json"
    assert envelope.schema_id == "lkw_failure_proof.v1"
    assert envelope.counts["artifact_ref_count"] == 1

    serialized = envelope.model_dump_json()
    assert '"artifact_refs"' not in serialized
    assert envelope_is_content_safe(envelope)
