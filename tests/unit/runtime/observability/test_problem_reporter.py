# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.runtime.observability.export_attributes import (
    ApplicationObservabilityAttributes,
    observability_attribute_key,
)
from intergrax.runtime.observability.export_boundary import (
    FORBIDDEN_EXPORT_CONTENT_FIELDS,
    ExportRecordKind,
    ExportStatus,
    ObservabilityExportEnvelope,
    ObservabilityExporter,
    envelope_is_content_safe,
)
from intergrax.runtime.observability.export_policy import ObservabilityExportPolicy
from intergrax.runtime.observability.problem_reporter import (
    ProblemReportContext,
    ProblemReporter,
    build_problem_export_envelope,
    build_problem_signal,
    report_problem,
)

pytestmark = pytest.mark.unit


class SensitiveProblemAttributes(ApplicationObservabilityAttributes):
    namespace: str = "test"
    operation: str = "problem"
    prompt: str = "secret prompt"
    query: str = "secret query"
    content: str = "secret content"
    chunks: list[str] = ["secret chunk"]
    tool_args: str = "secret args"
    source_count: int = 1


class _RaisingObservabilityExporter:
    async def export(self, envelope: ObservabilityExportEnvelope) -> None:
        _ = envelope
        raise RuntimeError("export backend unavailable")


@pytest.fixture
def problem_context() -> ProblemReportContext:
    return ProblemReportContext(
        run_id="run-1",
        task_id="task-1",
        tenant_id="tenant-1",
        workspace_id="workspace-1",
        agent_id="agent-1",
        capability="local.workspace.search",
        correlation_id="corr-1",
    )


def test_build_problem_signal_uses_context_defaults(
    problem_context: ProblemReportContext,
) -> None:
    signal = build_problem_signal(
        context=problem_context,
        problem_kind="lkw.retrieve_failed",
        severity="error",
        error_code="LKW_RETRIEVE_FAILED",
        source_layer="lkw.pipeline",
        source_component="local_search",
        tool_id="rag.retrieve",
    )

    assert signal.problem_kind == "lkw.retrieve_failed"
    assert signal.severity == "error"
    assert signal.error_code == "LKW_RETRIEVE_FAILED"
    assert signal.run_id == problem_context.run_id
    assert signal.task_id == problem_context.task_id
    assert signal.agent_id == problem_context.agent_id
    assert signal.capability == problem_context.capability
    assert signal.correlation_id == problem_context.correlation_id
    assert signal.problem_id.startswith("problem-")
    assert len(signal.problem_id) > len("problem-")

    serialized = signal.model_dump_json()
    assert '"tenant_id"' not in serialized
    assert '"workspace_id"' not in serialized


def test_build_problem_export_envelope_applies_context_and_problem_mapping(
    problem_context: ProblemReportContext,
) -> None:
    envelope = build_problem_export_envelope(
        context=problem_context,
        problem_kind="lkw.retrieve_failed",
        severity="error",
        error_code="LKW_RETRIEVE_FAILED",
        problem_id="problem-1",
        tool_id="rag.retrieve",
    )

    assert envelope.record_kind == ExportRecordKind.PROBLEM_SIGNAL
    assert envelope.problem_kind == "lkw.retrieve_failed"
    assert envelope.problem_severity == "error"
    assert envelope.problem_error_code == "LKW_RETRIEVE_FAILED"
    assert envelope.tenant_id == problem_context.tenant_id
    assert envelope.workspace_id == problem_context.workspace_id
    assert envelope.event_type == "lkw.retrieve_failed"
    assert envelope.status == ExportStatus.FAILED


@pytest.mark.asyncio
async def test_report_problem_applies_policy_and_returns_export_policy_result(
    problem_context: ProblemReportContext,
) -> None:
    result = await report_problem(
        context=problem_context,
        problem_kind="lkw.retrieve_failed",
        error_code="LKW_RETRIEVE_FAILED",
        policy=ObservabilityExportPolicy(enabled=True),
    )

    assert result.exported is True
    assert result.envelope is not None
    assert result.envelope.record_kind == ExportRecordKind.PROBLEM_SIGNAL
    assert result.envelope.application_attributes is None
    assert envelope_is_content_safe(result.envelope)


@pytest.mark.asyncio
async def test_problem_reporter_bound_facade_reports_with_short_call(
    problem_context: ProblemReportContext,
) -> None:
    reporter = ProblemReporter(
        context=problem_context,
        policy=ObservabilityExportPolicy(enabled=True),
    )

    result = await reporter.report(
        problem_kind="lkw.retrieve_failed",
        error_code="LKW_RETRIEVE_FAILED",
        source_component="local_search",
    )

    assert result.exported is True
    assert result.envelope is not None
    assert result.envelope.run_id == problem_context.run_id
    assert result.envelope.task_id == problem_context.task_id
    assert result.envelope.agent_id == problem_context.agent_id
    assert result.envelope.capability == problem_context.capability
    assert result.envelope.correlation_id == problem_context.correlation_id


@pytest.mark.asyncio
async def test_report_problem_sanitizes_forbidden_application_attributes(
    problem_context: ProblemReportContext,
) -> None:
    result = await report_problem(
        context=problem_context,
        problem_kind="lkw.retrieve_failed",
        application_attributes=SensitiveProblemAttributes(),
        policy=ObservabilityExportPolicy(enabled=True),
    )

    assert result.exported is True
    assert result.envelope is not None
    sanitized = result.envelope.sanitized_application_attributes
    assert sanitized is not None
    for forbidden in ("prompt", "query", "content", "chunks", "tool_args"):
        assert observability_attribute_key("test", forbidden) not in sanitized.attributes
    assert sanitized.attributes[observability_attribute_key("test", "source_count")] == 1

    serialized = result.envelope.model_dump_json()
    for key in FORBIDDEN_EXPORT_CONTENT_FIELDS:
        assert f'"{key}"' not in serialized
    for raw in ("secret prompt", "secret query", "secret content", "secret chunk", "secret args"):
        assert raw not in serialized


@pytest.mark.asyncio
async def test_report_problem_isolates_exporter_failures(
    problem_context: ProblemReportContext,
) -> None:
    result = await report_problem(
        context=problem_context,
        problem_kind="lkw.retrieve_failed",
        policy=ObservabilityExportPolicy(enabled=True),
        exporter=_RaisingObservabilityExporter(),  # type: ignore[arg-type]
    )

    assert result.exported is False
    assert result.reason == "exporter_failed"
    assert result.envelope is not None
    assert envelope_is_content_safe(result.envelope)
