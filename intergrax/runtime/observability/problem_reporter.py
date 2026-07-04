# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Developer-facing platform problem reporting helper (LKW-PF-ERR-1)."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from intergrax.runtime.observability.export_attributes import (
    ApplicationObservabilityAttributes,
    ObservabilityArtifactReference,
)
from intergrax.runtime.observability.export_boundary import (
    ObservabilityExportEnvelope,
    ObservabilityExporter,
)
from intergrax.runtime.observability.export_policy import (
    ExportPolicyResult,
    ObservabilityExportPolicy,
    try_export_observability_envelope,
)
from intergrax.runtime.observability.problem_export import envelope_from_problem_signal
from intergrax.runtime.observability.problem_signal import (
    PROBLEM_SEVERITY_ERROR,
    PROBLEM_STATUS_DETECTED,
    PlatformProblemSignal,
)


@dataclass(frozen=True, slots=True)
class ProblemReportContext:
    run_id: str = ""
    task_id: str = ""
    tenant_id: str = ""
    workspace_id: str = ""
    agent_id: str = ""
    capability: str = ""
    correlation_id: str = ""


def _default_problem_id() -> str:
    return f"problem-{uuid4().hex}"


def build_problem_signal(
    *,
    context: ProblemReportContext,
    problem_kind: str,
    severity: str = PROBLEM_SEVERITY_ERROR,
    error_code: str = "",
    source_layer: str = "",
    source_component: str = "",
    status: str = PROBLEM_STATUS_DETECTED,
    problem_id: str | None = None,
    event_id: str = "",
    tool_id: str = "",
    application_attributes: ApplicationObservabilityAttributes | None = None,
    artifact_refs: tuple[ObservabilityArtifactReference, ...] = (),
) -> PlatformProblemSignal:
    """Build a platform problem signal from developer context and taxonomy fields."""
    return PlatformProblemSignal(
        problem_id=problem_id or _default_problem_id(),
        problem_kind=problem_kind,
        severity=severity,
        source_layer=source_layer,
        source_component=source_component,
        status=status,
        error_code=error_code,
        run_id=context.run_id,
        task_id=context.task_id,
        event_id=event_id,
        agent_id=context.agent_id,
        tool_id=tool_id,
        capability=context.capability,
        correlation_id=context.correlation_id,
        application_attributes=application_attributes,
        artifact_refs=artifact_refs,
    )


def build_problem_export_envelope(
    *,
    context: ProblemReportContext,
    problem_kind: str,
    severity: str = PROBLEM_SEVERITY_ERROR,
    error_code: str = "",
    source_layer: str = "",
    source_component: str = "",
    status: str = PROBLEM_STATUS_DETECTED,
    problem_id: str | None = None,
    event_id: str = "",
    tool_id: str = "",
    application_attributes: ApplicationObservabilityAttributes | None = None,
    artifact_refs: tuple[ObservabilityArtifactReference, ...] = (),
) -> ObservabilityExportEnvelope:
    """Map a problem signal to an export envelope with context-only tenant/workspace fields."""
    signal = build_problem_signal(
        context=context,
        problem_kind=problem_kind,
        severity=severity,
        error_code=error_code,
        source_layer=source_layer,
        source_component=source_component,
        status=status,
        problem_id=problem_id,
        event_id=event_id,
        tool_id=tool_id,
        application_attributes=application_attributes,
        artifact_refs=artifact_refs,
    )
    envelope = envelope_from_problem_signal(signal)
    return envelope.model_copy(
        update={
            "tenant_id": context.tenant_id,
            "workspace_id": context.workspace_id,
        }
    )


async def report_problem(
    *,
    context: ProblemReportContext,
    problem_kind: str,
    severity: str = PROBLEM_SEVERITY_ERROR,
    error_code: str = "",
    source_layer: str = "",
    source_component: str = "",
    status: str = PROBLEM_STATUS_DETECTED,
    problem_id: str | None = None,
    event_id: str = "",
    tool_id: str = "",
    application_attributes: ApplicationObservabilityAttributes | None = None,
    artifact_refs: tuple[ObservabilityArtifactReference, ...] = (),
    exporter: ObservabilityExporter | None = None,
    policy: ObservabilityExportPolicy | None = None,
) -> ExportPolicyResult:
    """Build a problem export envelope, apply policy, and export with failure isolation."""
    envelope = build_problem_export_envelope(
        context=context,
        problem_kind=problem_kind,
        severity=severity,
        error_code=error_code,
        source_layer=source_layer,
        source_component=source_component,
        status=status,
        problem_id=problem_id,
        event_id=event_id,
        tool_id=tool_id,
        application_attributes=application_attributes,
        artifact_refs=artifact_refs,
    )
    return await try_export_observability_envelope(
        envelope,
        exporter=exporter,
        policy=policy,
    )


@dataclass(frozen=True, slots=True)
class ProblemReporter:
    """Bound facade for reporting problems with a fixed context and export posture."""

    context: ProblemReportContext
    exporter: ObservabilityExporter | None = None
    policy: ObservabilityExportPolicy | None = None

    async def report(
        self,
        *,
        problem_kind: str,
        severity: str = PROBLEM_SEVERITY_ERROR,
        error_code: str = "",
        source_layer: str = "",
        source_component: str = "",
        status: str = PROBLEM_STATUS_DETECTED,
        problem_id: str | None = None,
        event_id: str = "",
        tool_id: str = "",
        application_attributes: ApplicationObservabilityAttributes | None = None,
        artifact_refs: tuple[ObservabilityArtifactReference, ...] = (),
    ) -> ExportPolicyResult:
        return await report_problem(
            context=self.context,
            problem_kind=problem_kind,
            severity=severity,
            error_code=error_code,
            source_layer=source_layer,
            source_component=source_component,
            status=status,
            problem_id=problem_id,
            event_id=event_id,
            tool_id=tool_id,
            application_attributes=application_attributes,
            artifact_refs=artifact_refs,
            exporter=self.exporter,
            policy=self.policy,
        )
