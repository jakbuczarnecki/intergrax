# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Platform problem signal → observability export envelope mapping (OBS-PROBLEM-2)."""

from __future__ import annotations

from intergrax.runtime.observability.export_boundary import (
    ExportRecordKind,
    ExportStatus,
    ObservabilityExportEnvelope,
)
from intergrax.runtime.observability.problem_signal import (
    PLATFORM_PROBLEM_SIGNAL_SCHEMA,
    PROBLEM_SEVERITY_CRITICAL,
    PROBLEM_SEVERITY_ERROR,
    PlatformProblemSignal,
)


def problem_signal_export_status(signal: PlatformProblemSignal) -> ExportStatus:
    """Map problem severity to export status with minimal, plugin-safe semantics."""
    severity = signal.severity.casefold()
    if severity in {PROBLEM_SEVERITY_ERROR, PROBLEM_SEVERITY_CRITICAL}:
        return ExportStatus.FAILED
    return ExportStatus.UNKNOWN


def envelope_from_problem_signal(signal: PlatformProblemSignal) -> ObservabilityExportEnvelope:
    """Map a platform problem signal to a vendor-neutral observability export envelope."""
    counts: dict[str, int] = {}
    artifact_ref = ""
    sha256 = ""
    safe_relative_path = ""
    schema_id = PLATFORM_PROBLEM_SIGNAL_SCHEMA

    if signal.artifact_refs:
        primary = signal.artifact_refs[0]
        artifact_ref = primary.artifact_ref
        sha256 = primary.sha256
        safe_relative_path = primary.safe_relative_path
        if primary.schema_id:
            schema_id = primary.schema_id
        counts["artifact_ref_count"] = len(signal.artifact_refs)

    # Agent attribute export mapping is deferred until the envelope supports
    # multiple typed extension scopes or a safe merge strategy (OBS-PROBLEM-2).
    #
    # Safe human-readable issue summary export via signal.safe_message is deferred
    # to a later problem projection enrichment task.

    return ObservabilityExportEnvelope(
        record_kind=ExportRecordKind.PROBLEM_SIGNAL,
        recorded_at=signal.occurred_at,
        run_id=signal.run_id,
        task_id=signal.task_id,
        agent_id=signal.agent_id,
        capability=signal.capability,
        tool_id=signal.tool_id,
        event_type=signal.problem_kind,
        status=problem_signal_export_status(signal),
        counts=counts,
        artifact_ref=artifact_ref,
        sha256=sha256,
        safe_relative_path=safe_relative_path,
        schema_id=schema_id,
        source_schema_id=PLATFORM_PROBLEM_SIGNAL_SCHEMA,
        correlation_id=signal.correlation_id,
        event_id=signal.problem_id or signal.event_id,
        problem_kind=signal.problem_kind,
        problem_severity=signal.severity,
        problem_error_code=signal.error_code,
        application_attributes=signal.application_attributes,
    )
