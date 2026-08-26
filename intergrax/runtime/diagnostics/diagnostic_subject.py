# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Typed tenant-scoped diagnostic subject identity (HOST-DIAG-2)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.contracts.execution_identity import RunId, TaskId, validate_run_id, validate_task_id


class DiagnosticSubjectKind(StrEnum):
    """Explicit discriminator for diagnostic subject domains."""

    EXECUTION = "execution"
    APPLICATION_INSTANCE = "application_instance"


@dataclass(frozen=True, slots=True)
class ExecutionDiagnosticSubjectRef:
    """One tenant/task/run execution diagnostic subject."""

    tenant_id: str
    task_id: TaskId
    run_id: RunId

    @property
    def kind(self) -> DiagnosticSubjectKind:
        return DiagnosticSubjectKind.EXECUTION


@dataclass(frozen=True, slots=True)
class ApplicationDiagnosticSubjectRef:
    """One tenant/application/instance hosted lifecycle diagnostic subject."""

    tenant_id: str
    application_id: str
    instance_id: str

    @property
    def kind(self) -> DiagnosticSubjectKind:
        return DiagnosticSubjectKind.APPLICATION_INSTANCE


DiagnosticSubjectRef = ExecutionDiagnosticSubjectRef | ApplicationDiagnosticSubjectRef


def _require_tenant_id(tenant_id: str) -> str:
    if type(tenant_id) is not str:
        raise TypeError("tenant_id must be str")
    normalized = tenant_id.strip()
    if not normalized:
        raise ValueError("tenant_id must be non-empty and not whitespace-only")
    if tenant_id != normalized:
        raise ValueError("tenant_id must not contain leading or trailing whitespace")
    return normalized


def _require_semantic_identifier(value: str, *, field_name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be str")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty and not whitespace-only")
    if value != normalized:
        raise ValueError(f"{field_name} must not contain leading or trailing whitespace")
    return normalized


def validate_execution_diagnostic_subject_ref(
    subject: ExecutionDiagnosticSubjectRef,
) -> ExecutionDiagnosticSubjectRef:
    tenant_id = _require_tenant_id(subject.tenant_id)
    task_id = validate_task_id(subject.task_id)
    run_id = validate_run_id(subject.run_id)
    if tenant_id != subject.tenant_id or task_id != subject.task_id or run_id != subject.run_id:
        return ExecutionDiagnosticSubjectRef(
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
        )
    return subject


def validate_application_diagnostic_subject_ref(
    subject: ApplicationDiagnosticSubjectRef,
) -> ApplicationDiagnosticSubjectRef:
    tenant_id = _require_tenant_id(subject.tenant_id)
    application_id = _require_semantic_identifier(
        subject.application_id,
        field_name="application_id",
    )
    instance_id = _require_semantic_identifier(
        subject.instance_id,
        field_name="instance_id",
    )
    if (
        tenant_id != subject.tenant_id
        or application_id != subject.application_id
        or instance_id != subject.instance_id
    ):
        return ApplicationDiagnosticSubjectRef(
            tenant_id=tenant_id,
            application_id=application_id,
            instance_id=instance_id,
        )
    return subject


def diagnostic_subject_index_token(subject: DiagnosticSubjectRef) -> str:
    """Stable persistence index token for one diagnostic subject (tenant is partition boundary)."""
    if type(subject) is ExecutionDiagnosticSubjectRef:
        validated = validate_execution_diagnostic_subject_ref(subject)
        return (
            f"{DiagnosticSubjectKind.EXECUTION.value}:"
            f"{validated.task_id}:{validated.run_id}"
        )
    if type(subject) is ApplicationDiagnosticSubjectRef:
        validated = validate_application_diagnostic_subject_ref(subject)
        return (
            f"{DiagnosticSubjectKind.APPLICATION_INSTANCE.value}:"
            f"{validated.application_id}:{validated.instance_id}"
        )
    raise TypeError(f"unsupported diagnostic subject type: {type(subject).__name__}")
