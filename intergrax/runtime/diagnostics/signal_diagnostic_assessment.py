# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Bounded non-execution diagnostic assessment from PlatformProblemSignal (HOST-DIAG-2)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.runtime.diagnostics.diagnostic_subject import (
    ApplicationDiagnosticSubjectRef,
    validate_application_diagnostic_subject_ref,
)
from intergrax.runtime.observability.problem_signal import PlatformProblemSignal


class SignalDiagnosticAssessmentIntegrityError(Exception):
    """Raised when signal assessment input violates contracts."""


@dataclass(frozen=True, slots=True)
class SignalDiagnosticFinding:
    """Typed operational problem signal facts — no lifecycle semantics."""

    problem_kind: str
    severity: str
    source_layer: str
    source_component: str
    status: str
    error_code: str | None
    exception_type: str | None


@dataclass(frozen=True, slots=True)
class SignalDiagnosticAssessment:
    """Derived assessment for one application-instance diagnostic subject."""

    tenant_id: str
    application_id: str
    instance_id: str
    findings: tuple[SignalDiagnosticFinding, ...]

    @property
    def subject_ref(self) -> ApplicationDiagnosticSubjectRef:
        return ApplicationDiagnosticSubjectRef(
            tenant_id=self.tenant_id,
            application_id=self.application_id,
            instance_id=self.instance_id,
        )

    @property
    def has_findings(self) -> bool:
        return bool(self.findings)


def _semantic_identifier(value: str, *, field_name: str) -> str:
    if type(value) is not str:
        raise SignalDiagnosticAssessmentIntegrityError(f"{field_name} must be str")
    normalized = value.strip()
    if not normalized:
        raise SignalDiagnosticAssessmentIntegrityError(f"{field_name} must be non-empty")
    return normalized


def _optional_semantic_identifier(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    return normalized


def _finding_from_signal(signal: PlatformProblemSignal) -> SignalDiagnosticFinding:
    return SignalDiagnosticFinding(
        problem_kind=_semantic_identifier(signal.problem_kind, field_name="problem_kind"),
        severity=_semantic_identifier(signal.severity, field_name="severity"),
        source_layer=_semantic_identifier(signal.source_layer, field_name="source_layer"),
        source_component=_semantic_identifier(
            signal.source_component,
            field_name="source_component",
        ),
        status=_semantic_identifier(signal.status, field_name="status"),
        error_code=_optional_semantic_identifier(signal.error_code, field_name="error_code"),
        exception_type=_optional_semantic_identifier(
            signal.exception_type,
            field_name="exception_type",
        ),
    )


class SignalDiagnosticAssessmentBuilder:
    """
    Deterministic bounded assessment over typed PlatformProblemSignal inputs.

    Does not interpret safe_message, logs, or infer root cause.
    """

    def assess(
        self,
        subject_ref: ApplicationDiagnosticSubjectRef,
        problem_signals: tuple[PlatformProblemSignal, ...],
    ) -> SignalDiagnosticAssessment:
        validated_subject = validate_application_diagnostic_subject_ref(subject_ref)
        if type(problem_signals) is not tuple:
            raise TypeError("problem_signals must be tuple")

        findings: list[SignalDiagnosticFinding] = []
        for index, signal in enumerate(problem_signals):
            if type(signal) is not PlatformProblemSignal:
                raise TypeError(f"problem_signals[{index}] must be PlatformProblemSignal")
            findings.append(_finding_from_signal(signal))

        return SignalDiagnosticAssessment(
            tenant_id=validated_subject.tenant_id,
            application_id=validated_subject.application_id,
            instance_id=validated_subject.instance_id,
            findings=tuple(findings),
        )
