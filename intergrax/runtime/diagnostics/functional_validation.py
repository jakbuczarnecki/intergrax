# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Functional validation helpers and signal builders (DIAG-FUNCTIONAL-1)."""

from __future__ import annotations

import hashlib

from intergrax.contracts.execution_identity import EventId, validate_event_id
from intergrax.runtime.observability.functional_validation_evidence import (
    DiagnosticExecutionCorrelation,
    ExpectedActualRelation,
    FunctionalValidationEvidence,
    FunctionalValidationKind,
    FunctionalValidationOutcome,
    FunctionalValidatorRef,
    PLATFORM_FUNCTIONAL_VALIDATION_EVIDENCE_SCHEMA,
)
from intergrax.runtime.observability.problem_signal import (
    PROBLEM_KIND_PLATFORM_FUNCTIONAL_OUTCOME_INVALID,
    PROBLEM_SEVERITY_ERROR,
    PROBLEM_STATUS_DETECTED,
    PlatformProblemSignal,
)


class FunctionalValidationIntegrityError(Exception):
    """Raised when functional validation correlation or signal alignment fails."""


def functional_validation_evidence_id(
    *,
    validator_id: str,
    validation_kind: FunctionalValidationKind,
    correlation: DiagnosticExecutionCorrelation,
    idempotency_key: str,
) -> EventId:
    """Deterministic validation identity for duplicate-safe emission."""
    digest = hashlib.sha256(
        ":".join(
            (
                validator_id,
                validation_kind.value,
                correlation.tenant_id,
                str(correlation.task_id),
                str(correlation.run_id),
                idempotency_key,
            )
        ).encode("utf-8")
    ).hexdigest()[:32]
    return validate_event_id(f"evt_{digest}")


def validate_functional_validation_correlation(
    *,
    tenant_id: str,
    correlation: DiagnosticExecutionCorrelation,
) -> DiagnosticExecutionCorrelation:
    """Fail closed when tenant scope does not match correlation."""
    if type(tenant_id) is not str:
        raise FunctionalValidationIntegrityError("tenant_id must be str")
    normalized_tenant = tenant_id.strip()
    if not normalized_tenant:
        raise FunctionalValidationIntegrityError("tenant_id must be non-empty")
    if tenant_id != normalized_tenant:
        raise FunctionalValidationIntegrityError(
            "tenant_id must not contain leading or trailing whitespace",
        )
    if correlation.tenant_id != normalized_tenant:
        raise FunctionalValidationIntegrityError(
            "functional validation correlation tenant_id mismatch",
        )
    return correlation


def validate_problem_signal_correlation_alignment(
    *,
    signal_task_id: str,
    signal_run_id: str,
    correlation: DiagnosticExecutionCorrelation,
) -> None:
    """Fail closed when problem signal identity fields diverge from typed correlation."""
    if signal_task_id and signal_task_id != str(correlation.task_id):
        raise FunctionalValidationIntegrityError(
            "problem signal task_id does not match functional validation correlation",
        )
    if signal_run_id and signal_run_id != str(correlation.run_id):
        raise FunctionalValidationIntegrityError(
            "problem signal run_id does not match functional validation correlation",
        )


def build_functional_outcome_invalid_signal(
    *,
    validation: FunctionalValidationEvidence,
    safe_message: str = "",
    source_layer: str = "",
    source_component: str = "",
    problem_id: str = "",
    correlation_id: str = "",
) -> tuple[FunctionalValidationEvidence, PlatformProblemSignal]:
    """
    Build a canonical functional-outcome-invalid problem signal from validation evidence.

    Execution terminal state remains independent — this signal records domain failure only.
    """
    if validation.outcome is not FunctionalValidationOutcome.FAILED:
        raise FunctionalValidationIntegrityError(
            "functional outcome invalid signal requires FAILED validation outcome",
        )
    validate_functional_validation_correlation(
        tenant_id=validation.correlation.tenant_id,
        correlation=validation.correlation,
    )
    signal = PlatformProblemSignal(
        problem_id=problem_id,
        problem_kind=PROBLEM_KIND_PLATFORM_FUNCTIONAL_OUTCOME_INVALID,
        severity=PROBLEM_SEVERITY_ERROR,
        source_layer=source_layer,
        source_component=source_component,
        status=PROBLEM_STATUS_DETECTED,
        safe_message=safe_message,
        run_id=str(validation.correlation.run_id),
        task_id=str(validation.correlation.task_id),
        event_id=str(validation.correlation.event_id or validation.validation_id),
        correlation_id=correlation_id or str(validation.correlation.task_id),
        functional_validation=validation,
    )
    validate_problem_signal_correlation_alignment(
        signal_task_id=signal.task_id,
        signal_run_id=signal.run_id,
        correlation=validation.correlation,
    )
    return validation, signal


__all__ = [
    "DiagnosticExecutionCorrelation",
    "ExpectedActualRelation",
    "FunctionalValidationEvidence",
    "FunctionalValidationIntegrityError",
    "FunctionalValidationKind",
    "FunctionalValidationOutcome",
    "FunctionalValidatorRef",
    "PLATFORM_FUNCTIONAL_VALIDATION_EVIDENCE_SCHEMA",
    "build_functional_outcome_invalid_signal",
    "functional_validation_evidence_id",
    "validate_functional_validation_correlation",
    "validate_problem_signal_correlation_alignment",
]
