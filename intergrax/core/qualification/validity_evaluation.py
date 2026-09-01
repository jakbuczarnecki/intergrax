# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Provider qualification evidence validity evaluation (PROVIDER-QUAL-5)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from intergrax.core.qualification.provider import ProviderQualificationRun
from intergrax.core.qualification.validity import (
    ProviderQualificationValidityContext,
    QualificationEvidenceValidity,
    QualificationRunId,
    QualificationValidityInterpretation,
    QualificationValidityRecord,
    ValidityEvaluationId,
    new_validity_evaluation_id,
    validate_qualification_run_id,
    validate_validity_evaluation_id,
)


class QualificationValidityEstablishmentError(Exception):
    """Raised when current validity cannot be established safely."""


class QualificationValidityNotFoundError(Exception):
    """Raised when no validity evaluations exist for a qualification run."""


def validity_context_from_run(run: ProviderQualificationRun) -> ProviderQualificationValidityContext:
    """Build a validity context from the immutable run subject and source anchor."""
    subject = run.subject
    return ProviderQualificationValidityContext(
        provider_id=subject.provider_id,
        provider_version=subject.provider_version,
        capability_id=subject.capability_id,
        domain=subject.domain,
        intergrax_revision=subject.intergrax_revision,
        qualification_suite_id=subject.qualification_suite_id,
        qualification_suite_version=subject.qualification_suite_version,
        environment_id=subject.environment_id,
        source_revision=run.source_revision,
        adapter_identity=subject.adapter_identity,
    )


def evaluate_provider_qualification_validity(
    run: ProviderQualificationRun,
    current_context: ProviderQualificationValidityContext,
    *,
    evaluated_at: datetime,
    validity_evaluation_id: ValidityEvaluationId | None = None,
) -> QualificationValidityRecord:
    """
    Compare immutable run evidence scope against current platform facts.

    Returns CURRENT when all identity-defining and freshness-defining dimensions match.
    Returns STALE on the first explicit drift dimension mismatch.
    """
    if not isinstance(run, ProviderQualificationRun):
        raise TypeError("run must be ProviderQualificationRun")
    if not isinstance(current_context, ProviderQualificationValidityContext):
        raise TypeError("current_context must be ProviderQualificationValidityContext")

    evaluation_id = validity_evaluation_id or new_validity_evaluation_id()
    validate_validity_evaluation_id(evaluation_id)

    subject = run.subject
    drift_checks: tuple[tuple[str, str, str], ...] = (
        ("provider_id", subject.provider_id, current_context.provider_id),
        ("provider_version", subject.provider_version, current_context.provider_version),
        ("capability_id", subject.capability_id, current_context.capability_id),
        ("domain", subject.domain, current_context.domain),
        (
            "qualification_suite_id",
            subject.qualification_suite_id,
            current_context.qualification_suite_id,
        ),
        ("environment_id", subject.environment_id, current_context.environment_id),
        (
            "intergrax_revision",
            subject.intergrax_revision,
            current_context.intergrax_revision,
        ),
        (
            "qualification_suite_version",
            subject.qualification_suite_version,
            current_context.qualification_suite_version,
        ),
        ("source_revision", run.source_revision, current_context.source_revision),
    )

    for field_name, recorded, current in drift_checks:
        if recorded != current:
            return QualificationValidityRecord(
                qualification_run_id=run.qualification_run_id,
                validity_evaluation_id=evaluation_id,
                validity=QualificationEvidenceValidity.STALE,
                evaluated_at=evaluated_at,
                reason=f"{field_name}_changed",
                evaluation_context=current_context,
            )

    if subject.adapter_identity != current_context.adapter_identity:
        return QualificationValidityRecord(
            qualification_run_id=run.qualification_run_id,
            validity_evaluation_id=evaluation_id,
            validity=QualificationEvidenceValidity.STALE,
            evaluated_at=evaluated_at,
            reason="adapter_identity_changed",
            evaluation_context=current_context,
        )

    return QualificationValidityRecord(
        qualification_run_id=run.qualification_run_id,
        validity_evaluation_id=evaluation_id,
        validity=QualificationEvidenceValidity.CURRENT,
        evaluated_at=evaluated_at,
        evaluation_context=current_context,
    )


def record_provider_qualification_validity_revocation(
    qualification_run_id: QualificationRunId | str,
    *,
    reason: str,
    evaluated_at: datetime,
    validity_evaluation_id: ValidityEvaluationId | None = None,
    evaluation_context: ProviderQualificationValidityContext | None = None,
) -> QualificationValidityRecord:
    """Append an explicit REVOKED validity evaluation without mutating the historical run."""
    validated_run_id = validate_qualification_run_id(qualification_run_id)
    evaluation_id = validity_evaluation_id or new_validity_evaluation_id()
    validate_validity_evaluation_id(evaluation_id)
    return QualificationValidityRecord(
        qualification_run_id=validated_run_id,
        validity_evaluation_id=evaluation_id,
        validity=QualificationEvidenceValidity.REVOKED,
        evaluated_at=evaluated_at,
        reason=reason,
        evaluation_context=evaluation_context,
    )


def resolve_latest_qualification_validity(
    records: Sequence[QualificationValidityRecord],
) -> QualificationValidityRecord | None:
    """Return the latest validity evaluation; None when no records exist."""
    if not records:
        return None
    return max(
        records,
        key=lambda record: (record.evaluated_at, str(record.validity_evaluation_id)),
    )


def interpret_latest_qualification_validity(
    qualification_run_id: QualificationRunId | str,
    records: Sequence[QualificationValidityRecord],
) -> QualificationValidityInterpretation:
    """Derive the current validity view from append-only evaluation history."""
    validated_run_id = validate_qualification_run_id(qualification_run_id)
    scoped = tuple(
        record
        for record in records
        if record.qualification_run_id == validated_run_id
    )
    latest = resolve_latest_qualification_validity(scoped)
    if latest is None:
        raise QualificationValidityNotFoundError(
            f"no validity evaluations for qualification_run_id {validated_run_id!s}",
        )
    return QualificationValidityInterpretation(
        qualification_run_id=validated_run_id,
        validity=latest.validity,
        latest_record=latest,
    )


def get_current_qualification_validity(
    qualification_run_id: QualificationRunId | str,
    records: Sequence[QualificationValidityRecord],
) -> QualificationValidityInterpretation:
    """Resolve the latest validity interpretation for one qualification run."""
    return interpret_latest_qualification_validity(qualification_run_id, records)


def establish_current_qualification_validity(
    qualification_run_id: QualificationRunId | str,
    records: Sequence[QualificationValidityRecord],
) -> QualificationValidityInterpretation:
    """
    Fail-closed validity establishment.

    Raises when records are corrupt, scoped to another run, or missing.
    """
    validated_run_id = validate_qualification_run_id(qualification_run_id)
    for record in records:
        if record.qualification_run_id != validated_run_id:
            raise QualificationValidityEstablishmentError(
                "validity record qualification_run_id mismatch",
            )
    try:
        return interpret_latest_qualification_validity(validated_run_id, records)
    except QualificationValidityNotFoundError as exc:
        raise QualificationValidityEstablishmentError(str(exc)) from exc
