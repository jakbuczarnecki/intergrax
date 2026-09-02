# © Artur Czarnecki. All rights reserved.

"""Independent deterministic oracle for Q4 model-routing qualification."""

from __future__ import annotations

from intergrax.runtime.diagnostics.functional_evidence import PipelineEvidenceScope
from intergrax.runtime.diagnostics.functional_validation import (
    DiagnosticExecutionCorrelation,
    ExpectedActualRelation,
    FunctionalValidationEvidence,
    FunctionalValidationKind,
    FunctionalValidationOutcome,
    FunctionalValidatorRef,
    functional_validation_evidence_id,
)

Q4_ORACLE_VALIDATOR_ID = "q4.model.functional_oracle.v1"
EXPECTED_NUMERIC_ANSWER = "42"


def independent_model_oracle_passes(
    *,
    answer: str | None,
    selected_profile_artifact: str | None,
    expected_profile_artifact: str,
) -> bool:
    if selected_profile_artifact != expected_profile_artifact:
        return False
    combined = (answer or "").strip()
    return EXPECTED_NUMERIC_ANSWER in combined


def build_independent_validation_evidence(
    scope: PipelineEvidenceScope,
    *,
    answer: str | None,
    selected_profile_artifact: str | None,
    expected_profile_artifact: str,
    idempotency_key: str,
) -> FunctionalValidationEvidence:
    passed = independent_model_oracle_passes(
        answer=answer,
        selected_profile_artifact=selected_profile_artifact,
        expected_profile_artifact=expected_profile_artifact,
    )
    correlation = DiagnosticExecutionCorrelation(
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        attempt_id=scope.attempt_id,
    )
    validation_id = functional_validation_evidence_id(
        validator_id=Q4_ORACLE_VALIDATOR_ID,
        validation_kind=FunctionalValidationKind.ORACLE_ASSERTION,
        correlation=correlation,
        idempotency_key=idempotency_key,
    )
    return FunctionalValidationEvidence(
        validation_id=validation_id,
        validator=FunctionalValidatorRef(validator_id=Q4_ORACLE_VALIDATOR_ID),
        validation_kind=FunctionalValidationKind.ORACLE_ASSERTION,
        outcome=(
            FunctionalValidationOutcome.PASSED if passed else FunctionalValidationOutcome.FAILED
        ),
        correlation=correlation,
        expected_actual_relation=ExpectedActualRelation.CONTAINS,
    )


__all__ = [
    "EXPECTED_NUMERIC_ANSWER",
    "Q4_ORACLE_VALIDATOR_ID",
    "build_independent_validation_evidence",
    "independent_model_oracle_passes",
]
